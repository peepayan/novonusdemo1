"""Stage 5 augmentation engine.

Re-simulates one baseline demo under one sampled :class:`AugmentationParams`.
The arm motion is fixed (we replay the baseline's saved joint-position
trajectory through MuJoCo's position actuators), so the only thing that
varies between an augmentation and its baseline is the *physics under the
perturbed scene parameters*. This is what makes the resulting dataset a
real, physics-grounded augmentation rather than synthetic noise.

Design notes:
- Per scenario we rebuild ``mujoco_warp``'s GPU model + data after mutating
  the CPU ``MjModel`` with the sampled params (mass, friction, stiffness,
  lighting). This is the only path that lets per-scenario parameters
  actually take effect, since ``put_model`` snapshots the host model.
- We reuse one renderer instance across scenarios — the renderer reads
  from the CPU MjData which we re-sync from the GPU each timestep, so the
  rendered frames reflect the perturbed physics.
- Frames stream out to a per-scenario MP4 (h264, ~1.4 MB) so the disk
  footprint stays bounded as the dataset grows. ~80 GB of PNGs for 4000
  scenarios is not feasible on this hardware.
- The weld equality that "glues" the grape to the gripper TCP during
  transit is replayed from the baseline's ``intent_labels`` — it activates
  at the GRIPPING→STABILIZING transition and deactivates at the
  RELEASING→REST transition. Without replaying the weld, almost every
  scenario would fail because the friction-only grasp slips out under
  perturbed contact dynamics, and we'd be measuring controller robustness
  instead of physical-condition robustness.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import mujoco
import numpy as np

from ..stage3_sim import physics_constants as pc
from ..stage3_sim.warp_backend import WarpBackend
from .params import AugmentationParams


_ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
)
_JAW_ACTUATOR_NAMES: tuple[str, ...] = ("left_jaw_act", "right_jaw_act")

# Intent label IDs are duplicated here (rather than imported) to avoid a
# cross-stage import cycle. The mapping is frozen in physics_constants.
_INTENT_GRIPPING: int = 2
_INTENT_STABILIZING: int = 3
_INTENT_RELEASING: int = 4
_INTENT_REST: int = 0


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------

@dataclass
class BaselineDemo:
    """Subset of a Stage 4 demo needed for replay + verification."""
    demo_idx: int
    joint_positions: np.ndarray        # (T, 6) float32 — replayed as ctrl
    gripper_state: np.ndarray          # (T,)  0=closed, 1=open
    end_effector_pos: np.ndarray       # (T, 3) — baseline EE for Check 1
    intent_labels: np.ndarray          # (T,) int — for weld replay
    applied_force: np.ndarray          # (T,) — baseline force trace for Check 5
    grape_initial_pos: np.ndarray      # (3,)
    metadata: dict[str, Any]


def load_baseline_demos(demos_dir: Path) -> list[BaselineDemo]:
    out: list[BaselineDemo] = []
    for p in sorted(Path(demos_dir).glob("demo_*.npz")):
        z = np.load(p, allow_pickle=False)
        idx = int(p.stem.split("_")[1])
        out.append(BaselineDemo(
            demo_idx=idx,
            joint_positions=np.asarray(z["joint_positions"], dtype=np.float32),
            gripper_state=np.asarray(z["gripper_state"], dtype=np.float32),
            end_effector_pos=np.asarray(z["end_effector_pos"], dtype=np.float32),
            intent_labels=np.asarray(z["intent_labels"], dtype=np.int32),
            applied_force=np.asarray(z["applied_force"], dtype=np.float32),
            grape_initial_pos=np.asarray(z["grape_initial_pos"], dtype=np.float32),
            metadata=json.loads(str(z["metadata"])),
        ))
    return out


def derive_weld_active_timeline(intent_labels: np.ndarray) -> np.ndarray:
    """Return a (T,) bool array — True where the gripper weld should be
    active. Active from the GRIPPING→STABILIZING transition through the
    last RELEASING frame.

    Matches the Stage 4 controller's behavior closely enough that replayed
    physics looks identical to the baseline when no params are perturbed.
    """
    T = int(intent_labels.shape[0])
    active = np.zeros(T, dtype=bool)

    # Find first STABILIZING frame (weld on)
    stab_idxs = np.where(intent_labels == _INTENT_STABILIZING)[0]
    if stab_idxs.size == 0:
        return active
    on_idx = int(stab_idxs[0])

    # Find last RELEASING frame (weld off after that)
    rel_idxs = np.where(intent_labels == _INTENT_RELEASING)[0]
    off_idx = int(rel_idxs[-1]) if rel_idxs.size else T - 1

    active[on_idx:off_idx + 1] = True
    return active


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AugmentationEngine:
    """Per-scenario simulator. Reuses one WarpBackend instance across
    scenarios — the backend is reset and its GPU model + data are rebuilt
    from a mutated CPU model on every ``simulate``."""

    def __init__(self) -> None:
        # one backend instance — we re-upload the GPU model + data per
        # scenario via _apply_params (cheap relative to the per-scenario
        # step loop)
        self.backend = WarpBackend()
        self.model = self.backend.model     # mutable CPU MjModel
        self.data = self.backend.data
        # cache ids
        self._grape_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "grape")
        self._grape_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "grape_geom")
        self._table_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self._tcp_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._grape_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "grape_free")
        self._grape_qadr = int(self.model.jnt_qposadr[self._grape_jid])
        self._grape_dadr = int(self.model.jnt_dofadr[self._grape_jid])
        self._weld_eq = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grape_grip")
        self._aid = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in _ARM_ACTUATOR_NAMES + _JAW_ACTUATOR_NAMES
        }
        missing = [n for n, i in self._aid.items() if i < 0]
        if missing:
            raise RuntimeError(
                f"actuators not found in scene: {missing}. Available: "
                f"{[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]}")
        # snapshot nominal values so each scenario starts from a clean slate
        self._nominal_grape_mass = float(self.model.body_mass[self._grape_body])
        self._nominal_grape_inertia = self.model.body_inertia[
            self._grape_body].copy()
        self._nominal_grape_friction = self.model.geom_friction[
            self._grape_geom].copy()
        self._nominal_grape_solref = self.model.geom_solref[
            self._grape_geom].copy()
        self._nominal_table_friction = self.model.geom_friction[
            self._table_geom].copy()
        self._nominal_headlight_ambient = self.model.vis.headlight.ambient.copy()
        self._nominal_headlight_diffuse = self.model.vis.headlight.diffuse.copy()

        # renderer (shared)
        self._renderer = mujoco.Renderer(
            self.model, height=pc.RENDER_H, width=pc.RENDER_W)
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, pc.CAMERA_NAME)

    # ------------------------------------------------------------------

    def _apply_params(self, params: AugmentationParams) -> None:
        """Mutate the CPU ``MjModel`` to encode this scenario's params,
        then re-upload to GPU. Lighting is *not* applied here — recreating
        the ``mujoco.Renderer`` after ``warp.init`` corrupts the GL state
        and produces all-black frames, so visual lighting jitter is
        applied at render time via ``scene.lights[]`` instead.
        """
        # Grape mass (scale inertia proportionally so the dynamics stay
        # consistent — a heavier grape resists motion more, etc.)
        scale = params.grape_mass_kg / self._nominal_grape_mass
        self.model.body_mass[self._grape_body] = float(params.grape_mass_kg)
        self.model.body_inertia[self._grape_body] = (
            self._nominal_grape_inertia * scale)

        # Grape-jaw friction (jaws keep their nominal 1.0; grape's coeff
        # is the binding factor because MuJoCo combines per-geom friction
        # by min/avg).
        new_grape_fric = self._nominal_grape_friction.copy()
        new_grape_fric[0] = float(params.grape_jaw_friction)
        self.model.geom_friction[self._grape_geom] = new_grape_fric

        # Contact stiffness — modulate solref[0] (response time).
        new_solref = self._nominal_grape_solref.copy()
        new_solref[0] = float(params.grape_solref0)
        self.model.geom_solref[self._grape_geom] = new_solref

        # Table friction
        new_table_fric = self._nominal_table_friction.copy()
        new_table_fric[0] = float(params.table_friction0)
        self.model.geom_friction[self._table_geom] = new_table_fric

        # Re-upload to GPU.
        self.backend.m_gpu = self.backend._mjwp.put_model(self.model)

    def _apply_lighting(self, params: AugmentationParams) -> None:
        """Per-render lighting adjustment via mjvScene.lights — does not
        require recreating the Renderer."""
        scene = self._renderer.scene
        n = int(scene.nlight)
        for i in range(n):
            light = scene.lights[i]
            for c in range(3):
                light.ambient[c] = float(np.clip(
                    self._nominal_headlight_ambient[c]
                    * float(params.light_ambient_scale), 0.0, 1.0))
                light.diffuse[c] = float(np.clip(
                    self._nominal_headlight_diffuse[c]
                    * float(params.light_ambient_scale), 0.0, 1.0))

    def _reset_for_scenario(self, baseline: BaselineDemo,
                            params: AugmentationParams) -> None:
        """Reset state to the baseline's initial pose, then offset the
        grape per params. The ``home`` keyframe already places the arm at
        HOVER and the jaws open, so we only need to override the grape."""
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Grape pose = baseline initial pos + per-scenario XY offset, plus
        # the sampled yaw quaternion.
        gx = float(baseline.grape_initial_pos[0]
                   + params.grape_xy_offset_m[0])
        gy = float(baseline.grape_initial_pos[1]
                   + params.grape_xy_offset_m[1])
        gz = float(baseline.grape_initial_pos[2])
        yaw = float(params.grape_yaw_rad)
        qw = float(np.cos(yaw / 2.0))
        qz = float(np.sin(yaw / 2.0))
        self.data.qpos[self._grape_qadr:self._grape_qadr + 3] = (gx, gy, gz)
        self.data.qpos[self._grape_qadr + 3:self._grape_qadr + 7] = (
            qw, 0.0, 0.0, qz)
        self.data.qvel[self._grape_dadr:self._grape_dadr + 6] = 0.0

        # Open jaws, deactivate weld
        self.data.ctrl[:] = 0.0
        if self._weld_eq >= 0:
            self.data.eq_active[self._weld_eq] = 0

        mujoco.mj_forward(self.model, self.data)
        self.backend.d_gpu = self.backend._mjwp.put_data(
            self.model, self.data)

    # ------------------------------------------------------------------

    def _set_weld(self, active: bool) -> None:
        if self._weld_eq < 0:
            return
        if active and not bool(self.data.eq_active[self._weld_eq]):
            # snap grape to TCP first (matches Stage 4)
            tcp = np.asarray(self.data.site_xpos[self._tcp_sid]).copy()
            self.data.qpos[self._grape_qadr:self._grape_qadr + 3] = tcp
            self.data.qpos[self._grape_qadr + 3:self._grape_qadr + 7] = (
                1, 0, 0, 0)
            self.data.qvel[self._grape_dadr:self._grape_dadr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
        self.data.eq_active[self._weld_eq] = 1 if active else 0

    # ------------------------------------------------------------------

    def _jaw_contact_force_N(self) -> float:
        """Magnitude of normal contact force between grape and either jaw,
        divided by 2 so it reads as 'force on grape'."""
        g = self._grape_geom
        lid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_jaw_geom")
        rid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_jaw_geom")
        total = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g in (g1, g2) and (lid in (g1, g2) or rid in (g1, g2)):
                f6 = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, c, f6)
                total += float(abs(f6[0]))
        return total / 2.0

    # ------------------------------------------------------------------

    def simulate(self, baseline: BaselineDemo,
                 params: AugmentationParams,
                 mp4_path: Path) -> dict[str, Any]:
        """Run one scenario end-to-end. Frames stream to ``mp4_path``."""
        self._apply_params(params)
        self._reset_for_scenario(baseline, params)

        weld_active_timeline = derive_weld_active_timeline(
            baseline.intent_labels)

        traj = baseline.joint_positions          # (T, 6)
        gstate = baseline.gripper_state          # (T,)
        T = int(traj.shape[0])

        dt = pc.SIM_TIMESTEP_S
        physics_per_frame = max(
            1, int(round((1.0 / pc.RENDER_FPS) / dt)))

        ee_pos = np.zeros((T, 3), dtype=np.float32)
        joint_pos = np.zeros((T, 6), dtype=np.float32)
        joint_vel = np.zeros((T, 6), dtype=np.float32)
        grape_pos = np.zeros((T, 3), dtype=np.float32)
        contact_N = np.zeros((T,), dtype=np.float32)
        gripper_out = np.zeros((T,), dtype=np.float32)

        mp4_path = Path(mp4_path)
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        writer = iio.get_writer(
            str(mp4_path), fps=pc.RENDER_FPS, codec="libx264",
            quality=None, bitrate="900k", macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        try:
            weld_state = False
            for t in range(T):
                # weld replay
                want_weld = bool(weld_active_timeline[t])
                if want_weld != weld_state:
                    self._set_weld(want_weld)
                    # the weld snap mutates data.qpos directly; re-upload
                    self.backend.d_gpu = self.backend._mjwp.put_data(
                        self.model, self.data)
                    weld_state = want_weld

                # Build ctrl from baseline arm trajectory + sampled
                # approach offset on wrist_3, and derive jaw ctrl from
                # gripper_state.
                arm_q = traj[t].copy()
                arm_q[5] = float(arm_q[5] + params.approach_angle_rad)
                jaw_ctrl = float((1.0 - float(gstate[t])) * 0.045)

                ctrl = np.zeros(self.model.nu, dtype=np.float64)
                for q, name in zip(arm_q, _ARM_ACTUATOR_NAMES):
                    ctrl[self._aid[name]] = float(q)
                ctrl[self._aid["left_jaw_act"]] = jaw_ctrl
                ctrl[self._aid["right_jaw_act"]] = jaw_ctrl

                self.backend.set_ctrl(ctrl)
                for _ in range(physics_per_frame):
                    self.backend.step()

                # readouts
                ee_pos[t] = np.asarray(
                    self.data.site_xpos[self._tcp_sid], dtype=np.float32)
                grape_pos[t] = np.asarray(
                    self.data.xpos[self._grape_body], dtype=np.float32)
                contact_N[t] = self._jaw_contact_force_N()
                gripper_out[t] = float(gstate[t])      # commanded gripper
                # achieved joint state
                for ji, name in enumerate(_ARM_ACTUATOR_NAMES):
                    qadr = int(self.model.jnt_qposadr[
                        self.model.actuator_trnid[self._aid[name], 0]])
                    dadr = int(self.model.jnt_dofadr[
                        self.model.actuator_trnid[self._aid[name], 0]])
                    joint_pos[t, ji] = float(self.data.qpos[qadr])
                    joint_vel[t, ji] = float(self.data.qvel[dadr])

                self._renderer.update_scene(self.data, camera=self._cam_id)
                self._apply_lighting(params)
                frame = self._renderer.render()
                writer.append_data(frame)
        finally:
            writer.close()

        # final grape pose (post-release)
        final_grape = np.asarray(
            self.data.xpos[self._grape_body], dtype=np.float32).copy()

        return {
            "joint_positions": joint_pos,
            "joint_velocities": joint_vel,
            "end_effector_pos": ee_pos,
            "gripper_state": gripper_out,
            "contact_force": contact_N,
            "grape_positions": grape_pos,
            "grape_initial_pos": np.array(
                [baseline.grape_initial_pos[0] + params.grape_xy_offset_m[0],
                 baseline.grape_initial_pos[1] + params.grape_xy_offset_m[1],
                 baseline.grape_initial_pos[2]], dtype=np.float32),
            "grape_final_xyz": final_grape,
            "frames_mp4": str(mp4_path.name),
            "T": T,
            "duration_s": float(T) / float(pc.RENDER_FPS),
        }


# ---------------------------------------------------------------------------
# Scenario save format
# ---------------------------------------------------------------------------

def save_scenario_npz(
    *,
    recorded: dict[str, Any],
    params: AugmentationParams,
    verification: dict[str, Any],
    baseline_demo_idx: int,
    scenario_idx: int,
    out_path: Path,
) -> Path:
    """Save one passing scenario in a Stage 4-compatible layout."""
    metadata = {
        "scenario_idx": int(scenario_idx),
        "baseline_demo_idx": int(baseline_demo_idx),
        "timestamp_unix": time.time(),
        "stage": "stage5_augmented",
        "note": ("Physics-simulated re-run of Stage 4 baseline under "
                 "perturbed conditions. Joint trajectory was replayed; "
                 "only the physical scene parameters were varied."),
        "render_fps": int(pc.RENDER_FPS),
        "duration_s": float(recorded["duration_s"]),
        "num_timesteps": int(recorded["T"]),
        "crush_threshold_N": float(pc.GRAPE_CRUSH_THRESHOLD_N),
    }
    save = {
        "joint_positions": recorded["joint_positions"],
        "joint_velocities": recorded["joint_velocities"],
        "end_effector_pos": recorded["end_effector_pos"],
        "gripper_state": recorded["gripper_state"],
        "contact_force": recorded["contact_force"],
        "grape_positions": recorded["grape_positions"],
        "grape_initial_pos": recorded["grape_initial_pos"],
        "grape_final_xyz": recorded["grape_final_xyz"],
        "frames_mp4": np.array(str(recorded["frames_mp4"])),
        "baseline_demo_idx": np.int32(baseline_demo_idx),
        "success": np.bool_(True),
        "augmentation_params": np.array(json.dumps(params.to_dict())),
        "verification_results": np.array(json.dumps(verification)),
        "metadata": np.array(json.dumps(metadata)),
    }
    out_path = Path(out_path); out_path.parent.mkdir(
        parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save)
    return out_path


__all__ = [
    "BaselineDemo", "load_baseline_demos", "derive_weld_active_timeline",
    "AugmentationEngine", "save_scenario_npz",
]

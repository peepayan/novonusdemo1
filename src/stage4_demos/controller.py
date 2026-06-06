"""Scripted pick-and-place controller for Stage 4.

Each demonstration is one full cycle through the same four phases used by
the Stage 3 synced demo (REACHING -> GRIPPING -> STABILIZING -> RELEASING),
but driven by deterministic interpolation between IK-resolved joint
waypoints rather than the LSTM state machine. Small controlled variations
between demonstrations (grape XY position, approach angle around the
gripper axis, gripper close speed) give Stage 5's augmentation a richer
seed without making any demo fail.

The grasp itself reuses the same ``grape_grip`` weld equality from
``scene.xml`` that made the Stage 3 synced demo land the grape in the
target zone: the controller activates the weld a short settle after the
gripper closes, holds it through the lift and transit, and deactivates it
once the arm arrives at the release waypoint above the target zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import mujoco
import numpy as np

from ..stage3_sim import physics_constants as pc
from ..stage3_sim.arm_controller import (
    _ARM_NAMES, _JAW_NAMES, jaw_close_ctrl_for_force,
)
from ..stage3_sim.ik import solve_ik


PHASE_NAMES_ORDER: tuple[str, ...] = (
    "REST", "REACHING", "GRIPPING", "STABILIZING", "RELEASING",
)

# Mapping demo phases -> Stage 2 intent class ids
INTENT_ID: dict[str, int] = {
    "REST": 0, "REACHING": 1, "GRIPPING": 2, "STABILIZING": 3, "RELEASING": 4,
}


@dataclass
class DemoParams:
    """Per-demo controlled variation."""
    demo_id: int
    grape_xy_offset: tuple[float, float]      # (+- ~3 cm around table centre)
    approach_angle_rad: float                 # +- 10 deg = +- 0.175 rad
    close_speed_halflife: float               # 0.05 .. 0.15 s
    grip_force_N: float                       # 2-4 N
    durations_s: dict[str, float] = field(default_factory=lambda: {
        "REST_PRE":    0.5,
        "REACHING":    3.0,
        "GRIPPING":    3.0,
        "STABILIZING": 2.5,
        "RELEASING":   4.0,
        "REST_POST":   1.0,
    })


def sample_demo_params(demo_id: int, rng: np.random.Generator) -> DemoParams:
    """Sample controlled variation for one demo."""
    # disk sample within ~3 cm
    r = float(np.sqrt(rng.uniform(0.0, 1.0)) * 0.030)
    th = float(rng.uniform(0.0, 2.0 * np.pi))
    grape_offset = (r * np.cos(th), r * np.sin(th))
    approach = float(rng.uniform(-np.deg2rad(10.0), np.deg2rad(10.0)))
    close_hl = float(rng.uniform(0.05, 0.15))
    grip_force = float(rng.uniform(
        pc.GRAPE_SAFE_GRIP_MIN_N + 0.2, pc.GRAPE_SAFE_GRIP_MAX_N - 0.2,
    ))
    return DemoParams(
        demo_id=demo_id,
        grape_xy_offset=grape_offset,
        approach_angle_rad=approach,
        close_speed_halflife=close_hl,
        grip_force_N=grip_force,
    )


# ---------------------------------------------------------------------------

def _grape_xyz(params: DemoParams) -> np.ndarray:
    cx, cy = pc.GRAPE_INIT_XY
    return np.array(
        [cx + params.grape_xy_offset[0],
         cy + params.grape_xy_offset[1],
         pc.GRAPE_INIT_Z], dtype=np.float64)


def _ik_waypoints(model, data,
                  params: DemoParams) -> dict[str, np.ndarray]:
    """Return joint-space waypoints for one demo (6 joints each)."""
    grape = _grape_xyz(params)
    targets = {
        "HOVER":      grape + np.array([0.0, 0.0, 0.18]),
        "PREGRIP":    grape + np.array([0.0, 0.0, 0.07]),
        "GRIP":       grape + np.array([0.0, 0.0, 0.0]),
        "STABILIZE":  grape + np.array([0.0, 0.0, 0.10]),
        # release pose: low over target zone (matches Stage 3 working value)
        "RELEASE":    np.array(
            [pc.TARGET_ZONE_XY[0], pc.TARGET_ZONE_XY[1], 0.78]),
    }
    seeds = {
        "HOVER":     np.array(pc.ARM_HOVER_QPOS, dtype=np.float64),
        "PREGRIP":   np.array(pc.ARM_PREGRIP_QPOS, dtype=np.float64),
        "GRIP":      np.array(pc.ARM_GRIP_QPOS, dtype=np.float64),
        "STABILIZE": np.array(pc.ARM_STABILIZE_QPOS, dtype=np.float64),
        "RELEASE":   np.array(pc.ARM_RELEASE_QPOS, dtype=np.float64),
    }
    out: dict[str, np.ndarray] = {}
    prev = seeds["HOVER"]
    for name in ("HOVER", "PREGRIP", "GRIP", "STABILIZE", "RELEASE"):
        r = solve_ik(model, data, targets[name], start_qpos=prev,
                     max_iters=200)
        out[name] = r.qpos.copy()
        prev = r.qpos
    # apply approach-angle variation: rotate wrist_3 around its own axis
    # (this is the only joint that affects the gripper roll). Stage 4 demos
    # use this to spin the jaws within +- 10 deg while still landing the
    # grasp on the same grape position.
    for name in ("PREGRIP", "GRIP", "STABILIZE"):
        out[name][5] = float(params.approach_angle_rad)
    return out


def _smoothstep_interp(q_a: np.ndarray, q_b: np.ndarray,
                       s: float) -> np.ndarray:
    """Smoothstep (cubic ease-in-out) joint-space interpolation s in [0, 1]."""
    s = float(max(0.0, min(1.0, s)))
    t = s * s * (3.0 - 2.0 * s)
    return q_a + (q_b - q_a) * t


# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    t_s: float
    phase: str
    intent_id: int
    arm_q: np.ndarray             # (6,)
    arm_qvel: np.ndarray          # (6,)
    ee_pos: np.ndarray            # (3,)
    ee_quat: np.ndarray           # (4,)
    gripper_state: float          # 0..1 (0 closed, 1 open)
    applied_force_N: float
    contact_force_N: float
    weld_active: bool


class ScriptedController:
    """Phase-scripted controller producing one full pick-and-place demo."""

    def __init__(self, backend, params: DemoParams):
        self.backend = backend
        self.params = params
        self.model = backend.model
        self.data = backend.data
        # actuator ids
        self._aid = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in _ARM_NAMES + _JAW_NAMES
        }
        # arm joint qpos addresses (for placing the grape under the gripper)
        self._grape_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "grape_free")
        self._grape_qadr = int(self.model.jnt_qposadr[self._grape_jid])
        self._grape_dadr = int(self.model.jnt_dofadr[self._grape_jid])
        # wrist body id (for ee pose)
        self._wrist_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        # tcp site id
        self._tcp_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        # weld equality id
        self._grip_eq_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grape_grip")
        # arm-joint qpos addresses
        self._arm_qadr = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
            for jn in ("shoulder_pan_joint", "shoulder_lift_joint",
                       "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                       "wrist_3_joint")
        ]

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Place the grape at the demo's randomized initial position and put
        the arm at HOVER."""
        # keyframe reset
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            mujoco.mj_resetData(self.model, self.data)
        # move grape
        gxyz = _grape_xyz(self.params)
        self.data.qpos[self._grape_qadr:self._grape_qadr + 3] = gxyz
        self.data.qpos[self._grape_qadr + 3:self._grape_qadr + 7] = (1, 0, 0, 0)
        self.data.qvel[self._grape_dadr:self._grape_dadr + 6] = 0.0
        # arm at HOVER
        hover = list(pc.ARM_HOVER_QPOS)
        for v, ad in zip(hover, self._arm_qadr):
            self.data.qpos[ad] = float(v)
        # jaws open
        self.data.ctrl[:] = 0.0
        for v, name in zip(hover, _ARM_NAMES):
            self.data.ctrl[self._aid[name]] = float(v)
        # deactivate weld
        if self._grip_eq_id >= 0:
            self.data.eq_active[self._grip_eq_id] = 0
        mujoco.mj_forward(self.model, self.data)
        # propagate to GPU
        self.backend.d_gpu = self.backend._mjwp.put_data(self.model, self.data)

    # ------------------------------------------------------------------

    def _set_grip_weld(self, active: bool) -> None:
        if self._grip_eq_id < 0:
            return
        if active:
            # snap grape to TCP first
            tcp = np.asarray(self.data.site_xpos[self._tcp_sid]).copy()
            self.data.qpos[self._grape_qadr:self._grape_qadr + 3] = tcp
            self.data.qpos[self._grape_qadr + 3:self._grape_qadr + 7] = (
                1, 0, 0, 0)
            self.data.qvel[self._grape_dadr:self._grape_dadr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self.data.eq_active[self._grip_eq_id] = 1
        else:
            self.data.eq_active[self._grip_eq_id] = 0

    # ------------------------------------------------------------------

    def _solve_waypoints_isolated(self) -> dict[str, np.ndarray]:
        """Call ``_ik_waypoints`` while restoring ``data.qpos`` afterwards so
        the IK iteration doesn't corrupt the simulation state that the rest
        of ``run`` depends on. The GPU copy is re-uploaded.
        """
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        try:
            wpts = _ik_waypoints(self.model, self.data, self.params)
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)
            self.backend.d_gpu = self.backend._mjwp.put_data(
                self.model, self.data)
        return wpts

    def run(self, frame_renderer=None) -> tuple[list[StepRecord], list[np.ndarray], dict]:
        """Execute the full scripted demo at ``pc.RENDER_FPS``.

        Records one StepRecord and (optionally) one camera frame per output
        timestep. Each output timestep contains ``physics_steps_per_frame``
        underlying GPU physics steps so the dynamics remain at 500 Hz.
        Returns ``(records, frames, meta)``.

        ``meta["phase_frame_boundaries"]`` -> dict[phase] = (first_frame,
        last_frame_exclusive); ``meta["phase_frame_count"]`` -> dict[phase]
        = n_frames so the EMG pairer can size each phase's EMG package.
        """
        wpts = self._solve_waypoints_isolated()
        segments: list[tuple[str, np.ndarray, np.ndarray, float]] = [
            ("REST_PRE",    np.array(pc.ARM_HOVER_QPOS), wpts["HOVER"],
             self.params.durations_s["REST_PRE"]),
            ("REACHING",    wpts["HOVER"], wpts["PREGRIP"],
             self.params.durations_s["REACHING"]),
            ("GRIPPING",    wpts["PREGRIP"], wpts["GRIP"],
             self.params.durations_s["GRIPPING"]),
            ("STABILIZING", wpts["GRIP"], wpts["STABILIZE"],
             self.params.durations_s["STABILIZING"]),
            ("RELEASING",   wpts["STABILIZE"], wpts["RELEASE"],
             self.params.durations_s["RELEASING"]),
            ("REST_POST",   wpts["RELEASE"], wpts["RELEASE"],
             self.params.durations_s["REST_POST"]),
        ]

        dt = pc.SIM_TIMESTEP_S
        physics_per_frame = max(1, int(round((1.0 / pc.RENDER_FPS) / dt)))
        frame_dt = physics_per_frame * dt

        jaw_l, jaw_r = 0.0, 0.0
        weld_active = False
        gripping_elapsed = 0.0
        records: list[StepRecord] = []
        frames: list[np.ndarray] = []
        phase_frame_boundaries: dict[str, tuple[int, int]] = {}
        phase_frame_count: dict[str, int] = {}

        frame_idx = 0
        t_total = 0.0
        for seg_idx, (phase, q_a, q_b, dur) in enumerate(segments):
            n_frames = max(1, int(round(dur / frame_dt)))
            start_idx = frame_idx
            for k in range(n_frames):
                s = (k + 1) / float(n_frames)
                q_target = _smoothstep_interp(q_a, q_b, s)

                public_phase = phase if phase not in (
                    "REST_PRE", "REST_POST") else "REST"
                if phase == "GRIPPING":
                    gripping_elapsed += frame_dt
                    # Hold jaws open while the gripper descends, then close
                    # them around the grape near the bottom of the stroke.
                    # Without this delay the jaws collide with the grape from
                    # the side mid-descent and bat it away.
                    if s < 0.80:
                        jc_target = 0.0
                        ja = 1.0 - 0.5 ** (frame_dt / 0.18)
                    else:
                        jc_target = jaw_close_ctrl_for_force(
                            self.params.grip_force_N)
                        ja = 1.0 - 0.5 ** (frame_dt / max(
                            self.params.close_speed_halflife, 1e-3))
                    jaw_l += ja * (jc_target - jaw_l)
                    jaw_r += ja * (jc_target - jaw_r)
                    # Only weld once the gripper has reached the grape AND
                    # the jaws have visibly closed around it.
                    if (not weld_active) and s >= 0.95:
                        self._set_grip_weld(True)
                        weld_active = True
                    applied_force = (self.params.grip_force_N
                                     if s >= 0.80 else 0.0)
                elif phase == "STABILIZING":
                    jc = jaw_close_ctrl_for_force(self.params.grip_force_N)
                    ja = 1.0 - 0.5 ** (frame_dt / 0.10)
                    jaw_l += ja * (jc - jaw_l)
                    jaw_r += ja * (jc - jaw_r)
                    applied_force = self.params.grip_force_N
                elif phase == "RELEASING":
                    arm_err = float(np.linalg.norm(q_target - q_b))
                    if weld_active and arm_err <= 0.02:
                        self._set_grip_weld(False)
                        weld_active = False
                    if weld_active:
                        jc = jaw_close_ctrl_for_force(self.params.grip_force_N)
                        ja = 1.0 - 0.5 ** (frame_dt / 0.10)
                        jaw_l += ja * (jc - jaw_l)
                        jaw_r += ja * (jc - jaw_r)
                        applied_force = self.params.grip_force_N
                    else:
                        ja = 1.0 - 0.5 ** (frame_dt / 0.18)
                        jaw_l += ja * (0.0 - jaw_l)
                        jaw_r += ja * (0.0 - jaw_r)
                        applied_force = 0.0
                else:
                    ja = 1.0 - 0.5 ** (frame_dt / 0.18)
                    jaw_l += ja * (0.0 - jaw_l)
                    jaw_r += ja * (0.0 - jaw_r)
                    applied_force = 0.0
                    if weld_active and phase == "REST_POST":
                        self._set_grip_weld(False)
                        weld_active = False

                ctrl = np.zeros(self.model.nu, dtype=np.float64)
                for q, name in zip(q_target, _ARM_NAMES):
                    ctrl[self._aid[name]] = float(q)
                ctrl[self._aid["left_jaw_act"]] = float(jaw_l)
                ctrl[self._aid["right_jaw_act"]] = float(jaw_r)

                self.backend.set_ctrl(ctrl)
                for _ in range(physics_per_frame):
                    self.backend.step()

                # record state at this step
                ee_pos = np.asarray(
                    self.data.site_xpos[self._tcp_sid]).copy().astype(
                    np.float32)
                # ee orientation from wrist_3
                xmat = np.asarray(self.data.xmat[self._wrist_bid]).reshape(3, 3)
                quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat, xmat.flatten())
                contact_N = self.backend.jaw_contact_force_N()
                arm_q_now = np.asarray(
                    [float(self.data.qpos[a]) for a in self._arm_qadr],
                    dtype=np.float32)
                # arm qvel
                arm_qvel = np.asarray(
                    [float(self.data.qvel[
                        int(self.model.jnt_dofadr[mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)])])
                     for jn in ("shoulder_pan_joint", "shoulder_lift_joint",
                                "elbow_joint", "wrist_1_joint",
                                "wrist_2_joint", "wrist_3_joint")],
                    dtype=np.float32)
                # gripper_state: 0=closed, 1=open. Use average of left+right
                # ctrls scaled to 0..1 (range 0..0.045).
                gripper_state = 1.0 - 0.5 * (jaw_l + jaw_r) / 0.045
                gripper_state = float(max(0.0, min(1.0, gripper_state)))

                records.append(StepRecord(
                    t_s=t_total,
                    phase=public_phase,
                    intent_id=INTENT_ID[public_phase],
                    arm_q=arm_q_now,
                    arm_qvel=arm_qvel,
                    ee_pos=ee_pos.astype(np.float32),
                    ee_quat=quat.astype(np.float32),
                    gripper_state=gripper_state,
                    applied_force_N=float(applied_force),
                    contact_force_N=float(contact_N),
                    weld_active=bool(weld_active),
                ))
                if frame_renderer is not None:
                    frames.append(frame_renderer.render(self.data))
                frame_idx += 1
                t_total += frame_dt
            end_idx = frame_idx
            phase_frame_boundaries[phase] = (start_idx, end_idx)
            phase_frame_count[phase] = end_idx - start_idx

        meta = {
            "phase_frame_boundaries": phase_frame_boundaries,
            "phase_frame_count": phase_frame_count,
            "total_frames": frame_idx,
            "duration_s": t_total,
            "render_fps": pc.RENDER_FPS,
            "physics_per_frame": physics_per_frame,
        }
        return records, frames, meta

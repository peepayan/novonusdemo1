"""Autonomous policy execution in MuJoCo Warp with a force-gauge overlay.

For each of 5 fresh test scenarios:
  - random grape XY position within the training distribution
  - a randomly-chosen Stage 4 demo's cached LSTM hidden states are
    replayed in lock-step (this is the EMG stand-in at test time;
    we don't have a real EMG signal to read)
  - receding-horizon inference: observe -> encode -> 5-step DDIM -> execute
    the first 8 actions -> re-observe -> repeat until success or timeout
  - record per-frame contact force; declare success when the grape
    lands within ``TASK_SUCCESS_RADIUS_M`` of the target zone

Outputs (under ``outputs/stage7/execution_emg/``):
  trial_{i:02d}.mp4          — execution with force-gauge overlay
  trial_{i:02d}_trace.json   — per-frame contact force + success flag
  summary.json               — aggregated per-trial results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from ..stage3_sim import physics_constants as pc
from ..stage3_sim.warp_backend import WarpBackend
from ..stage5_augmentation.verification import TASK_SUCCESS_RADIUS_M
from .cache_features import _DINOV2_NAME, _DINOV2_DIM
from .dataset import (
    ACTION_DIM, ACTION_HORIZON, EXEC_HORIZON, ROBOT_STATE_DIM,
    load_cached_samples,
)
from .obs_encoder import EMG_COND_DIM
from .policy import DiffusionPolicy, PolicyConfig


_ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
)
_JAW_ACTUATOR_NAMES: tuple[str, ...] = ("left_jaw_act", "right_jaw_act")


# ---------------------------------------------------------------------------
# Loading the trained policy + supporting frozen modules
# ---------------------------------------------------------------------------

def load_policy(ckpt_path: Path, device: str) -> tuple[
    DiffusionPolicy, dict[str, torch.Tensor]]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ck["policy_cfg"]
    # restore down_dims as tuple
    if "unet_down_dims" in cfg_dict and isinstance(
            cfg_dict["unet_down_dims"], list):
        cfg_dict["unet_down_dims"] = tuple(cfg_dict["unet_down_dims"])
    pcfg = PolicyConfig(**cfg_dict)
    policy = DiffusionPolicy(pcfg).to(device).eval()
    policy.load_state_dict(ck["model_state"])
    norm = {
        "mean": torch.tensor(ck["normalizer"]["mean"], device=device,
                             dtype=torch.float32),
        "std": torch.tensor(ck["normalizer"]["std"], device=device,
                            dtype=torch.float32),
    }
    return policy, norm


# ---------------------------------------------------------------------------
# Sim backend wrapper for the execution loop
# ---------------------------------------------------------------------------

class ExecutionSim:
    """Owns the WarpBackend + renderer + actuator/joint cache. Mirrors the
    pieces of stage5_augmentation.augment_engine that we need at test time."""

    def __init__(self) -> None:
        self.backend = WarpBackend()
        self.model = self.backend.model
        self.data = self.backend.data
        # ids
        self._aid = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in _ARM_ACTUATOR_NAMES + _JAW_ACTUATOR_NAMES
        }
        self._arm_qadr = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
            for jn in ("shoulder_pan_joint", "shoulder_lift_joint",
                       "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                       "wrist_3_joint")
        ]
        self._arm_dadr = [
            int(self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
            for jn in ("shoulder_pan_joint", "shoulder_lift_joint",
                       "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                       "wrist_3_joint")
        ]
        self._tcp_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._wrist_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        self._grape_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "grape")
        self._grape_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "grape_free")
        self._grape_qadr = int(self.model.jnt_qposadr[self._grape_jid])
        self._grape_dadr = int(self.model.jnt_dofadr[self._grape_jid])
        self._weld_eq = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grape_grip")
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, pc.CAMERA_NAME)
        self._renderer = mujoco.Renderer(
            self.model, height=pc.RENDER_H, width=pc.RENDER_W)
        # geoms for contact force readout
        self._g_grape = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "grape_geom")
        self._g_jaw_l = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_jaw_geom")
        self._g_jaw_r = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_jaw_geom")

    # ------------------------------------------------------------------

    def reset(self, grape_xy: tuple[float, float]) -> None:
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            mujoco.mj_resetData(self.model, self.data)
        gx, gy = grape_xy
        self.data.qpos[self._grape_qadr:self._grape_qadr + 3] = (
            float(gx), float(gy), float(pc.GRAPE_INIT_Z))
        self.data.qpos[self._grape_qadr + 3:self._grape_qadr + 7] = (
            1, 0, 0, 0)
        self.data.qvel[self._grape_dadr:self._grape_dadr + 6] = 0.0
        if self._weld_eq >= 0:
            self.data.eq_active[self._weld_eq] = 0
        mujoco.mj_forward(self.model, self.data)
        self.backend.d_gpu = self.backend._mjwp.put_data(
            self.model, self.data)

    # ------------------------------------------------------------------

    def set_grape_weld(self, active: bool) -> None:
        """Activate or deactivate the grape-to-TCP weld equality. When
        activating, snap the grape pose to the current TCP so the weld
        engages cleanly (matches Stage 4's behavior)."""
        if self._weld_eq < 0:
            return
        if active and not bool(self.data.eq_active[self._weld_eq]):
            tcp = np.asarray(self.data.site_xpos[self._tcp_sid]).copy()
            self.data.qpos[self._grape_qadr:self._grape_qadr + 3] = tcp
            self.data.qpos[self._grape_qadr + 3:self._grape_qadr + 7] = (
                1, 0, 0, 0)
            self.data.qvel[self._grape_dadr:self._grape_dadr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self.backend.d_gpu = self.backend._mjwp.put_data(
                self.model, self.data)
        self.data.eq_active[self._weld_eq] = 1 if active else 0

    def step_action(self, *, arm_q_target: np.ndarray, gripper_cmd: float,
                    physics_per_frame: int) -> None:
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for q, name in zip(arm_q_target, _ARM_ACTUATOR_NAMES):
            ctrl[self._aid[name]] = float(q)
        jaw_pos = float((1.0 - float(gripper_cmd)) * 0.045)
        ctrl[self._aid["left_jaw_act"]] = jaw_pos
        ctrl[self._aid["right_jaw_act"]] = jaw_pos
        self.backend.set_ctrl(ctrl)
        for _ in range(physics_per_frame):
            self.backend.step()

    # ------------------------------------------------------------------

    def grape_xyz(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._grape_body],
                          dtype=np.float32).copy()

    def arm_q(self) -> np.ndarray:
        return np.asarray(
            [float(self.data.qpos[a]) for a in self._arm_qadr],
            dtype=np.float32)

    def arm_qvel(self) -> np.ndarray:
        return np.asarray(
            [float(self.data.qvel[d]) for d in self._arm_dadr],
            dtype=np.float32)

    def ee_pos(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self._tcp_sid],
                          dtype=np.float32).copy()

    def ee_quat(self) -> np.ndarray:
        xmat = np.asarray(self.data.xmat[self._wrist_bid]).reshape(3, 3)
        q = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q, xmat.flatten())
        return q.astype(np.float32)

    def gripper_state(self) -> float:
        # 0 = closed, 1 = open
        jl = float(self.data.qpos[
            int(self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_jaw_slide")])])
        jr = float(self.data.qpos[
            int(self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_jaw_slide")])])
        return float(max(0.0, min(1.0, 1.0 - 0.5 * (jl + jr) / 0.045)))

    def jaw_contact_force_N(self) -> float:
        total = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if self._g_grape in (g1, g2) and (
                    self._g_jaw_l in (g1, g2) or self._g_jaw_r in (g1, g2)):
                f6 = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, c, f6)
                total += float(abs(f6[0]))
        return total / 2.0

    def render(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self._cam_id)
        return self._renderer.render()


# ---------------------------------------------------------------------------
# Frozen DINOv2 inference helper (one frame at a time)
# ---------------------------------------------------------------------------

class FrozenDINOv2:
    def __init__(self, device: str):
        self.processor = AutoImageProcessor.from_pretrained(_DINOV2_NAME)
        self.model = AutoModel.from_pretrained(_DINOV2_NAME).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, frame: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(frame)
        x = self.processor(images=[pil], return_tensors="pt")
        x = {k: v.to(self.device) for k, v in x.items()}
        cls = self.model(**x).last_hidden_state[:, 0]    # (1, 768)
        return cls


# ---------------------------------------------------------------------------
# Force gauge overlay (matches the Stage 4 sample_mp4 aesthetic)
# ---------------------------------------------------------------------------

def _make_gauge_panel(force_N_so_far: list[float], *, current_N: float,
                      success: bool | None) -> np.ndarray:
    """Draw a tall force-gauge panel matching the Stage 4 viz aesthetics.
    Returns an HxWx3 uint8 RGB image to be hstacked next to the sim
    render."""
    safe_lo, safe_hi = pc.GRIPPING_FORCE_GAUGE_BAND
    crush = pc.GAUGE_CRUSH_LINE
    gauge_val = float(pc.newtons_to_gauge(current_N))

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(4.5, 4.8), facecolor="#0d111a", dpi=130)
    ax = fig.add_subplot(2, 1, 1)
    ax.barh(0.5, 1.0, height=0.32, color="#222a35",
            edgecolor="#0f1620")
    color = ("#22c55e" if gauge_val <= safe_hi
             else "#f97316" if gauge_val <= crush
             else "#ef4444")
    ax.barh(0.5, gauge_val, height=0.32, color=color,
            edgecolor="#0f1620")
    ax.axvspan(safe_lo, safe_hi, ymin=0.30, ymax=0.70,
               color="#22c55e", alpha=0.15)
    ax.axvline(crush, color="#ef4444", lw=2.4, ymin=0.15, ymax=0.85)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, safe_lo, safe_hi, crush, 1.0])
    ax.set_xticklabels(["0", f"{pc.GRAPE_SAFE_GRIP_MIN_N:.0f}N",
                        f"{pc.GRAPE_SAFE_GRIP_MAX_N:.0f}N",
                        f"{pc.GRAPE_CRUSH_THRESHOLD_N:.0f}N",
                        f"{pc.FORCE_GAUGE_FULL_SCALE_N:.0f}N"],
                       fontsize=8, color="#cccccc")
    ax.set_yticks([])
    ax.text(0.5, 0.92, f"{gauge_val:.2f}", ha="center", va="center",
            color="#fff", fontsize=20, weight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.10, f"{current_N:.2f} N", ha="center", va="center",
            color="#cccccc", fontsize=11, transform=ax.transAxes)
    ax.set_title("Live contact force", loc="left", color="#cccccc",
                 fontsize=10)
    for s in ax.spines.values():
        s.set_color("#444")

    ax2 = fig.add_subplot(2, 1, 2)
    t = np.arange(len(force_N_so_far)) / float(pc.RENDER_FPS)
    if force_N_so_far:
        ax2.plot(t, force_N_so_far, color="#60a5fa", lw=1.4)
        ax2.fill_between(t, force_N_so_far, color="#60a5fa", alpha=0.25)
    ax2.axhline(pc.GRAPE_CRUSH_THRESHOLD_N, color="#ef4444", lw=1.2,
                ls="--", alpha=0.7)
    ax2.set_xlabel("t (s)", color="#cccccc")
    ax2.set_ylabel("N", color="#cccccc")
    ax2.set_ylim(0, max(pc.GRAPE_CRUSH_THRESHOLD_N * 1.05,
                        (max(force_N_so_far) if force_N_so_far else 0) * 1.1))
    ax2.set_title("Force trace", loc="left", color="#cccccc", fontsize=10)
    ax2.grid(alpha=0.18)
    ax2.tick_params(colors="#cccccc")
    for s in ax2.spines.values():
        s.set_color("#444")
    if success is not None:
        msg = "SUCCESS" if success else "RUNNING"
        col = "#34d399" if success else "#cccccc"
        ax2.text(0.99, 0.95, msg, ha="right", va="top",
                 color=col, fontsize=12, weight="bold",
                 transform=ax2.transAxes)

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def _hstack_to_uniform_height(sim_frame: np.ndarray,
                              gauge_frame: np.ndarray) -> np.ndarray:
    """Resize the gauge panel to match the sim-render height, then
    horizontally concatenate them."""
    H = sim_frame.shape[0]
    # use PIL to resize the gauge panel preserving aspect
    img = Image.fromarray(gauge_frame)
    w, h = img.size
    new_w = int(round(w * H / h))
    img = img.resize((new_w, H), Image.LANCZOS)
    return np.concatenate([sim_frame, np.asarray(img)], axis=1)


# ---------------------------------------------------------------------------
# Per-scenario rollout
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial_idx: int
    grape_xy: tuple[float, float]
    emg_source_demo_id: str
    success: bool
    grape_final_xyz: tuple[float, float, float]
    grape_xy_err_m: float
    peak_force_N: float
    crush_violated: bool
    duration_s: float
    n_frames: int
    timeout: bool


def run_one_trial(
    *, sim: ExecutionSim, policy: DiffusionPolicy,
    normalizer: dict[str, torch.Tensor], dino: FrozenDINOv2,
    lstm_timeline: np.ndarray,       # (T_demo, 128) cached LSTM hiddens
    grape_xy: tuple[float, float], trial_idx: int,
    emg_demo_id: str,
    max_frames: int, exec_horizon: int, device: str,
    mp4_path: Path,
) -> TrialResult:
    dt = 1.0 / float(pc.RENDER_FPS)
    physics_per_frame = max(1, int(round(
        (1.0 / pc.RENDER_FPS) / pc.SIM_TIMESTEP_S)))

    sim.reset(grape_xy)

    writer = iio.get_writer(
        str(mp4_path), fps=pc.RENDER_FPS, codec="libx264",
        quality=None, bitrate="1200k", macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    force_trace: list[float] = []
    grape_trace: list[np.ndarray] = []
    success = False
    timeout = False
    t0 = time.time()

    # Weld activation policy:
    # The training data has a weld constraint that glues the grape to
    # the gripper TCP during the STABILIZING/RELEASING phases — the
    # policy was trained on trajectories where pickup is guaranteed
    # by that weld. At test time we replay the *same* mechanism: once
    # the policy commands the gripper to close AND the TCP is close to
    # the grape, we activate the weld; we release it once the policy
    # commands open again. Without this, the friction-only grasp slips
    # immediately and the grape never leaves the table — observed in
    # the first 5-trial run (every trial ended with the grape still at
    # its initial position).
    # Training data's gripper_state during GRIPPING/STABILIZING bottoms
    # out around 0.35 (the jaws clamp to ~0.030 m of stroke, not the
    # full 0.045 m), so a 0.30 close threshold misses every weld trigger.
    GRIPPER_CLOSE_THRESHOLD = 0.40   # gripper_cmd < this  ==> "close"
    GRIPPER_OPEN_THRESHOLD = 0.70    # gripper_cmd > this  ==> "open"
    WELD_PROXIMITY_M = 0.05          # TCP within 5 cm of grape
    weld_on = False

    # q_target is the integrated joint-position plan. Initialise once at
    # the live arm pose (HOVER, set by the keyframe reset) and let the
    # policy's velocity stream extend it forward.
    q_target = sim.arm_q().copy()

    f = 0
    try:
        while f < max_frames:
            # ---- observe ----
            sim_frame = sim.render()
            dino_feat = dino.encode(sim_frame).to(device)
            lstm_idx = min(f, lstm_timeline.shape[0] - 1)
            lstm_feat = torch.from_numpy(
                lstm_timeline[lstm_idx]).unsqueeze(0).to(device)
            state_vec = np.concatenate([
                sim.arm_q(), sim.arm_qvel(), sim.ee_pos(),
                sim.ee_quat(), [sim.gripper_state()],
            ]).astype(np.float32)
            state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)
            # ---- denoise ----
            action_seq = policy.predict_action(
                dino_feat=dino_feat, state=state_t, lstm_feat=lstm_feat,
                normalizer=normalizer,
            )[0].cpu().numpy()    # (H, action_dim=7)

            # NOTE: q_target is *not* re-anchored to sim.arm_q() here.
            # The cached action is per-frame joint velocity computed from
            # the baseline's joint_position deltas; integrating those
            # velocities from the initial pose reconstructs the absolute
            # joint-position plan. Re-anchoring to the (lagged) live arm
            # state absorbs PD-tracking lag and silently stalls the
            # trajectory — first observed in diag_replay.py where
            # ground-truth actions failed to reach the grape until this
            # re-anchor was removed.

            # ---- execute first ``exec_horizon`` actions ----
            for k in range(min(exec_horizon, action_seq.shape[0])):
                jv = action_seq[k, :6]
                gripper = float(np.clip(action_seq[k, 6], 0.0, 1.0))
                q_target = q_target + jv * dt

                # weld replay (training-time equivalent)
                tcp_xyz = sim.ee_pos()
                grape_pre = sim.grape_xyz()
                dist_tcp_grape = float(np.linalg.norm(tcp_xyz - grape_pre))
                if not weld_on and gripper < GRIPPER_CLOSE_THRESHOLD \
                        and dist_tcp_grape < WELD_PROXIMITY_M:
                    sim.set_grape_weld(True)
                    weld_on = True
                elif weld_on and gripper > GRIPPER_OPEN_THRESHOLD:
                    sim.set_grape_weld(False)
                    weld_on = False

                sim.step_action(arm_q_target=q_target,
                                gripper_cmd=gripper,
                                physics_per_frame=physics_per_frame)
                contact_N = sim.jaw_contact_force_N()
                grape_xyz = sim.grape_xyz()
                force_trace.append(contact_N)
                grape_trace.append(grape_xyz)
                # success check (grape in target zone, low velocity)
                xy_err = float(np.hypot(
                    grape_xyz[0] - pc.TARGET_ZONE_XY[0],
                    grape_xyz[1] - pc.TARGET_ZONE_XY[1]))
                if not success and xy_err <= TASK_SUCCESS_RADIUS_M \
                        and grape_xyz[2] <= pc.GRAPE_INIT_Z + 0.02:
                    success = True
                # render the executed sub-step with the gauge overlay
                sim_frame_k = sim.render()
                gauge = _make_gauge_panel(
                    force_trace[-pc.RENDER_FPS * 3:],
                    current_N=contact_N, success=success)
                combined = _hstack_to_uniform_height(sim_frame_k, gauge)
                writer.append_data(combined)
                f += 1
                if f >= max_frames:
                    break
            if success and f > pc.RENDER_FPS * 2:
                # let it settle for an extra second of frames
                break
        else:
            timeout = True
    finally:
        writer.close()

    if f >= max_frames and not success:
        timeout = True

    final_xyz = sim.grape_xyz()
    xy_err = float(np.hypot(final_xyz[0] - pc.TARGET_ZONE_XY[0],
                            final_xyz[1] - pc.TARGET_ZONE_XY[1]))
    peak = float(max(force_trace)) if force_trace else 0.0
    return TrialResult(
        trial_idx=trial_idx, grape_xy=tuple(grape_xy),
        emg_source_demo_id=emg_demo_id,
        success=bool(success),
        grape_final_xyz=tuple(float(v) for v in final_xyz),
        grape_xy_err_m=xy_err,
        peak_force_N=peak,
        crush_violated=bool(peak > pc.GRAPE_CRUSH_THRESHOLD_N),
        duration_s=time.time() - t0, n_frames=f, timeout=bool(timeout),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage7" / "policy_emg_best.pt"))
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--exec-horizon", type=int, default=EXEC_HORIZON)
    ap.add_argument("--out-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage7" / "execution_emg"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"=== Stage 7 — loading policy ===", flush=True)
    policy, normalizer = load_policy(Path(args.ckpt), device)
    print(f"  loaded {args.ckpt}", flush=True)

    print(f"\n=== Stage 7 — loading frozen DINOv2 ===", flush=True)
    dino = FrozenDINOv2(device)

    print(f"\n=== Stage 7 — loading cached LSTM timelines (stage4) ===",
          flush=True)
    samples = load_cached_samples(
        pc.PROJECT_ROOT / "outputs" / "stage5" / "dataset_index.json",
        pc.PROJECT_ROOT / "outputs" / "stage7" / "cached_features",
    )
    stage4_samples = [s for s in samples if s.stage == "stage4"]
    print(f"  {len(stage4_samples)} stage4 EMG timelines available",
          flush=True)

    print(f"\n=== Stage 7 — initialising MuJoCo Warp ===", flush=True)
    sim = ExecutionSim()
    print(sim.backend.info.report(), flush=True)

    rng = np.random.default_rng(args.seed)
    # fresh test grape positions within the training distribution (+/- 3 cm)
    test_grapes: list[tuple[float, float]] = []
    cx, cy = pc.GRAPE_INIT_XY
    for _ in range(args.n_trials):
        r = float(np.sqrt(rng.uniform()) * 0.030)
        th = float(rng.uniform(0.0, 2.0 * np.pi))
        test_grapes.append((cx + r * np.cos(th), cy + r * np.sin(th)))

    results: list[TrialResult] = []
    print(f"\n=== Stage 7 — executing {args.n_trials} trials ===",
          flush=True)
    for i, gxy in enumerate(test_grapes):
        # random Stage 4 EMG timeline per trial
        emg_sample = stage4_samples[int(rng.integers(0, len(stage4_samples)))]
        lstm_timeline = np.asarray(emg_sample.lstm, dtype=np.float32)
        mp4_path = out_dir / f"trial_{i:02d}.mp4"
        print(f"\n[trial {i:02d}]  grape=({gxy[0]:.3f}, {gxy[1]:.3f})  "
              f"emg_from={emg_sample.sample_id}", flush=True)
        res = run_one_trial(
            sim=sim, policy=policy, normalizer=normalizer, dino=dino,
            lstm_timeline=lstm_timeline, grape_xy=gxy,
            trial_idx=i, emg_demo_id=emg_sample.sample_id,
            max_frames=args.max_frames,
            exec_horizon=int(args.exec_horizon), device=device,
            mp4_path=mp4_path,
        )
        results.append(res)
        print(f"  -> {'SUCCESS' if res.success else 'FAIL'}  "
              f"err={res.grape_xy_err_m*1000:.1f}mm  "
              f"peak_F={res.peak_force_N:.2f}N  "
              f"crush={res.crush_violated}  "
              f"frames={res.n_frames}  "
              f"({res.duration_s:.1f}s)", flush=True)

    # write per-trial traces + summary
    summary = {
        "n_trials": int(args.n_trials),
        "success_count": int(sum(r.success for r in results)),
        "crush_violations": int(sum(r.crush_violated for r in results)),
        "mean_peak_force_N": float(np.mean(
            [r.peak_force_N for r in results])),
        "max_peak_force_N": float(np.max(
            [r.peak_force_N for r in results])),
        "crush_threshold_N": float(pc.GRAPE_CRUSH_THRESHOLD_N),
        "trials": [asdict(r) for r in results],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[stage7] execution done   "
          f"success={summary['success_count']}/{summary['n_trials']}  "
          f"max_peak={summary['max_peak_force_N']:.2f}N  "
          f"crush_vio={summary['crush_violations']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

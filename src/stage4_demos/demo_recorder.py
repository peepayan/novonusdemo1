"""One-shot demo recorder: scripted run + EMG pairing + success check + save.

Failed attempts are discarded; only successful demos are written to disk.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import mujoco
import numpy as np

from ..stage2_lstm import dataset as ds
from ..stage2_lstm.model import IntentForceLSTM
from ..stage3_sim import physics_constants as pc
from ..stage3_sim.warp_backend import WarpBackend
from .controller import (
    DemoParams, ScriptedController, INTENT_ID,
)
from .emg_pairing import PairedEMG, ValEMGProbe, build_paired_emg_for_demo
from .renderer import FrameRenderer


SUCCESS_DIST_M: float = 0.030     # grape must end <= 30 mm from zone centre


@dataclass
class DemoSuccessCheck:
    grape_in_zone: bool
    grape_xy_err_m: float
    grape_final_xyz: tuple[float, float, float]
    max_contact_force_N: float
    force_below_crush: bool
    phases_seen: dict[str, int]
    all_phases_executed: bool

    def passed(self) -> bool:
        return (self.grape_in_zone and self.force_below_crush and
                self.all_phases_executed)


def evaluate_demo(records, backend) -> DemoSuccessCheck:
    grape_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY,
                                 "grape")
    final = backend.data.xpos[grape_id].copy()
    xy_err = float(np.hypot(final[0] - pc.TARGET_ZONE_XY[0],
                            final[1] - pc.TARGET_ZONE_XY[1]))
    contact_arr = np.asarray([r.contact_force_N for r in records],
                             dtype=np.float32)
    max_contact = float(contact_arr.max()) if contact_arr.size else 0.0
    phases_seen: dict[str, int] = {}
    for r in records:
        phases_seen[r.phase] = phases_seen.get(r.phase, 0) + 1
    required = ("REACHING", "GRIPPING", "STABILIZING", "RELEASING")
    all_ok = all(phases_seen.get(p, 0) > 0 for p in required)
    return DemoSuccessCheck(
        grape_in_zone=(xy_err <= SUCCESS_DIST_M),
        grape_xy_err_m=xy_err,
        grape_final_xyz=tuple(float(v) for v in final),
        max_contact_force_N=max_contact,
        force_below_crush=(max_contact <
                           pc.GRAPE_CRUSH_THRESHOLD_N),
        phases_seen=phases_seen,
        all_phases_executed=all_ok,
    )


# ---------------------------------------------------------------------------

def record_one_demo(backend: WarpBackend,
                    params: DemoParams,
                    arrays: ds.Stage1Arrays,
                    probe: ValEMGProbe,
                    model: IntentForceLSTM,
                    rng: np.random.Generator,
                    device: str = "cuda:0",
                    ) -> tuple[bool, dict | None, DemoSuccessCheck]:
    """Run one scripted demo and, if it passes the success check, return a
    fully-built data dict ready for ``np.savez_compressed``.
    """
    controller = ScriptedController(backend, params)
    controller.reset()
    renderer = FrameRenderer(backend.model)
    records, frames, meta = controller.run(frame_renderer=renderer)
    check = evaluate_demo(records, backend)
    if not check.passed():
        return False, None, check

    # build EMG pairing sized to per-phase frame counts
    paired = build_paired_emg_for_demo(
        arrays, probe, model,
        phase_step_boundaries={k: v for k, v in
                               meta["phase_frame_boundaries"].items()},
        total_frames_per_phase=meta["phase_frame_count"],
        rng=rng, device=device,
    )

    T = len(records)
    if paired.emg_envelope_T.shape[0] != T:
        # safety: pad/truncate
        n = paired.emg_envelope_T.shape[0]
        if n < T:
            pad = T - n
            paired.emg_envelope_T = np.concatenate(
                (paired.emg_envelope_T,
                 np.repeat(paired.emg_envelope_T[-1:], pad, axis=0)), axis=0)
            paired.emg_force_intensity_T = np.concatenate(
                (paired.emg_force_intensity_T,
                 np.repeat(paired.emg_force_intensity_T[-1:], pad, axis=0)),
                axis=0)
            paired.intent_labels_T = np.concatenate(
                (paired.intent_labels_T,
                 np.repeat(paired.intent_labels_T[-1:], pad, axis=0)),
                axis=0)
        else:
            paired.emg_envelope_T = paired.emg_envelope_T[:T]
            paired.emg_force_intensity_T = paired.emg_force_intensity_T[:T]
            paired.intent_labels_T = paired.intent_labels_T[:T]

    # convert records to arrays
    joint_positions = np.stack([r.arm_q for r in records],
                               axis=0).astype(np.float32)
    joint_velocities = np.stack([r.arm_qvel for r in records],
                                axis=0).astype(np.float32)
    end_effector_pos = np.stack([r.ee_pos for r in records],
                                axis=0).astype(np.float32)
    end_effector_quat = np.stack([r.ee_quat for r in records],
                                 axis=0).astype(np.float32)
    gripper_state = np.array([r.gripper_state for r in records],
                             dtype=np.float32)
    applied_force = np.array([r.applied_force_N for r in records],
                             dtype=np.float32)

    intent_labels = paired.intent_labels_T.astype(np.int8)

    data_dict = {
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "end_effector_pos": end_effector_pos,
        "end_effector_quat": end_effector_quat,
        "gripper_state": gripper_state,
        "applied_force": applied_force,
        "emg_envelope": paired.emg_envelope_T,
        "emg_force_intensity": paired.emg_force_intensity_T,
        "intent_labels": intent_labels,
        "grape_initial_pos": np.array(
            (pc.GRAPE_INIT_XY[0] + params.grape_xy_offset[0],
             pc.GRAPE_INIT_XY[1] + params.grape_xy_offset[1],
             pc.GRAPE_INIT_Z), dtype=np.float32),
        "success": True,
        "_frames": frames,                       # popped before saving
        "_check": check,
    }
    metadata = {
        "demo_id": params.demo_id,
        "timestamp_unix": time.time(),
        "num_timesteps": int(T),
        "render_fps": int(pc.RENDER_FPS),
        "duration_s": float(meta["duration_s"]),
        "phase_frame_boundaries": {
            k: list(v) for k, v in meta["phase_frame_boundaries"].items()},
        "phase_frame_count": {k: int(v) for k, v in
                              meta["phase_frame_count"].items()},
        "grape_xy_offset_m": list(params.grape_xy_offset),
        "approach_angle_rad": float(params.approach_angle_rad),
        "close_speed_halflife_s": float(params.close_speed_halflife),
        "grip_force_N": float(params.grip_force_N),
        "grape_final_xyz": list(check.grape_final_xyz),
        "grape_xy_err_m": float(check.grape_xy_err_m),
        "max_contact_force_N": float(check.max_contact_force_N),
        "crush_threshold_N": float(pc.GRAPE_CRUSH_THRESHOLD_N),
        "emg_pairing_note": paired.note,
        "emg_phase_source_ranges": {
            k: list(v) for k, v in paired.phase_source_ranges.items()},
    }
    data_dict["metadata"] = json.dumps(metadata)
    return True, data_dict, check


def save_demo_npz(data_dict: dict, frames: list,
                  npz_path: Path, frames_dir: Path) -> Path:
    npz_path = Path(npz_path); npz_path.parent.mkdir(parents=True,
                                                     exist_ok=True)
    # save frames as PNG sequence
    import imageio.v2 as iio
    frames_dir = Path(frames_dir); frames_dir.mkdir(parents=True,
                                                    exist_ok=True)
    for i, f in enumerate(frames):
        iio.imwrite(str(frames_dir / f"{i:04d}.png"), f)
    save_dict = {k: v for k, v in data_dict.items() if not k.startswith("_")}
    save_dict["frames"] = str(frames_dir.relative_to(npz_path.parent))
    np.savez_compressed(npz_path, **save_dict)
    return npz_path

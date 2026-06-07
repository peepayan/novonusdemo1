"""Diagnostic: replay a *ground-truth* action sequence (the cached
action.npy for one training demo) through the execute.py loop. If this
produces a successful pickup, the closed-loop velocity-integration is
fine and the failure is the policy itself. If it also fails, my
execution interpretation of the action is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np

from ..stage3_sim import physics_constants as pc
from .dataset import EXEC_HORIZON, load_cached_samples
from .execute import ExecutionSim


def main() -> int:
    cache_dir = pc.PROJECT_ROOT / "outputs" / "stage7" / "cached_features"
    samples = load_cached_samples(
        pc.PROJECT_ROOT / "outputs" / "stage5" / "dataset_index.json",
        cache_dir,
    )
    s = samples[0]   # stage4_demo_000
    print(f"replay {s.sample_id}", flush=True)
    action = np.asarray(s.action, dtype=np.float32)   # (T, 7)
    state0 = np.asarray(s.state[0], dtype=np.float32)
    arm_q_init = state0[:6]
    print(f"  T={action.shape[0]}  arm_q_init={arm_q_init}", flush=True)

    sim = ExecutionSim()
    # initial grape position from the cached state's training time —
    # the saved state's ee_pos is not the grape, so use the npz's
    # grape_initial_pos for an apples-to-apples replay.
    z = np.load(pc.PROJECT_ROOT / "outputs/stage4/demos/demo_000.npz",
                allow_pickle=False)
    gx, gy = float(z['grape_initial_pos'][0]), float(z['grape_initial_pos'][1])
    print(f"  grape init=({gx:.3f}, {gy:.3f})", flush=True)
    sim.reset((gx, gy))

    dt = 1.0 / float(pc.RENDER_FPS)
    physics_per_frame = max(1, int(round(
        (1.0 / pc.RENDER_FPS) / pc.SIM_TIMESTEP_S)))

    out_mp4 = pc.PROJECT_ROOT / "outputs" / "stage7" / "diag_replay.mp4"
    writer = iio.get_writer(
        str(out_mp4), fps=pc.RENDER_FPS, codec="libx264",
        quality=None, bitrate="900k", macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    # Weld activation: replay using saved phase_source_ranges via baseline
    # intent_labels. Simpler: mirror stage4 controller — weld on when
    # gripper command < 0.3 AND TCP near grape.
    weld_on = False
    q_target = sim.arm_q().copy()

    print(f"  initial q_target = {q_target}", flush=True)
    print(f"  arm_q_init from state = {arm_q_init}", flush=True)
    # The cached state's first frame was taken AFTER the keyframe reset
    # snapped the arm to HOVER. Our reset does the same, so q_target
    # should equal arm_q_init within FP tolerance.

    peak_F = 0.0
    success = False
    try:
        # Anchor q_target ONCE at f=0 to the initial arm position, then
        # integrate the velocity stream WITHOUT re-anchoring. Each
        # commanded target = initial_pose + cumsum(vel * dt), which is
        # the policy's planned absolute joint position trajectory.
        for f in range(action.shape[0]):
            jv = action[f, :6]
            gripper = float(np.clip(action[f, 6], 0.0, 1.0))
            q_target = q_target + jv * dt

            tcp = sim.ee_pos()
            grape = sim.grape_xyz()
            dist = float(np.linalg.norm(tcp - grape))
            if not weld_on and gripper < 0.30 and dist < 0.05:
                sim.set_grape_weld(True)
                weld_on = True
                print(f"  [f={f}] WELD ON  gripper={gripper:.2f}  "
                      f"dist={dist*1000:.1f}mm", flush=True)
            elif weld_on and gripper > 0.70:
                sim.set_grape_weld(False)
                weld_on = False
                print(f"  [f={f}] WELD OFF  gripper={gripper:.2f}", flush=True)

            sim.step_action(arm_q_target=q_target,
                            gripper_cmd=gripper,
                            physics_per_frame=physics_per_frame)
            cN = sim.jaw_contact_force_N()
            peak_F = max(peak_F, cN)
            grape_xyz = sim.grape_xyz()
            xy_err = float(np.hypot(
                grape_xyz[0] - pc.TARGET_ZONE_XY[0],
                grape_xyz[1] - pc.TARGET_ZONE_XY[1]))
            if not success and xy_err <= 0.03 and grape_xyz[2] <= pc.GRAPE_INIT_Z + 0.02:
                success = True
                print(f"  [f={f}] SUCCESS  xy_err={xy_err*1000:.1f}mm", flush=True)
            if f % 50 == 0:
                print(f"  [f={f:3d}]  gripper={gripper:.2f}  "
                      f"dist={dist*1000:.1f}mm  contact={cN:.2f}N  "
                      f"grape_z={grape_xyz[2]:.3f}", flush=True)
            writer.append_data(sim.render())
    finally:
        writer.close()

    final = sim.grape_xyz()
    xy_err = float(np.hypot(final[0] - pc.TARGET_ZONE_XY[0],
                            final[1] - pc.TARGET_ZONE_XY[1]))
    print(f"\nfinal grape = {final}  xy_err = {xy_err*1000:.1f}mm  "
          f"peak={peak_F:.2f}N  success={success}", flush=True)
    print(f"wrote {out_mp4}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

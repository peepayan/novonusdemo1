"""Rebuild MP4s for already-saved scenarios via kinematic replay.

Used when the physics data in ``scenarios/scenario_*.npz`` is correct but
the matching MP4 was rendered with a broken Renderer state (the early
Stage 5 batch produced all-black frames because the engine recreated the
``mujoco.Renderer`` mid-run, which corrupts the GL context after
``warp.init``).

This script does NOT re-run physics — it loads each scenario's saved
``joint_positions`` and ``grape_positions`` arrays and replays them
kinematically through ``mj_forward``, rendering each pose to a fresh MP4.
Grape orientation per timestep is not saved by the engine; we use the
identity quaternion (a sphere is rotationally symmetric so the visual
difference is negligible).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np

from ..stage3_sim import physics_constants as pc
from ..stage3_sim.warp_backend import WarpBackend
from .params import AugmentationParams
from .augment_engine import _ARM_ACTUATOR_NAMES


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage5" / "scenarios"))
    ap.add_argument("--frames-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage5" / "scenario_frames"))
    args = ap.parse_args(argv)

    scen_dir = Path(args.scen_dir)
    frames_dir = Path(args.frames_dir); frames_dir.mkdir(
        parents=True, exist_ok=True)
    paths = sorted(scen_dir.glob("scenario_*.npz"))
    if not paths:
        print(f"[error] no scenarios in {scen_dir}", file=sys.stderr)
        return 1

    # Spin up WarpBackend just to keep init ordering identical to the
    # engine (warp.init has to happen before the Renderer that we then
    # use).
    print("[rerender] initialising backend...", flush=True)
    backend = WarpBackend()
    model = backend.model
    data = backend.data

    # cache ids
    arm_qadr = []
    for jn in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        arm_qadr.append(int(model.jnt_qposadr[jid]))
    jaw_l_qadr = int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "left_jaw_slide")])
    jaw_r_qadr = int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_jaw_slide")])
    grape_jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "grape_free")
    grape_qadr = int(model.jnt_qposadr[grape_jid])
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                               pc.CAMERA_NAME)

    renderer = mujoco.Renderer(model, height=pc.RENDER_H, width=pc.RENDER_W)

    # nominal headlight (for visual lighting jitter replay)
    nom_amb = np.array(model.vis.headlight.ambient, dtype=np.float32).copy()
    nom_dif = np.array(model.vis.headlight.diffuse, dtype=np.float32).copy()

    t0 = time.time()
    for i, npz_path in enumerate(paths):
        z = np.load(npz_path, allow_pickle=False)
        jp = np.asarray(z["joint_positions"])
        gp = np.asarray(z["grape_positions"])
        gripper = np.asarray(z["gripper_state"])
        T = int(jp.shape[0])
        mp4_name = str(z["frames_mp4"])
        ap_params = json.loads(str(z["augmentation_params"]))
        light_scale = float(ap_params["light_ambient_scale"])

        out_mp4 = frames_dir / mp4_name
        writer = iio.get_writer(
            str(out_mp4), fps=pc.RENDER_FPS, codec="libx264",
            quality=None, bitrate="900k", macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        try:
            for t in range(T):
                # arm joints
                for k, qa in enumerate(arm_qadr):
                    data.qpos[qa] = float(jp[t, k])
                # jaws (gripper_state: 1=open -> 0 ctrl, 0=closed -> 0.045 ctrl)
                jaw_q = float((1.0 - float(gripper[t])) * 0.045)
                data.qpos[jaw_l_qadr] = jaw_q
                data.qpos[jaw_r_qadr] = jaw_q
                # grape position (identity orientation — orientation is
                # not saved per timestep; sphere is rotationally
                # symmetric so this is visually fine)
                data.qpos[grape_qadr:grape_qadr + 3] = (
                    float(gp[t, 0]), float(gp[t, 1]), float(gp[t, 2]))
                data.qpos[grape_qadr + 3:grape_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

                renderer.update_scene(data, camera=cam_id)
                # lighting jitter (per-scenario) applied at scene level
                for li in range(int(renderer.scene.nlight)):
                    light = renderer.scene.lights[li]
                    for c in range(3):
                        light.ambient[c] = float(np.clip(
                            nom_amb[c] * light_scale, 0.0, 1.0))
                        light.diffuse[c] = float(np.clip(
                            nom_dif[c] * light_scale, 0.0, 1.0))
                frame = renderer.render()
                writer.append_data(frame)
        finally:
            writer.close()
        print(f"  [{i+1}/{len(paths)}] {npz_path.name} -> {out_mp4.name}  "
              f"({T} frames)", flush=True)

    print(f"[rerender] done in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

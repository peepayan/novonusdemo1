"""Small rendering helpers: static scene PNG, idle motion MP4."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np

from . import physics_constants as pc
from .warp_backend import WarpBackend


def _renderer(model):
    return mujoco.Renderer(model, height=pc.RENDER_H, width=pc.RENDER_W)


def render_scene_png(backend: WarpBackend, out_path: Path) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    r = _renderer(backend.model)
    cam_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA,
                               pc.CAMERA_NAME)
    r.update_scene(backend.data, camera=cam_id)
    img = r.render()
    iio.imwrite(str(out_path), img)
    return out_path


def render_idle_motion_mp4(backend: WarpBackend, out_path: Path,
                           duration_s: float = 7.0, fps: int = 30) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    r = _renderer(backend.model)
    cam_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA,
                               pc.CAMERA_NAME)

    n_frames = int(duration_s * fps)
    physics_per_frame = max(1, int(round((1.0 / fps) / pc.SIM_TIMESTEP_S)))

    writer = iio.get_writer(str(out_path), fps=fps,
                            codec="libx264", quality=8)
    try:
        # gentle hover: oscillate shoulder_pan and wrist_3 a tiny bit
        import math
        ctrl = backend.data.ctrl.copy()
        # actuator ids
        aid_pan = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    "shoulder_pan")
        aid_wrist3 = mujoco.mj_name2id(backend.model,
                                       mujoco.mjtObj.mjOBJ_ACTUATOR, "wrist_3")
        base_pan = float(ctrl[aid_pan])
        base_w3 = float(ctrl[aid_wrist3])
        for k in range(n_frames):
            t = k / fps
            ctrl[aid_pan] = base_pan + 0.05 * math.sin(2 * math.pi * 0.3 * t)
            ctrl[aid_wrist3] = base_w3 + 0.15 * math.sin(2 * math.pi * 0.2 * t)
            backend.set_ctrl(ctrl)
            for _ in range(physics_per_frame):
                backend.step()
            r.update_scene(backend.data, camera=cam_id)
            writer.append_data(r.render())
    finally:
        writer.close()
    return out_path

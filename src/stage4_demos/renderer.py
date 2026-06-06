"""Offscreen camera renderer used by Stage 4 demo recording."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as iio
import mujoco
import numpy as np

from ..stage3_sim import physics_constants as pc


class FrameRenderer:
    def __init__(self, model, w: int = pc.RENDER_W, h: int = pc.RENDER_H,
                 cam_name: str = pc.CAMERA_NAME):
        self._r = mujoco.Renderer(model, height=h, width=w)
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            raise RuntimeError(f"camera {cam_name!r} not found in scene")
        self._cid = cid

    def render(self, data) -> np.ndarray:
        self._r.update_scene(data, camera=self._cid)
        return self._r.render()


def save_frames_dir(frames: Iterable[np.ndarray], out_dir: Path) -> int:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, f in enumerate(frames):
        iio.imwrite(str(out_dir / f"{i:04d}.png"), f)
        n += 1
    return n

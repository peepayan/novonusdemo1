"""MuJoCo Warp GPU backend wrapper.

Exposes a ``WarpBackend`` object that owns:
  - the CPU ``MjModel`` + ``MjData`` (for kinematics queries, rendering)
  - the GPU ``mujoco_warp`` model + data (for stepping physics on cuda:0)

The CPU/GPU copies are synchronised at each step: control is written on the
CPU side then pushed to GPU, the GPU steps, and ``qpos``/``qvel``/sensor data
is pulled back so MuJoCo's renderer and Python code see the post-step state.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

from . import physics_constants as pc
from . import scene as scene_mod


@dataclass
class BackendInfo:
    warp_version: str
    mujoco_warp_version: str
    cuda_device: str
    cuda_device_name: str
    cuda_sm: str
    solver: str
    integrator: str
    timestep_s: float
    nq: int
    nv: int
    nu: int

    def report(self) -> str:
        d = asdict(self)
        w = max(len(k) for k in d) + 1
        return "\n".join(f"{k:<{w}} : {v}" for k, v in d.items())


class WarpBackend:
    """GPU-backed MuJoCo Warp wrapper."""

    def __init__(self, scene_xml: Path = pc.SCENE_XML_OUT,
                 device: str = "cuda:0"):
        scene_xml = Path(scene_xml)
        if not scene_xml.exists():
            scene_mod.build_scene_xml(out_path=scene_xml)

        # CPU model + data — render / inspect from here
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

        # initialise Warp on the requested CUDA device
        wp.init()
        cuda_devs = wp.get_cuda_devices()
        if not cuda_devs:
            raise RuntimeError("No CUDA devices visible to Warp")
        self.wp_device = wp.get_device(device)
        wp.set_device(device)

        # GPU model + data
        import mujoco_warp as mjwp
        self._mjwp = mjwp
        self.m_gpu = mjwp.put_model(self.model)
        self.d_gpu = mjwp.put_data(self.model, self.data)

        # describe what we're actually running on
        dev0 = cuda_devs[0]
        sm = getattr(dev0, "arch", "") or getattr(dev0, "sm", "")
        self.info = BackendInfo(
            warp_version=wp.__version__,
            mujoco_warp_version=getattr(mjwp, "__version__", "unknown"),
            cuda_device=device,
            cuda_device_name=str(dev0),
            cuda_sm=str(sm),
            solver="Newton (mujoco_warp)",
            integrator="implicitfast (GPU)",
            timestep_s=float(self.model.opt.timestep),
            nq=int(self.model.nq),
            nv=int(self.model.nv),
            nu=int(self.model.nu),
        )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        # rebuild GPU data from the freshly reset CPU data
        self.d_gpu = self._mjwp.put_data(self.model, self.data)

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        """Write into ``data.ctrl`` then push to GPU."""
        np.copyto(self.data.ctrl, ctrl.astype(np.float64, copy=False))
        # re-upload the data (cheap: just the small state arrays)
        self.d_gpu = self._mjwp.put_data(self.model, self.data)

    def step(self) -> None:
        """Step physics on GPU, then pull state back to CPU for rendering."""
        self._mjwp.step(self.m_gpu, self.d_gpu)
        # pull qpos/qvel/sensor data back into self.data so the renderer sees
        # the post-step state. mujoco_warp.get_data_into copies into the CPU
        # MjData container in-place.
        self._mjwp.get_data_into(self.data, self.model, self.d_gpu)
        mujoco.mj_forward(self.model, self.data)

    def step_many(self, n: int) -> None:
        for _ in range(n):
            self.step()

    # convenience accessors --------------------------------------------

    def body_xpos(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return np.asarray(self.data.xpos[bid]).copy()

    def jaw_contact_force_N(self) -> float:
        """Return the magnitude of the contact force between the two jaws and
        the grape, summed and divided by 2 (so it reads as 'force on grape',
        not 'sum of forces on grape'). Capped at the crush threshold for
        display safety.
        """
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "grape_geom")
        lid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_jaw_geom")
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_jaw_geom")
        total = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if gid in (g1, g2) and (lid in (g1, g2) or rid in (g1, g2)):
                f6 = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, c, f6)
                total += float(abs(f6[0]))   # normal force
        return total / 2.0


def save_backend_report(backend: WarpBackend,
                        out_dir: Path = pc.STAGE3_OUT) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "backend.txt"
    p.write_text(backend.info.report() + "\n", encoding="utf-8")
    (out_dir / "backend.json").write_text(
        json.dumps(asdict(backend.info), indent=2), encoding="utf-8")
    return p


if __name__ == "__main__":
    b = WarpBackend()
    print(b.info.report())
    p = save_backend_report(b)
    print(f"wrote {p}")

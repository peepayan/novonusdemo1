"""Damped least-squares IK for the UR5e arm in the Stage 3 scene.

Targets the world-frame position of the ``tcp`` site (the midpoint between
the two gripper jaws). Solves for the 6 UR5e arm joints only; jaws and
grape free-joint dofs are held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


_ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


@dataclass
class IKResult:
    qpos: np.ndarray             # length-6 joint values
    final_error_m: float
    iterations: int
    converged: bool


def _arm_qpos_addr(model) -> np.ndarray:
    addrs = []
    for name in _ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(name)
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addrs, dtype=np.int64)


def solve_ik(model, data, target_xyz: np.ndarray,
             start_qpos: np.ndarray | None = None,
             max_iters: int = 200,
             tol_m: float = 1e-3,
             damping: float = 1e-2) -> IKResult:
    """Return the 6 UR5e joint values that put the ``tcp`` site at
    ``target_xyz`` (world coordinates).
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if site_id < 0:
        raise RuntimeError("'tcp' site missing from scene — rebuild scene.xml")
    addrs = _arm_qpos_addr(model)

    if start_qpos is not None:
        data.qpos[addrs] = np.asarray(start_qpos, dtype=np.float64)
    mujoco.mj_forward(model, data)

    jac_p = np.zeros((3, model.nv), dtype=np.float64)
    jac_r = np.zeros((3, model.nv), dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)

    # we update along the 6 arm joint dofs; build a per-joint dof index list.
    dof_addrs = []
    for name in _ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_addrs.append(int(model.jnt_dofadr[jid]))
    dof_addrs = np.asarray(dof_addrs, dtype=np.int64)

    last_err = float("inf")
    for it in range(max_iters):
        cur = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        err = target - cur
        e = float(np.linalg.norm(err))
        if e < tol_m:
            return IKResult(qpos=data.qpos[addrs].copy(),
                            final_error_m=e, iterations=it, converged=True)
        last_err = e
        mujoco.mj_jacSite(model, data, jac_p, jac_r, site_id)
        J = jac_p[:, dof_addrs]                  # 3 x 6
        # damped least squares
        JJt = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err)     # 6
        # take a measured step so we don't overshoot wrap-around limits
        step = 0.5
        data.qpos[addrs] = data.qpos[addrs] + step * dq
        mujoco.mj_forward(model, data)
    return IKResult(qpos=data.qpos[addrs].copy(),
                    final_error_m=last_err, iterations=max_iters,
                    converged=False)

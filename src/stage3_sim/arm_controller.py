"""Arm-controller — smooth interpolation between joint waypoints.

The controller converts a phase command (one of REST / REACHING / GRIPPING /
STABILIZING / RELEASING) plus a target grip force in Newtons into the 8-dim
control vector (6 UR5e joints + 2 gripper slides) for the next physics step.

Joint targets are blended toward the phase's waypoint using an
exponential-smoothing low-pass filter parameterised by the phase's halflife
so the visible motion is smooth and physically plausible. The gripper jaws
are driven by a small commanded inward displacement proportional to the
grip force — the actuator's force range caps the contact force at the
grape's crush threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import physics_constants as pc


# UR5e joint actuator names in order; jaws appended last.
_ARM_NAMES = ("shoulder_pan", "shoulder_lift", "elbow",
              "wrist_1", "wrist_2", "wrist_3")
_JAW_NAMES = ("left_jaw_act", "right_jaw_act")


@dataclass
class ControllerState:
    ctrl: np.ndarray            # length 8
    phase: str
    grip_force_N: float


def jaw_open_ctrl() -> float:
    return 0.0


def jaw_close_ctrl_for_force(grip_force_N: float) -> float:
    """Map a desired Newtonian grip force to a jaw closure command.

    The gripper jaws are position-controlled with kp = 180 N/m and a
    forcerange that caps the actuator output. A jaw target deeper than the
    grape's radius produces a positive position error which the actuator
    converts into an inward force; we choose the commanded position so the
    saturated actuator force equals the requested grip force.

    With kp = 180, an extra position error of dx (m) yields kp * dx Newtons
    of force on the jaw. So dx = grip_force_N / kp.
    """
    kp = 180.0
    base_close = 0.022   # close until just touching a 2 cm grape
    extra = max(0.0, float(grip_force_N)) / kp
    cmd = base_close + extra
    return float(min(0.045, cmd))


class ArmController:
    def __init__(self, model):
        self.model = model
        # actuator id lookup
        import mujoco
        self._aid = {
            n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in _ARM_NAMES + _JAW_NAMES
        }
        # current arm ctrl (smoothed)
        self._arm_q = np.array(pc.ARM_HOVER_QPOS, dtype=np.float64)
        # current jaw ctrl
        self._jaw_l = 0.0
        self._jaw_r = 0.0
        self._phase = "REST"

    @property
    def current_arm_q(self) -> np.ndarray:
        return self._arm_q.copy()

    def step(self, phase: str, grip_force_N: float, dt_s: float) -> ControllerState:
        spec = pc.PHASE_SPECS.get(phase, pc.PHASE_SPECS["REST"])
        target = np.array(spec.target_qpos, dtype=np.float64)
        # exponential smoothing toward target with the phase's halflife
        alpha = 1.0 - 0.5 ** (dt_s / max(spec.halflife_s, 1e-3))
        self._arm_q = self._arm_q + alpha * (target - self._arm_q)

        if spec.apply_grip:
            jc = jaw_close_ctrl_for_force(grip_force_N)
            # smooth toward closed
            ja = 1.0 - 0.5 ** (dt_s / 0.15)
            self._jaw_l += ja * (jc - self._jaw_l)
            self._jaw_r += ja * (jc - self._jaw_r)
        else:
            ja = 1.0 - 0.5 ** (dt_s / 0.20)
            self._jaw_l += ja * (jaw_open_ctrl() - self._jaw_l)
            self._jaw_r += ja * (jaw_open_ctrl() - self._jaw_r)

        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for q, name in zip(self._arm_q, _ARM_NAMES):
            ctrl[self._aid[name]] = float(q)
        ctrl[self._aid["left_jaw_act"]] = float(self._jaw_l)
        ctrl[self._aid["right_jaw_act"]] = float(self._jaw_r)
        self._phase = phase
        return ControllerState(ctrl=ctrl, phase=phase, grip_force_N=float(grip_force_N))


def distance_to_grape(backend) -> float:
    """L2 distance from the gripper midpoint to the grape body."""
    import mujoco
    lid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, "left_jaw")
    rid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, "right_jaw")
    gid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, "grape")
    mid = 0.5 * (backend.data.xpos[lid] + backend.data.xpos[rid])
    g = backend.data.xpos[gid]
    return float(np.linalg.norm(mid - g))

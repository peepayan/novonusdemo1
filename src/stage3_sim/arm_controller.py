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
    forcerange that caps the actuator output. The base closure is sized to
    bring each jaw to the grape's surface (jaw at ±0.030, grape radius
    0.020 -> each jaw must travel 0.010 m inward to just touch). Any
    requested grip force becomes additional commanded over-travel; the
    actuator force-range cap prevents that over-travel from ever exceeding
    the grape's crush threshold.
    """
    kp = 180.0
    base_close = 0.010    # each jaw travels 1 cm to touch a 2 cm-radius grape
    extra = max(0.0, float(grip_force_N)) / kp
    cmd = base_close + extra
    return float(min(0.045, cmd))


class ArmController:
    """Arm + gripper controller with a pick-and-place execution rule.

    The LSTM-driven `phase` selects the joint waypoint, but the gripper
    open/close decision honours two extra physical invariants so the demo
    actually performs a pick-and-place:

      - Jaws close (apply grip force) the moment we *enter* GRIPPING and
        stay closed throughout STABILIZING **and** RELEASING-while-in-transit.
        Without this, the grape would drop the instant the LSTM hands the
        phase over to RELEASING.
      - Jaws open only when the arm has actually arrived at the RELEASE
        waypoint (within ``release_open_dist_rad`` joint-space distance).
        That guarantees the grape is placed *over the target zone* before
        being let go.

    Grasp is enforced by toggling the ``grape_grip`` weld equality
    constraint defined in scene.xml. The constraint is enabled once the
    LSTM-driven phase reaches GRIPPING (after a brief settle so the jaws
    are around the grape) and disabled the moment we open the jaws to
    place. This makes the pick-and-place deterministic on top of the
    underlying soft-contact physics and is the visual proof the user sees
    on screen: the grape leaves the table with the gripper, travels over,
    and lands inside the target zone.
    """

    def __init__(self, model, data=None, release_open_dist_rad: float = 0.02):
        self.model = model
        import mujoco
        self._mj = mujoco
        self._aid = {
            n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in _ARM_NAMES + _JAW_NAMES
        }
        # current smoothed arm + jaw ctrl
        self._arm_q = np.array(pc.ARM_HOVER_QPOS, dtype=np.float64)
        self._jaw_l = 0.0
        self._jaw_r = 0.0
        self._phase = "REST"
        self._held_force_N = 0.0
        self._release_dist_thresh = float(release_open_dist_rad)
        # weld constraint id (grape grip)
        self._grip_eq_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, "grape_grip")
        self._grip_active = False
        # ticks spent in GRIPPING so far (to delay weld until jaws closed)
        self._gripping_ticks = 0

    @property
    def current_arm_q(self) -> np.ndarray:
        return self._arm_q.copy()

    @property
    def holding(self) -> bool:
        return self._held_force_N > 0.0

    def _set_grip(self, data, active: bool) -> None:
        if self._grip_eq_id < 0:
            return
        if active and not self._grip_active:
            # snap grape into the gripper centre before activating the weld
            # so the rest pose used by the constraint matches the current
            # arm pose — otherwise the constraint would jerk both bodies.
            wrist_id = self._mj.mj_name2id(
                self.model, self._mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
            grape_id = self._mj.mj_name2id(
                self.model, self._mj.mjtObj.mjOBJ_BODY, "grape")
            tcp_world = data.xpos[wrist_id] + data.xmat[wrist_id].reshape(3, 3) @ np.array(
                [0.0, 0.11, 0.0])
            # grape free joint qpos addresses
            grape_jid = self._mj.mj_name2id(
                self.model, self._mj.mjtObj.mjOBJ_JOINT, "grape_free")
            qadr = int(self.model.jnt_qposadr[grape_jid])
            data.qpos[qadr:qadr + 3] = tcp_world
            # zero grape velocity
            dadr = int(self.model.jnt_dofadr[grape_jid])
            data.qvel[dadr:dadr + 6] = 0.0
            self._mj.mj_forward(self.model, data)
            data.eq_active[self._grip_eq_id] = 1
        elif (not active) and self._grip_active:
            data.eq_active[self._grip_eq_id] = 0
        self._grip_active = bool(active)

    def step(self, phase: str, grip_force_N: float, dt_s: float,
             data=None) -> ControllerState:
        spec = pc.PHASE_SPECS.get(phase, pc.PHASE_SPECS["REST"])
        target = np.array(spec.target_qpos, dtype=np.float64)
        alpha = 1.0 - 0.5 ** (dt_s / max(spec.halflife_s, 1e-3))
        self._arm_q = self._arm_q + alpha * (target - self._arm_q)

        # phase-based grip + weld decisions
        if phase == "GRIPPING":
            keep_closed = True
            self._held_force_N = float(max(self._held_force_N, grip_force_N))
            self._gripping_ticks += 1
            # wait until jaws have had time to physically close, then weld
            if data is not None and self._gripping_ticks * dt_s > 0.40:
                self._set_grip(data, True)
        elif phase == "STABILIZING":
            keep_closed = True
            self._held_force_N = float(max(self._held_force_N, grip_force_N))
            if data is not None:
                self._set_grip(data, True)
        elif phase == "RELEASING" and self.holding:
            arm_err = float(np.linalg.norm(self._arm_q - target))
            if arm_err > self._release_dist_thresh:
                keep_closed = True
                if data is not None:
                    self._set_grip(data, True)
            else:
                keep_closed = False
                self._held_force_N = 0.0
                if data is not None:
                    self._set_grip(data, False)
        else:
            keep_closed = False
            self._held_force_N = 0.0
            self._gripping_ticks = 0
            if data is not None:
                self._set_grip(data, False)

        if keep_closed:
            f = max(grip_force_N, self._held_force_N)
            jc = jaw_close_ctrl_for_force(f)
            ja = 1.0 - 0.5 ** (dt_s / 0.10)
            self._jaw_l += ja * (jc - self._jaw_l)
            self._jaw_r += ja * (jc - self._jaw_r)
        else:
            ja = 1.0 - 0.5 ** (dt_s / 0.18)
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

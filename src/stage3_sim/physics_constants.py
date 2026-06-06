"""Stage 3 physics constants — single source of truth.

Every other module in Stage 3 imports values from here. Stage 7 and beyond
should also import from this module rather than redefining numbers, so the
crush threshold and grip-force safe band stay consistent across the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
ASSETS_DIR: Path = PROJECT_ROOT / "data" / "assets"
UR5E_DIR: Path = ASSETS_DIR / "ur5e"
UR5E_XML: Path = UR5E_DIR / "ur5e.xml"
SCENE_XML_OUT: Path = ASSETS_DIR / "scene.xml"
STAGE1_DIR: Path = PROJECT_ROOT / "outputs" / "stage1"
STAGE2_DIR: Path = PROJECT_ROOT / "outputs" / "stage2"
STAGE3_OUT: Path = PROJECT_ROOT / "outputs" / "stage3"

# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------

SIM_TIMESTEP_S: float = 0.002          # 500 Hz physics
SIM_HZ: float = 1.0 / SIM_TIMESTEP_S
RENDER_FPS: int = 30
RENDER_W: int = 640                    # feeds DINOv2 in Stage 7
RENDER_H: int = 480

# ---------------------------------------------------------------------------
# Scene geometry
# ---------------------------------------------------------------------------

TABLE_SIZE_XYZ: tuple[float, float, float] = (0.40, 0.30, 0.02)   # half-extents
TABLE_HEIGHT: float = 0.75
TABLE_CENTER_XY: tuple[float, float] = (0.55, 0.0)

ARM_BASE_XY: tuple[float, float] = (0.0, 0.0)
ARM_BASE_Z: float = TABLE_HEIGHT

GRAPE_RADIUS: float = 0.02
GRAPE_INIT_XY: tuple[float, float] = (0.55, 0.0)
GRAPE_INIT_Z: float = TABLE_HEIGHT + TABLE_SIZE_XYZ[2] + GRAPE_RADIUS + 0.001

TARGET_ZONE_XY: tuple[float, float] = (0.55, 0.15)
TARGET_ZONE_RADIUS: float = 0.04

# ---------------------------------------------------------------------------
# Grape fragility — the demo's thesis-proving constants
# ---------------------------------------------------------------------------

GRAPE_MASS_KG: float = 0.005                   # ~5 g, fresh grape
GRAPE_CRUSH_THRESHOLD_N: float = 6.0           # the 6 N ceiling
GRAPE_SAFE_GRIP_MIN_N: float = 2.0
GRAPE_SAFE_GRIP_MAX_N: float = 4.0
GRAPE_FRICTION: tuple[float, float, float] = (0.7, 0.05, 0.001)
GRAPE_SOLREF: tuple[float, float] = (0.02, 1.0)   # softer contact response
GRAPE_SOLIMP: tuple[float, float, float] = (0.6, 0.85, 0.001)

# Normalized force gauge value -> Newtons.
# The display gauge maxes out at 10 N so the safe grape grip band [2, 4] N
# reads as a LOW 0.2-0.4 on the gauge (the visual cue that we are gentle
# on a fragile object) while the 6 N crush threshold lands at 0.6 — still
# clearly visible above the operating band.
FORCE_GAUGE_FULL_SCALE_N: float = 10.0
FORCE_GAUGE_TO_NEWTONS: float = FORCE_GAUGE_FULL_SCALE_N   # 1.0 -> 10 N
GRIPPING_FORCE_GAUGE_BAND: tuple[float, float] = (
    GRAPE_SAFE_GRIP_MIN_N / FORCE_GAUGE_TO_NEWTONS,       # 0.20
    GRAPE_SAFE_GRIP_MAX_N / FORCE_GAUGE_TO_NEWTONS,       # 0.40
)
GAUGE_CRUSH_LINE: float = GRAPE_CRUSH_THRESHOLD_N / FORCE_GAUGE_TO_NEWTONS  # 0.6
# Visual target gauge value during gripping (kept LOW to communicate fragile).
GRIPPING_GAUGE_TARGET_LOW: float = 0.30


def newtons_to_gauge(n: float) -> float:
    return max(0.0, min(1.0, float(n) / FORCE_GAUGE_TO_NEWTONS))


def gauge_to_newtons(g: float) -> float:
    return max(0.0, min(1.0, float(g))) * FORCE_GAUGE_TO_NEWTONS


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

CAMERA_NAME: str = "scene_cam"
CAMERA_FOVY_DEG: float = 45.0

# ---------------------------------------------------------------------------
# Arm waypoints (radians) — neutral hover above table, descend, lift, etc.
# Joints order matches the UR5e XML: shoulder_pan, shoulder_lift, elbow,
# wrist_1, wrist_2, wrist_3.
# ---------------------------------------------------------------------------

import math as _m

ARM_HOME_QPOS: tuple[float, ...] = (
    0.0, -_m.pi / 2, _m.pi / 2, -_m.pi / 2, -_m.pi / 2, 0.0,
)
# Waypoints solved by `ik.solve_ik` to target the gripper TCP at the
# specified world-frame offsets above the grape:
#   HOVER     18 cm above the grape
#   PREGRIP    8 cm above the grape
#   GRIP        flush with the grape (jaws straddle it)
#   STABILIZE 12 cm above the grape (lift while held)
#   RELEASE   10 cm above + 15 cm in +y toward the target zone disc
# Numbers were extracted from the IK pass and frozen here so the demo
# is deterministic across runs.
ARM_HOVER_QPOS: tuple[float, ...] = (
    -0.2326, -1.2416, 1.8124, -1.8782, -1.6308, 0.0,
)
ARM_PREGRIP_QPOS: tuple[float, ...] = (
    -0.2355, -1.0626, 1.8250, -1.8987, -1.6225, 0.0,
)
ARM_GRIP_QPOS: tuple[float, ...] = (
    -0.2371, -0.9100, 1.8066, -1.9256, -1.6147, 0.0,
)
ARM_STABILIZE_QPOS: tuple[float, ...] = (
    -0.2349, -1.1354, 1.8256, -1.8892, -1.6254, 0.0,
)
ARM_RELEASE_QPOS: tuple[float, ...] = (
    0.0269, -1.0711, 1.7835, -1.9130, -1.5658, 0.0,
)

# ---------------------------------------------------------------------------
# Phase mapping: Stage 2 intent id -> Stage 3 arm phase name.
# Stage 2 intent ids: 0=REST 1=REACHING 2=GRIPPING 3=STABILIZING 4=RELEASING
# ---------------------------------------------------------------------------

INTENT_TO_PHASE: dict[int, str] = {
    0: "REST",
    1: "REACHING",
    2: "GRIPPING",
    3: "STABILIZING",
    4: "RELEASING",
}
PHASE_NAMES: tuple[str, ...] = (
    "REST", "REACHING", "GRIPPING", "STABILIZING", "RELEASING",
)


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    target_qpos: tuple[float, ...]
    # interpolation halflife in seconds (smaller = snappier transition)
    halflife_s: float
    # whether grip force is applied during this phase
    apply_grip: bool


PHASE_SPECS: dict[str, PhaseSpec] = {
    "REST":        PhaseSpec("REST",        ARM_HOVER_QPOS,     0.40, False),
    "REACHING":    PhaseSpec("REACHING",    ARM_PREGRIP_QPOS,   0.30, False),
    "GRIPPING":    PhaseSpec("GRIPPING",    ARM_GRIP_QPOS,      0.25, True),
    "STABILIZING": PhaseSpec("STABILIZING", ARM_STABILIZE_QPOS, 0.30, True),
    "RELEASING":   PhaseSpec("RELEASING",   ARM_RELEASE_QPOS,   0.35, False),
}


__all__ = [
    "PROJECT_ROOT", "ASSETS_DIR", "UR5E_DIR", "UR5E_XML", "SCENE_XML_OUT",
    "STAGE1_DIR", "STAGE2_DIR", "STAGE3_OUT",
    "SIM_TIMESTEP_S", "SIM_HZ", "RENDER_FPS", "RENDER_W", "RENDER_H",
    "TABLE_SIZE_XYZ", "TABLE_HEIGHT", "TABLE_CENTER_XY",
    "ARM_BASE_XY", "ARM_BASE_Z",
    "GRAPE_RADIUS", "GRAPE_INIT_XY", "GRAPE_INIT_Z",
    "TARGET_ZONE_XY", "TARGET_ZONE_RADIUS",
    "GRAPE_MASS_KG", "GRAPE_CRUSH_THRESHOLD_N",
    "GRAPE_SAFE_GRIP_MIN_N", "GRAPE_SAFE_GRIP_MAX_N",
    "GRAPE_FRICTION", "GRAPE_SOLREF", "GRAPE_SOLIMP",
    "FORCE_GAUGE_TO_NEWTONS", "GRIPPING_FORCE_GAUGE_BAND",
    "GRIPPING_GAUGE_TARGET_LOW",
    "newtons_to_gauge", "gauge_to_newtons",
    "CAMERA_NAME", "CAMERA_FOVY_DEG",
    "ARM_HOME_QPOS", "ARM_HOVER_QPOS", "ARM_PREGRIP_QPOS",
    "ARM_GRIP_QPOS", "ARM_STABILIZE_QPOS", "ARM_RELEASE_QPOS",
    "INTENT_TO_PHASE", "PHASE_NAMES", "PHASE_SPECS", "PhaseSpec",
]

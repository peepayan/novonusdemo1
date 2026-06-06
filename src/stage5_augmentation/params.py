"""Stage 5 augmentation parameter space.

Every range is a *named constant* with a comment explaining why that range
is physically plausible. Stage 6 verification imports the same ranges to
keep its thresholds aligned with what the engine actually perturbs.

A sampled :class:`AugmentationParams` is applied to one scene before
re-simulating a single baseline demo trajectory under those conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from ..stage3_sim import physics_constants as pc


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------

# Grape initial XY offset relative to the *baseline demo's* grape position.
# +/-3 cm keeps the grape inside the IK-reachable workspace used by the
# Stage 4 demos (which themselves vary by +/-3 cm), so the saved arm
# trajectory still passes over the grape.
GRAPE_XY_OFFSET_M: float = 0.030

# Random rotation about Z (the table normal). A sphere is rotationally
# symmetric in shape, but the contact dynamics are not — surface dimples
# and friction patches behave differently depending on which face touches
# the jaws, so we still randomize.
GRAPE_YAW_RAD_RANGE: tuple[float, float] = (0.0, 2.0 * np.pi)

# Grape mass: nominal 5 g, +/-20% covers the size variation of real
# pick-and-place targets (small connectors, berries, pills).
GRAPE_MASS_FRAC: float = 0.20

# Contact stiffness / softness — modulated by scaling solref and solimp.
# +/-30% covers part-to-part deformability variation. We tweak solref[0]
# (response time) since it has the largest visible effect.
GRAPE_STIFFNESS_FRAC: float = 0.30

# Friction coefficient between gripper jaws and grape. 0.4 = dry plastic
# on plastic, 0.9 = slightly moist / rubberised surface. Outside that
# range the grasp either slips out or sticks unphysically.
GRAPE_JAW_FRICTION_RANGE: tuple[float, float] = (0.4, 0.9)

# Gripper approach angle (wrist_3 rotation around the gripper axis).
# +/-10 degrees matches the Stage 4 controller's per-demo variation and
# is the largest reasonable operator inconsistency.
APPROACH_ANGLE_RAD: float = float(np.deg2rad(10.0))

# Ambient light intensity multiplier and headlight direction jitter.
# Visual-only — does not affect physics. Important for sim-to-real DINOv2
# feature robustness in Stage 7.
LIGHT_AMBIENT_FRAC: float = 0.20
LIGHT_DIR_JITTER_RAD: float = float(np.deg2rad(20.0))

# Table-top friction. +/-15% around the nominal 0.8 (set in scene.xml).
# Affects how far the grape slides after release.
TABLE_FRICTION_FRAC: float = 0.15

# Nominal values pulled from physics_constants / scene definitions.
NOMINAL_GRAPE_MASS_KG: float = pc.GRAPE_MASS_KG
NOMINAL_GRAPE_FRICTION0: float = pc.GRAPE_FRICTION[0]
NOMINAL_GRAPE_SOLREF0: float = pc.GRAPE_SOLREF[0]
NOMINAL_TABLE_FRICTION0: float = 0.8


# ---------------------------------------------------------------------------
# Dataclass + sampler
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AugmentationParams:
    """One sampled augmentation point."""
    grape_xy_offset_m: tuple[float, float]
    grape_yaw_rad: float
    grape_mass_kg: float
    grape_solref0: float
    grape_jaw_friction: float
    approach_angle_rad: float
    light_ambient_scale: float
    light_dir_jitter_rad: tuple[float, float]   # (dyaw, dpitch)
    table_friction0: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # tuples -> lists for JSON-friendliness
        d["grape_xy_offset_m"] = list(self.grape_xy_offset_m)
        d["light_dir_jitter_rad"] = list(self.light_dir_jitter_rad)
        return d


def sample_params(rng: np.random.Generator) -> AugmentationParams:
    """Sample one augmentation point. Each dimension is independent so
    Stage 6 can attribute failures to a single axis."""
    # Disk sample within the offset radius (uniform-over-disc, not uniform
    # in r, so the density does not pile at the centre).
    r = float(np.sqrt(rng.uniform(0.0, 1.0)) * GRAPE_XY_OFFSET_M)
    th = float(rng.uniform(0.0, 2.0 * np.pi))
    offset = (r * np.cos(th), r * np.sin(th))

    yaw = float(rng.uniform(*GRAPE_YAW_RAD_RANGE))
    mass = float(NOMINAL_GRAPE_MASS_KG * rng.uniform(
        1.0 - GRAPE_MASS_FRAC, 1.0 + GRAPE_MASS_FRAC))
    solref0 = float(NOMINAL_GRAPE_SOLREF0 * rng.uniform(
        1.0 - GRAPE_STIFFNESS_FRAC, 1.0 + GRAPE_STIFFNESS_FRAC))
    fric = float(rng.uniform(*GRAPE_JAW_FRICTION_RANGE))
    approach = float(rng.uniform(-APPROACH_ANGLE_RAD, APPROACH_ANGLE_RAD))
    amb = float(rng.uniform(1.0 - LIGHT_AMBIENT_FRAC,
                            1.0 + LIGHT_AMBIENT_FRAC))
    dyaw = float(rng.uniform(-LIGHT_DIR_JITTER_RAD, LIGHT_DIR_JITTER_RAD))
    dpit = float(rng.uniform(-LIGHT_DIR_JITTER_RAD, LIGHT_DIR_JITTER_RAD))
    tfric = float(NOMINAL_TABLE_FRICTION0 * rng.uniform(
        1.0 - TABLE_FRICTION_FRAC, 1.0 + TABLE_FRICTION_FRAC))

    return AugmentationParams(
        grape_xy_offset_m=offset,
        grape_yaw_rad=yaw,
        grape_mass_kg=mass,
        grape_solref0=solref0,
        grape_jaw_friction=fric,
        approach_angle_rad=approach,
        light_ambient_scale=amb,
        light_dir_jitter_rad=(dyaw, dpit),
        table_friction0=tfric,
    )


# ---------------------------------------------------------------------------
# Compact human-readable summary for grid overlays
# ---------------------------------------------------------------------------

def short_label(p: AugmentationParams) -> str:
    """Short overlay label for the augmentation grid."""
    return (f"f={p.grape_jaw_friction:.2f}  "
            f"m={p.grape_mass_kg*1000:.1f}g  "
            f"a={np.rad2deg(p.approach_angle_rad):+.1f}°")


__all__ = [
    "AugmentationParams", "sample_params", "short_label",
    "GRAPE_XY_OFFSET_M", "GRAPE_YAW_RAD_RANGE", "GRAPE_MASS_FRAC",
    "GRAPE_STIFFNESS_FRAC", "GRAPE_JAW_FRICTION_RANGE",
    "APPROACH_ANGLE_RAD", "LIGHT_AMBIENT_FRAC", "LIGHT_DIR_JITTER_RAD",
    "TABLE_FRICTION_FRAC",
    "NOMINAL_GRAPE_MASS_KG", "NOMINAL_GRAPE_FRICTION0",
    "NOMINAL_GRAPE_SOLREF0", "NOMINAL_TABLE_FRICTION0",
]

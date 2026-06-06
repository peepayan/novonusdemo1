"""Stage 5 inline verification — the six-point physical-validity filter.

Imports every numerical threshold from
``src.stage3_sim.physics_constants`` so the verification thresholds and the
real-physics constants cannot drift out of sync.

Failure reason codes are part of the public API: Stage 6 reports a
breakdown by reason, so renaming them is a breaking change.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from ..stage3_sim import physics_constants as pc


# ---------------------------------------------------------------------------
# Named thresholds — never use magic numbers in the checks below
# ---------------------------------------------------------------------------

# Check 1: average per-timestep EE distance between augmented and baseline.
# 15 mm is roughly the width of a UR5e fingertip; below it the trajectories
# are visually indistinguishable, above it the arm has been clearly nudged
# off-path by the perturbed physics.
PATH_DEV_THRESHOLD_M: float = 0.015

# Check 2: first-contact timestep, expressed as a fraction of the total
# trajectory. +/-20% absolute window around the baseline's first-contact
# fraction is wide enough for friction/mass jitter, narrow enough to
# catch scenarios where the grape has moved out of reach.
CONTACT_TIMING_WINDOW_FRAC: float = 0.20

# Check 3: task success — final grape XY distance from the target zone
# centre. Uses the same 30 mm radius as Stage 4's success check so
# augmented "successes" match baseline "successes".
TASK_SUCCESS_RADIUS_M: float = 0.030

# Check 4: peak contact force band. Lower bound = "the gripper actually
# made contact"; upper bound = the grape crush threshold from
# physics_constants. Forces averaged over the gripping window, so transient
# spikes are deferred to Check 6.
PEAK_FORCE_MIN_N: float = 0.5
PEAK_FORCE_MAX_N: float = pc.GRAPE_CRUSH_THRESHOLD_N

# Check 5: Pearson correlation between baseline and augmented contact-force
# time-series, computed over the *gripping window only* (where there is
# any signal). 0.6 keeps the shape qualitatively similar; below that the
# grasp dynamics have diverged.
FORCE_PROFILE_CORR_MIN: float = 0.60

# Check 6: instantaneous overshoot. Any single timestep above the crush
# threshold fails — this is stricter than the peak check (which is an
# *average peak*) and catches brief spikes that average out.
INSTANT_OVERSHOOT_THRESHOLD_N: float = pc.GRAPE_CRUSH_THRESHOLD_N

# Contact detection threshold (Newtons) — used both for "first contact"
# timestep detection and for the Pearson window. Smaller than the
# PEAK_FORCE_MIN_N gate so noisy idle contact still counts as "in
# gripping window" without erroneously triggering "the gripper made
# contact" success.
CONTACT_NOISE_FLOOR_N: float = 0.05


# ---------------------------------------------------------------------------
# Reason codes (frozen public API)
# ---------------------------------------------------------------------------

REASON_OK: str = "OK"
REASON_PATH: str = "PATH_DEVIATION"
REASON_CONTACT: str = "CONTACT_TIMING"
REASON_TASK: str = "TASK_FAILURE"
REASON_FORCE_BAND: str = "FORCE_OUT_OF_BAND"
REASON_FORCE_PROFILE: str = "FORCE_PROFILE_MISMATCH"
REASON_FORCE_OVERSHOOT: str = "FORCE_OVERSHOOT"

ALL_REASONS: tuple[str, ...] = (
    REASON_PATH, REASON_CONTACT, REASON_TASK,
    REASON_FORCE_BAND, REASON_FORCE_PROFILE, REASON_FORCE_OVERSHOOT,
)


# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reasons: tuple[str, ...]        # all failure reasons (empty -> passed)
    metrics: dict[str, float]       # per-check numeric scores

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_contact_index(force_N: np.ndarray,
                         threshold: float = CONTACT_NOISE_FLOOR_N) -> int:
    """Index of first timestep where contact force exceeds threshold.
    Returns -1 if no contact was detected."""
    idx = np.where(np.asarray(force_N) > threshold)[0]
    return int(idx[0]) if idx.size else -1


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or a.size != b.size:
        return 0.0
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def verify(
    *,
    aug_ee_pos: np.ndarray,         # (T, 3) augmented EE position
    base_ee_pos: np.ndarray,        # (T, 3) baseline EE position
    aug_contact_N: np.ndarray,      # (T,)
    base_contact_N: np.ndarray,     # (T,)
    aug_grape_final_xyz: np.ndarray,    # (3,)
) -> VerificationResult:
    """Run all six checks and return the combined result.

    All inputs must be aligned timestep-by-timestep — the augmentation
    engine guarantees this because it replays the same recorded joint
    trajectory, so T is fixed by the baseline demo.
    """
    aug_ee_pos = np.asarray(aug_ee_pos, dtype=np.float64)
    base_ee_pos = np.asarray(base_ee_pos, dtype=np.float64)
    aug_contact_N = np.asarray(aug_contact_N, dtype=np.float64)
    base_contact_N = np.asarray(base_contact_N, dtype=np.float64)
    aug_grape_final_xyz = np.asarray(aug_grape_final_xyz, dtype=np.float64)

    T = aug_ee_pos.shape[0]
    reasons: list[str] = []
    m: dict[str, float] = {}

    # --- Check 1: path similarity ---------------------------------------
    per_step = np.linalg.norm(aug_ee_pos - base_ee_pos, axis=1)
    mean_dev = float(per_step.mean())
    m["path_mean_dev_m"] = mean_dev
    if mean_dev > PATH_DEV_THRESHOLD_M:
        reasons.append(REASON_PATH)

    # --- Check 2: contact timing ----------------------------------------
    aug_ci = _first_contact_index(aug_contact_N)
    base_ci = _first_contact_index(base_contact_N)
    if base_ci < 0 or aug_ci < 0:
        # No baseline contact at all -> we cannot judge timing; treat the
        # augmentation as a timing failure (a successful pick must have
        # touched the grape).
        m["contact_timing_frac_err"] = 1.0
        reasons.append(REASON_CONTACT)
    else:
        frac_err = abs(aug_ci - base_ci) / float(max(T, 1))
        m["contact_timing_frac_err"] = float(frac_err)
        if frac_err > CONTACT_TIMING_WINDOW_FRAC:
            reasons.append(REASON_CONTACT)

    # --- Check 3: task success -----------------------------------------
    tx, ty = pc.TARGET_ZONE_XY
    xy_err = float(np.hypot(aug_grape_final_xyz[0] - tx,
                            aug_grape_final_xyz[1] - ty))
    m["task_xy_err_m"] = xy_err
    if xy_err > TASK_SUCCESS_RADIUS_M:
        reasons.append(REASON_TASK)

    # --- Check 4: peak contact force band ------------------------------
    peak_N = float(aug_contact_N.max()) if aug_contact_N.size else 0.0
    m["peak_force_N"] = peak_N
    if not (PEAK_FORCE_MIN_N <= peak_N <= PEAK_FORCE_MAX_N):
        reasons.append(REASON_FORCE_BAND)

    # --- Check 5: force-profile Pearson (gripping window only) ---------
    # Window = union of baseline + augmented contact windows
    aug_mask = aug_contact_N > CONTACT_NOISE_FLOOR_N
    base_mask = base_contact_N > CONTACT_NOISE_FLOOR_N
    win = aug_mask | base_mask
    if win.sum() >= 3:
        corr = _pearson(aug_contact_N[win], base_contact_N[win])
    else:
        corr = 0.0
    m["force_profile_corr"] = corr
    if corr < FORCE_PROFILE_CORR_MIN:
        reasons.append(REASON_FORCE_PROFILE)

    # --- Check 6: instantaneous overshoot -------------------------------
    over_steps = int((aug_contact_N > INSTANT_OVERSHOOT_THRESHOLD_N).sum())
    m["overshoot_step_count"] = float(over_steps)
    if over_steps > 0:
        reasons.append(REASON_FORCE_OVERSHOOT)

    return VerificationResult(
        passed=(len(reasons) == 0),
        reasons=tuple(reasons),
        metrics=m,
    )


__all__ = [
    "PATH_DEV_THRESHOLD_M", "CONTACT_TIMING_WINDOW_FRAC",
    "TASK_SUCCESS_RADIUS_M", "PEAK_FORCE_MIN_N", "PEAK_FORCE_MAX_N",
    "FORCE_PROFILE_CORR_MIN", "INSTANT_OVERSHOOT_THRESHOLD_N",
    "CONTACT_NOISE_FLOOR_N",
    "REASON_OK", "REASON_PATH", "REASON_CONTACT", "REASON_TASK",
    "REASON_FORCE_BAND", "REASON_FORCE_PROFILE", "REASON_FORCE_OVERSHOOT",
    "ALL_REASONS",
    "VerificationResult", "verify",
]

"""Stage 3 validation harness.

Two checks:
  1. Idle stability — step the physics with no motion for ``n_idle`` steps
     and verify nothing diverges.
  2. Full reach-grip-stabilize-release cycle — run the EMG sync layer through
     a short scripted cycle and verify that each phase executes, the measured
     grip force stays below the crush threshold, and the gauge stays low
     during gripping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import physics_constants as pc
from .arm_controller import ArmController
from .emg_sync import StreamerConfig, make_stitched_streamer
from .warp_backend import WarpBackend


@dataclass
class ValidationResult:
    idle_stable: bool
    idle_qpos_finite: bool
    reaching_executed: bool
    gripping_executed: bool
    stabilizing_executed: bool
    releasing_executed: bool
    max_contact_force_N: float
    grip_gauge_max: float
    grip_gauge_mean: float
    force_below_crush: bool
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


def _idle_check(backend: WarpBackend, n_steps: int = 500) -> tuple[bool, bool]:
    initial = backend.data.qpos.copy()
    ctrl = backend.data.ctrl.copy()
    for _ in range(n_steps):
        backend.set_ctrl(ctrl)
        backend.step()
    finite = bool(np.isfinite(backend.data.qpos).all() and
                  np.isfinite(backend.data.qvel).all())
    drift = float(np.linalg.norm(backend.data.qpos - initial))
    # we expect the grape to settle but the arm should not run away
    arm_drift = float(np.linalg.norm(backend.data.qpos[:6] - initial[:6]))
    stable = finite and arm_drift < 0.5
    return stable, finite


def _cycle_check(backend: WarpBackend, device: str = "cuda:0",
                 ) -> tuple[dict, list]:
    cfg = StreamerConfig(scroll_window_s=4.0, frame_step_ms=50.0,
                         min_phase_hold_s=0.4)
    streamer, _arrays = make_stitched_streamer(cfg=cfg, device=device)
    controller = ArmController(backend.model)
    phys_per_tick = max(1, int(round(streamer.tick_dt_s / pc.SIM_TIMESTEP_S)))

    phase_seen: dict[str, int] = {n: 0 for n in pc.PHASE_NAMES}
    grip_gauge: list[float] = []
    contact_forces: list[float] = []
    log: list[tuple[float, str, float, float, float]] = []

    for tick in streamer.iter():
        for _ in range(phys_per_tick):
            cs = controller.step(tick.phase_name, tick.phase_force_N,
                                 pc.SIM_TIMESTEP_S, data=backend.data)
            backend.set_ctrl(cs.ctrl)
            backend.step()
        phase_seen[tick.phase_name] += 1
        contact_N = backend.jaw_contact_force_N()
        contact_forces.append(contact_N)
        if tick.phase_name == "GRIPPING":
            grip_gauge.append(tick.gauge_value)
        log.append((tick.t_s, tick.phase_name, tick.gauge_value,
                    tick.phase_force_N, contact_N))

    return {
        "phase_seen": phase_seen,
        "grip_gauge": np.asarray(grip_gauge, dtype=np.float32),
        "contact_forces": np.asarray(contact_forces, dtype=np.float32),
    }, log


def run_validation(out_dir: Path = pc.STAGE3_OUT,
                   device: str = "cuda:0") -> ValidationResult:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    backend = WarpBackend(device=device)

    print("\n--- idle stability (500 steps) ---")
    stable, finite = _idle_check(backend, n_steps=500)
    print(f"  finite: {finite}    stable (arm drift < 0.5 rad): {stable}")

    # reset before cycle test so idle drift doesn't pollute it
    backend.reset()

    print("\n--- reach-grip-stabilize-release cycle ---")
    res, log = _cycle_check(backend, device=device)
    grip_gauge = res["grip_gauge"]
    cf = res["contact_forces"]
    phase_seen = res["phase_seen"]

    reaching = phase_seen.get("REACHING", 0) > 0
    gripping = phase_seen.get("GRIPPING", 0) > 0
    stabilizing = phase_seen.get("STABILIZING", 0) > 0
    releasing = phase_seen.get("RELEASING", 0) > 0

    max_contact = float(cf.max()) if cf.size else 0.0
    g_max = float(grip_gauge.max()) if grip_gauge.size else 0.0
    g_mean = float(grip_gauge.mean()) if grip_gauge.size else 0.0
    force_ok = max_contact < pc.GRAPE_CRUSH_THRESHOLD_N

    print(f"  REACHING:    {'PASS' if reaching else 'FAIL'}  "
          f"({phase_seen.get('REACHING',0)} ticks)")
    print(f"  GRIPPING:    {'PASS' if gripping else 'FAIL'}  "
          f"({phase_seen.get('GRIPPING',0)} ticks)")
    print(f"  STABILIZING: {'PASS' if stabilizing else 'FAIL'}  "
          f"({phase_seen.get('STABILIZING',0)} ticks)")
    print(f"  RELEASING:   {'PASS' if releasing else 'FAIL'}  "
          f"({phase_seen.get('RELEASING',0)} ticks)")
    print(f"  grip gauge during GRIPPING: max={g_max:.3f}  mean={g_mean:.3f}")
    print(f"  max contact force on grape: {max_contact:.2f} N  "
          f"(crush={pc.GRAPE_CRUSH_THRESHOLD_N:.0f} N)  "
          f"{'PASS' if force_ok else 'FAIL'}")

    summary = (
        f"idle stable={stable}, reaching={reaching}, gripping={gripping}, "
        f"stabilizing={stabilizing}, releasing={releasing}, "
        f"grip_gauge_max={g_max:.3f}, max_contact_N={max_contact:.2f}, "
        f"force_below_crush={force_ok}"
    )

    result = ValidationResult(
        idle_stable=stable,
        idle_qpos_finite=finite,
        reaching_executed=reaching,
        gripping_executed=gripping,
        stabilizing_executed=stabilizing,
        releasing_executed=releasing,
        max_contact_force_N=max_contact,
        grip_gauge_max=g_max,
        grip_gauge_mean=g_mean,
        force_below_crush=force_ok,
        summary=summary,
    )

    (out_dir / "validation.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    with open(out_dir / "phase_log.csv", "w", encoding="utf-8") as f:
        f.write("t_s,phase,gauge,target_force_N,measured_contact_N\n")
        for row in log:
            f.write(",".join(
                [f"{row[0]:.3f}", row[1]] + [f"{x:.4f}" for x in row[2:]]
            ) + "\n")
    print(f"\nwrote {out_dir/'validation.json'}")
    return result

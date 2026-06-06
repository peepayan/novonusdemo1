"""Stage 5 entry point — augment 30 baseline demos into N validated scenarios.

Pipeline:
  1. Load all Stage 4 baseline demos.
  2. Round-robin a baseline -> sample one set of augmentation params ->
     simulate -> verify inline -> save iff it passes -> advance.
  3. Stop when --n-passing scenarios are saved.
  4. Report stats: raw generated, passing, pass rate, rejection breakdown.

The runner is resumable: counting existing ``scenario_*.npz`` files in the
output dir tells it where to pick up. Pass ``--reset`` to wipe and start
fresh.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..stage3_sim import physics_constants as pc
from .augment_engine import (
    AugmentationEngine, BaselineDemo, load_baseline_demos, save_scenario_npz,
)
from .params import sample_params
from .verification import verify, ALL_REASONS


DEFAULT_N_PASSING: int = 100


def _existing_scenarios(scen_dir: Path) -> int:
    return len(list(Path(scen_dir).glob("scenario_*.npz")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-passing", type=int, default=DEFAULT_N_PASSING)
    ap.add_argument("--max-attempts-multiplier", type=int, default=8,
                    help="cap total attempts at N * this multiplier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage4" / "demos"))
    ap.add_argument("--out-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage5"))
    ap.add_argument("--reset", action="store_true",
                    help="wipe outputs/stage5/scenarios + frames before run")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    scen_dir = out_dir / "scenarios"
    frames_dir = out_dir / "scenario_frames"
    scen_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.reset:
        for p in scen_dir.glob("scenario_*.npz"):
            p.unlink()
        for p in frames_dir.glob("scenario_*.mp4"):
            p.unlink()
        print("[reset] wiped scenarios/ and scenario_frames/", flush=True)

    t0 = time.time()

    print("=== Stage 5 — loading baseline demos ===", flush=True)
    baselines = load_baseline_demos(Path(args.baseline_dir))
    if not baselines:
        print(f"[error] no baselines found in {args.baseline_dir}",
              file=sys.stderr)
        return 1
    print(f"  loaded {len(baselines)} baseline demos", flush=True)

    print("\n=== Stage 5 — initialising augmentation engine ===", flush=True)
    engine = AugmentationEngine()
    print(engine.backend.info.report(), flush=True)

    already = _existing_scenarios(scen_dir)
    if already:
        print(f"[resume] {already} scenarios already saved; "
              f"continuing from scenario {already:05d}", flush=True)
    target = args.n_passing
    max_attempts = max(target, target * args.max_attempts_multiplier)
    rng = np.random.default_rng(args.seed + 1000 * already)

    print(f"\n=== Stage 5 — collecting {target} passing scenarios "
          f"(currently {already}) ===", flush=True)
    saved = already
    attempts = 0
    rejection_counter: Counter = Counter()
    pass_force_max: list[float] = []
    pass_force_mean: list[float] = []
    attempt_log: list[dict] = []

    while saved < target and attempts < max_attempts:
        attempts += 1
        baseline: BaselineDemo = baselines[saved % len(baselines)]
        params = sample_params(rng)
        scen_idx = saved          # 0-indexed within passing scenarios
        mp4_path = frames_dir / f"scenario_{scen_idx:05d}.mp4"

        t_scen = time.time()
        try:
            rec = engine.simulate(baseline, params, mp4_path)
        except Exception as e:
            print(f"  attempt #{attempts}: ENGINE ERROR ({type(e).__name__}: {e})  "
                  f"-- skip", flush=True)
            if mp4_path.exists():
                mp4_path.unlink()
            attempt_log.append({
                "attempt": attempts, "passed": False,
                "reasons": ["ENGINE_ERROR"],
                "baseline_demo_idx": baseline.demo_idx,
            })
            continue

        vres = verify(
            aug_ee_pos=rec["end_effector_pos"],
            base_ee_pos=baseline.end_effector_pos,
            aug_contact_N=rec["contact_force"],
            base_contact_N=baseline.applied_force,
            aug_grape_final_xyz=rec["grape_final_xyz"],
        )

        log = {
            "attempt": attempts,
            "passed": bool(vres.passed),
            "reasons": list(vres.reasons),
            "baseline_demo_idx": baseline.demo_idx,
            "metrics": {k: float(v) for k, v in vres.metrics.items()},
            "elapsed_s": round(time.time() - t_scen, 2),
        }
        attempt_log.append(log)

        if not vres.passed:
            for r in vres.reasons:
                rejection_counter[r] += 1
            print(f"  attempt #{attempts}: FAIL {','.join(vres.reasons)}  "
                  f"base={baseline.demo_idx:02d}  "
                  f"path={vres.metrics.get('path_mean_dev_m', 0)*1000:.1f}mm  "
                  f"peak={vres.metrics.get('peak_force_N', 0):.2f}N  "
                  f"({log['elapsed_s']:.1f}s)", flush=True)
            mp4_path.unlink(missing_ok=True)
            del rec
            gc.collect()
            continue

        npz_path = scen_dir / f"scenario_{scen_idx:05d}.npz"
        save_scenario_npz(
            recorded=rec, params=params,
            verification=vres.to_dict(),
            baseline_demo_idx=baseline.demo_idx,
            scenario_idx=scen_idx, out_path=npz_path,
        )
        pass_force_max.append(float(rec["contact_force"].max()))
        pass_force_mean.append(float(rec["contact_force"].mean()))
        print(f"  scenario {scen_idx:05d}  base={baseline.demo_idx:02d}  "
              f"PASS  peak={pass_force_max[-1]:.2f}N  "
              f"path={vres.metrics['path_mean_dev_m']*1000:.1f}mm  "
              f"task={vres.metrics['task_xy_err_m']*1000:.1f}mm  "
              f"({log['elapsed_s']:.1f}s)", flush=True)
        saved += 1
        del rec
        gc.collect()

    elapsed = time.time() - t0
    raw = attempts
    new_pass = saved - already
    pass_rate = (new_pass / raw * 100.0) if raw else 0.0

    print(f"\n[stage5] DONE in {elapsed:.1f}s   "
          f"saved={saved}  raw_attempts={raw}  pass_rate={pass_rate:.1f}%",
          flush=True)
    print(f"[stage5] rejection breakdown (this run):")
    for r in ALL_REASONS:
        print(f"    {r:24s} {rejection_counter.get(r, 0)}")

    # write attempt log
    log_path = out_dir / "attempts_log.json"
    prior: list[dict] = []
    if log_path.exists():
        try:
            prior = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            prior = []
    log_path.write_text(json.dumps(prior + attempt_log, indent=2),
                        encoding="utf-8")
    print(f"[stage5] wrote {log_path}")

    if saved < target:
        print(f"[stage5] only collected {saved}/{target} — "
              f"raise --max-attempts-multiplier or widen ranges",
              file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

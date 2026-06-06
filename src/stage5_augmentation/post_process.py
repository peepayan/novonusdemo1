"""Stage 5 post-processing — build the grid PNG, stats JSON, summary TXT,
README.md, and combined dataset_index.json.

Run after ``run_stage5`` has collected its target N passing scenarios.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..stage3_sim import physics_constants as pc
from .augment_engine import load_baseline_demos
from .dataset import (
    build_augmentation_grid, compute_stats, write_dataset_index,
    write_readme, write_stats_json, write_summary_txt,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage5"))
    ap.add_argument("--baseline-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage4" / "demos"))
    ap.add_argument("--grid-seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    scen_dir = out_dir / "scenarios"
    frames_dir = out_dir / "scenario_frames"
    if not list(scen_dir.glob("scenario_*.npz")):
        print(f"[error] no scenarios in {scen_dir}", file=sys.stderr)
        return 1

    baselines = load_baseline_demos(Path(args.baseline_dir))
    n_baselines = len(baselines)

    print("=== Stage 5 — computing stats ===", flush=True)
    stats = compute_stats(
        scen_dir=scen_dir,
        attempts_log_path=out_dir / "attempts_log.json",
        n_baselines=n_baselines,
    )
    p = write_stats_json(stats, out_dir / "augmentation_stats.json")
    print(f"  wrote {p}", flush=True)
    p = write_summary_txt(stats, out_dir / "dataset_summary.txt")
    print(f"  wrote {p}", flush=True)
    p = write_readme(stats, out_dir / "README.md")
    print(f"  wrote {p}", flush=True)

    print("\n=== Stage 5 — building augmentation grid PNG ===", flush=True)
    p = build_augmentation_grid(
        scen_dir=scen_dir, frames_dir=frames_dir,
        out_png=out_dir / "augmentation_grid.png",
        seed=args.grid_seed,
    )
    print(f"  wrote {p}", flush=True)

    print("\n=== Stage 5 — writing combined dataset_index.json ===", flush=True)
    p = write_dataset_index(
        stage4_demos_dir=pc.PROJECT_ROOT / "outputs" / "stage4" / "demos",
        stage5_scen_dir=scen_dir,
        stage5_frames_dir=frames_dir,
        out_path=out_dir / "dataset_index.json",
    )
    print(f"  wrote {p}", flush=True)

    print(f"\n[stage5] post-process complete.   "
          f"raw={stats['raw_scenarios_generated']}  "
          f"pass={stats['passing_scenarios']}  "
          f"rate={stats['pass_rate_pct']:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

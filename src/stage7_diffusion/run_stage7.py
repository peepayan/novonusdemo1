"""Stage 7 end-to-end entry point.

Pipeline (skips any step whose outputs already exist):
  1. Pair EMG into the 100 Stage 5 scenarios (~1 min, idempotent).
  2. Cache DINOv2 + LSTM + state + action features (~12 min, idempotent).
  3. Train the EMG-conditioned diffusion policy (~30-60 min).
  4. Execute the trained policy on 5 fresh test scenarios with a
     force-gauge overlay (~5-15 min).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..stage3_sim import physics_constants as pc
from . import cache_features as _cache
from . import execute as _exec
from . import pair_stage5_emg as _pair
from . import train as _train


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pair", action="store_true")
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-execute", action="store_true")
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    out_root = pc.PROJECT_ROOT / "outputs" / "stage7"
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_pair:
        print("\n##### STAGE 7 / step 1: pair EMG into Stage 5 #####",
              flush=True)
        if _pair.main([]) != 0:
            return 1

    if not args.skip_cache:
        print("\n##### STAGE 7 / step 2: cache DINOv2 + LSTM features #####",
              flush=True)
        if _cache.main([]) != 0:
            return 2

    if not args.skip_train:
        print("\n##### STAGE 7 / step 3: train EMG-conditioned policy #####",
              flush=True)
        if _train.main([
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
        ]) != 0:
            return 3

    if not args.skip_execute:
        print("\n##### STAGE 7 / step 4: execute policy on test scenarios "
              "#####", flush=True)
        if _exec.main(["--n-trials", str(args.n_trials)]) != 0:
            return 4

    print("\n[stage7] pipeline complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

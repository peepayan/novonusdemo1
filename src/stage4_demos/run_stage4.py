"""Stage 4 entry point — collect 30 successful pick-and-place demos.

Pipeline:
  1. Load Stage 1 EMG arrays + Stage 2 LSTM.
  2. Probe the val EMG with the LSTM once (cached for all 30 demos).
  3. For each demo idx 0..29:
       a. sample random per-demo parameters
       b. run the scripted controller in the MuJoCo Warp scene
       c. evaluate success (grape in target zone, force below crush,
          all phases executed)
       d. if not successful, regenerate with a new RNG seed; otherwise
          build the EMG pairing, save .npz + frames PNG dir, advance
  4. Write the dataset summary, demo grid PNG, and sample MP4.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..stage2_lstm import dataset as ds
from ..stage3_sim import physics_constants as pc
from ..stage3_sim.warp_backend import WarpBackend
from .controller import sample_demo_params
from .demo_recorder import record_one_demo, save_demo_npz
from .dataset import make_demo_grid, write_dataset_summary
from .emg_pairing import probe_val_emg


N_TARGET_DEMOS: int = 30
MAX_TOTAL_ATTEMPTS: int = 200


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-demos", type=int, default=N_TARGET_DEMOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(pc.PROJECT_ROOT / "outputs"
                                              / "stage4"))
    ap.add_argument("--skip-sample-mp4", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    demos_dir = out_dir / "demos"; demos_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    print("=== Stage 4 — loading Stage 1 EMG arrays ===")
    arrays = ds.load_stage1_arrays(pc.STAGE1_DIR)
    print(f"  T = {arrays.n_samples:,}")

    print("\n=== Stage 4 — loading Stage 2 LSTM ===")
    from ..stage3_sim.emg_sync import load_lstm
    model = load_lstm()
    print(f"  loaded {pc.STAGE2_DIR/'lstm_best.pt'}")

    print("\n=== Stage 4 — probing val EMG with LSTM (one-time) ===")
    probe = probe_val_emg(arrays, model)
    print(f"  windows = {probe.ends.shape[0]:,}    "
          f"in-val = {(probe.preds >= 0).sum():,}")

    print("\n=== Stage 4 — initialising Warp backend ===")
    backend = WarpBackend()
    print(backend.info.report())

    rng = np.random.default_rng(args.seed)
    existing = sorted(demos_dir.glob("demo_*.npz"))
    existing_idx = {int(p.stem.split("_")[1]) for p in existing}
    saved = len(existing_idx)
    attempts = 0
    attempt_log: list[dict] = []
    if saved:
        # advance the rng so subsequent demos see *new* variations rather
        # than re-rolling the same ones we already saved.
        for prev_id in range(saved):
            _ = sample_demo_params(demo_id=prev_id, rng=rng)
        print(f"[resume] found {saved} demos already on disk; "
              f"continuing from demo {saved:03d}", flush=True)
    print(f"\n=== Stage 4 — collecting {args.n_demos} demos ===", flush=True)
    while saved < args.n_demos and attempts < MAX_TOTAL_ATTEMPTS:
        attempts += 1
        t_demo = time.time()
        params = sample_demo_params(demo_id=saved, rng=rng)
        ok, data_dict, check = record_one_demo(
            backend, params, arrays, probe, model, rng,
        )
        attempt_log.append({
            "attempt": attempts,
            "ok": bool(ok),
            "grape_in_zone": bool(check.grape_in_zone),
            "grape_xy_err_mm": float(check.grape_xy_err_m * 1000),
            "max_contact_force_N": float(check.max_contact_force_N),
            "all_phases_executed": bool(check.all_phases_executed),
        })
        if not ok:
            print(f"  attempt #{attempts}: FAIL  "
                  f"(in_zone={check.grape_in_zone} "
                  f"err={check.grape_xy_err_m*1000:.1f}mm "
                  f"contact_max={check.max_contact_force_N:.2f}N)  "
                  f"-- regenerate ({time.time()-t_demo:.1f}s)",
                  flush=True)
            # free per-attempt memory
            del data_dict
            gc.collect()
            continue
        npz_path = demos_dir / f"demo_{saved:03d}.npz"
        frames_dir = demos_dir / f"demo_{saved:03d}_frames"
        frames = data_dict.pop("_frames")
        data_dict.pop("_check", None)
        save_demo_npz(data_dict, frames, npz_path, frames_dir)
        print(f"  demo {saved:03d}  attempt #{attempts}  "
              f"err={check.grape_xy_err_m*1000:.1f}mm  "
              f"contact={check.max_contact_force_N:.2f}N  "
              f"applied={params.grip_force_N:.2f}N "
              f"({time.time()-t_demo:.1f}s) -> {npz_path.name}",
              flush=True)
        saved += 1
        # release the big arrays + frames + GC before the next demo
        del data_dict, frames
        gc.collect()

    if saved < args.n_demos:
        print(f"\n[stage4] FAILED to collect {args.n_demos} demos after "
              f"{attempts} attempts. Saved {saved}.")
        (out_dir / "attempts_log.json").write_text(
            json.dumps(attempt_log, indent=2), encoding="utf-8")
        return 1

    print(f"\n[stage4] collected {saved} demos in {attempts} attempts "
          f"(success rate {saved/attempts*100:.1f}%)")
    (out_dir / "attempts_log.json").write_text(
        json.dumps(attempt_log, indent=2), encoding="utf-8")

    print("\n=== Stage 4 — dataset summary ===")
    stats = write_dataset_summary(demos_dir, out_dir / "dataset_summary.txt")
    print(json.dumps(stats, indent=2))

    print("\n=== Stage 4 — demo grid PNG ===")
    grid_path = make_demo_grid(demos_dir, out_dir / "demo_grid.png")
    print(f"  wrote {grid_path}")

    if not args.skip_sample_mp4:
        print("\n=== Stage 4 — sample MP4 (3 demos) ===")
        from .sample_mp4 import render_sample_mp4
        sample = render_sample_mp4(
            demos_dir, out_dir / "demo_sample.mp4",
            demo_indices=(0, max(0, args.n_demos // 2),
                          max(0, args.n_demos - 1)),
        )
        print(f"  wrote {sample}")

    print(f"\n[stage4] DONE in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

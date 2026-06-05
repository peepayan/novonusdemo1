"""Stage 2 add-on entry point: inspect E3 -> preprocess -> dataset -> train
both an envelope-only baseline and a rich-feature model -> evaluate on the
held-out repetition and produce the headline accuracy, plots, and summary."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import data_e3 as d3
from . import preprocess_e3 as p3
from . import dataset_e3 as ds3
from . import train_e3 as t3
from . import evaluate_e3 as e3
from .model_e3 import E3ForceLSTM

DEFAULT_E3_MAT = r"C:\Users\deepa\novonusdemo1\data\ninapro_db2\DB2_s1\S1_E3_A1.mat"
DEFAULT_OUT = r"C:\Users\deepa\novonusdemo1\outputs\stage2\force_model_e3"


README_TEMPLATE = """# Stage 2 add-on — dedicated EMG-to-force regressor (E3)

This model is trained on the **real measured force** in DB2 block E3, not on
the EMG-amplitude proxy used by the main Stage 2 intent classifier. It is an
honest evaluation of EMG-to-real-force prediction capability on a held-out
repetition (rep 6) that the model never sees during training.

## Inputs

- Source : `data/ninapro_db2/DB2_s1/S1_E3_A1.mat`
- EMG    : 12-channel raw -> Stage 1 DSP chain (HP20-Notch-BP20-450-Rectify-RMS100ms-MVCnorm),
           plus the post-bandpass pre-rectify signal kept for rich features.
- Accel  : 36-channel, low-pass 5 Hz (same as Stage 1).
- Force  : 6-channel real measured force -> L2 magnitude across channels ->
           min-max normalized to [0, 1] using (1st, 99th) percentile bounds.
           Scaling factor (`force_lo`, `force_hi`) saved in the checkpoint so
           predictions can be converted back to the original calibrated force units:

           ```python
           real_force = norm_pred * (force_hi - force_lo) + force_lo
           ```

E3 has no glove modality, so this model uses **EMG envelope + accelerometer
only**. That is documented and is acceptable: this is a dedicated EMG-to-force
regressor, not a multimodal classifier.

## Features

Per-window features (1 vector per window, computed on the post-bandpass
pre-rectified EMG, 12 channels × 5 features = 60 dim):

- **RMS**  : sqrt(mean(x²))
- **MAV**  : mean(|x|)
- **WL**   : sum(|x[t+1] - x[t]|), waveform length
- **VAR**  : variance
- **MNF**  : mean spectral frequency from |X(f)|²

Standardized using train-set statistics (no test leakage).

## Split

Repetition-based, Ninapro protocol:

- reps 1-5 -> **train**
- rep 6   -> **held-out test**  (model never trains on it)
- rep 0   -> dropped (1.6k-sample gap)

A window is assigned to the fold of its END sample; cross-boundary windows are
dropped.

## Model

2-layer LSTM (hidden 128, dropout 0.2) over (400, 48) windows (EMG envelope +
accel). Optional concat of the 60-dim rich-feature vector to the shared 128-dim
hidden state. MLP head 128(+60) -> 64 -> 1 with sigmoid (target in [0,1]).
Hidden state exposed via `model.get_hidden_state(x)` (consistent with the
intent classifier).

## Outputs

| file | what it is |
|------|------------|
| `force_model_best_rich.pt`         | best rich-feature checkpoint (weights, scaling, hparams) |
| `force_model_best_envelope.pt`     | best envelope-only baseline checkpoint |
| `force_prediction_e3.png`          | real-vs-predicted force overlay, real units, rich model |
| `force_scatter_e3.png`             | predicted vs true scatter, real units, R² annotated |
| `force_accuracy_summary.txt`       | full metric report, envelope-only vs rich, honest interpretation |
| `README.md`                        | this file |

## Headline results

{HEADLINE}

## Caveats

E3 is sustained force-task data from one subject. Real-world contact-rich
accuracy will differ; this demonstrates the EMG-to-real-force prediction
capability under matched conditions, not field accuracy.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3-mat", default=DEFAULT_E3_MAT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--skip-envelope-baseline", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. inspect ----------------------------------------------------------
    print("=== E3 force model — inspect ===")
    block = d3.load_e3(args.e3_mat)
    raw_report = block.report()
    print(raw_report)

    # 2. preprocess -------------------------------------------------------
    print("\n=== E3 force model — preprocess ===")
    proc = p3.preprocess(block)

    # 3a. rich-feature run ----------------------------------------------
    print("\n=== E3 force model — split (rich) ===")
    tr_r, te_r, meta_r = ds3.build_splits(proc, use_rich_features=True)
    print(meta_r.report())

    cfg = t3.E3TrainConfig(max_epochs=args.epochs, batch_size=args.batch_size,
                           lr=args.lr, patience=args.patience)
    print("\n=== E3 force model — train (rich features) ===")
    model_rich, hist_rich, info_rich = t3.train(
        tr_r, te_r, meta_r, out_dir, cfg, variant_name="rich",
    )

    # 3b. envelope-only baseline ---------------------------------------
    info_env = None
    metrics_env = None
    if not args.skip_envelope_baseline:
        print("\n=== E3 force model — split (envelope only) ===")
        tr_e, te_e, meta_e = ds3.build_splits(proc, use_rich_features=False)
        print(meta_e.report())
        cfg_e = t3.E3TrainConfig(max_epochs=args.epochs, batch_size=args.batch_size,
                                 lr=args.lr, patience=args.patience)
        print("\n=== E3 force model — train (envelope only) ===")
        _, _, info_env = t3.train(
            tr_e, te_e, meta_e, out_dir, cfg_e, variant_name="envelope",
        )
        metrics_env = e3.compute_metrics(info_env["best_ys"], info_env["best_ps"], meta_e)

    # 4. evaluate -------------------------------------------------------
    print("\n=== E3 force model — evaluation (rich, held-out test) ===")
    metrics_rich = e3.compute_metrics(info_rich["best_ys"], info_rich["best_ps"], meta_r)
    print(f"  R^2              = {metrics_rich.r2:+.4f}")
    print(f"  Pearson r        = {metrics_rich.pearson_r:+.4f}")
    print(f"  RMSE  (norm)     = {metrics_rich.rmse_normalized:.4f}")
    print(f"  RMSE  (real)     = {metrics_rich.rmse_real:.4f}")
    print(f"  MAE   (real)     = {metrics_rich.mae_real:.4f}")
    print(f"  MAPE  (real, %)  = {metrics_rich.mape_real:.2f}")
    if metrics_env is not None:
        print()
        print(f"  envelope baseline R^2      = {metrics_env.r2:+.4f}")
        print(f"  envelope baseline RMSE (real) = {metrics_env.rmse_real:.4f}")
        print(f"  -> rich gain in R^2        = {metrics_rich.r2 - metrics_env.r2:+.4f}")
        print(f"  -> rich gain in RMSE (real) = {metrics_rich.rmse_real - metrics_env.rmse_real:+.4f}")

    # 5. plots ----------------------------------------------------------
    e3.plot_overlay_real_units(info_rich["best_ys"], info_rich["best_ps"],
                               meta_r,
                               out_dir / "force_prediction_e3.png")
    e3.plot_scatter(info_rich["best_ys"], info_rich["best_ps"],
                    meta_r, metrics_rich,
                    out_dir / "force_scatter_e3.png")

    # 6. summary --------------------------------------------------------
    from dataclasses import asdict
    e3.write_summary(
        out_dir / "force_accuracy_summary.txt",
        raw_report=raw_report,
        meta=meta_r,
        metrics_rich=metrics_rich,
        metrics_envelope=metrics_env,
        hparams=asdict(cfg),
    )

    # 7. README ----------------------------------------------------------
    headline = (
        f"- Rich-feature R^2 : **{metrics_rich.r2:+.3f}**\n"
        f"- Rich-feature Pearson r : **{metrics_rich.pearson_r:+.3f}**\n"
        f"- Rich-feature RMSE (real force units) : **{metrics_rich.rmse_real:.3f}**\n"
        f"- Rich-feature MAE  (real force units) : **{metrics_rich.mae_real:.3f}**\n"
    )
    if metrics_env is not None:
        headline += (
            f"\n_Envelope-only baseline R²={metrics_env.r2:+.3f}, "
            f"RMSE_real={metrics_env.rmse_real:.3f} -> "
            f"rich features add ΔR²={metrics_rich.r2 - metrics_env.r2:+.3f}._\n"
        )
    (out_dir / "README.md").write_text(
        README_TEMPLATE.replace("{HEADLINE}", headline), encoding="utf-8",
    )

    print(f"\n[e3] DONE in {time.time()-t0:.1f}s")
    print(f"[e3] outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

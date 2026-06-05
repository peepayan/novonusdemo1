"""Stage 2 entry point: end-to-end pipeline.

Steps:
  1. Print + save the class mapping.
  2. Load Stage 1 arrays, build windowed train/val splits.
  3. Train the LSTM (joint CE + force MSE).
  4. Evaluate, write confusion matrix + training summary.
  5. Render the live visualization MP4 (Segment 2 of the demo video).
  6. Run the E3 force-validation (independent honesty check).
  7. Write README.md for outputs/stage2/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import class_mapping as cm
from . import dataset as ds
from . import train as tr
from . import visualize as viz
from . import validate_e3 as v3
from .model import IntentForceLSTM

DEFAULT_STAGE1_DIR = r"C:\Users\deepa\novonusdemo1\outputs\stage1"
DEFAULT_OUT_DIR = r"C:\Users\deepa\novonusdemo1\outputs\stage2"
DEFAULT_E3_MAT = r"C:\Users\deepa\novonusdemo1\data\ninapro_db2\DB2_s1\S1_E3_A1.mat"


def write_class_mapping_json(out_dir: Path) -> None:
    p = out_dir / "class_mapping.json"
    p.write_text(json.dumps(cm.mapping_json_dict(), indent=2), encoding="utf-8")
    print(f"[stage2] wrote {p}")


def write_readme(out_dir: Path, summary: dict) -> None:
    txt = f"""# Stage 2 outputs — multimodal LSTM intent + force-intensity model

Trained on subject `DB2_s1`, blocks E1+E2 (Stage 1 outputs).

## Files

| file | what it is |
|------|------------|
| `lstm_best.pt` | best checkpoint: model weights, intent class mapping, hyperparameters |
| `loss_curve.png` | train/val loss, val accuracy, val force-MSE per epoch |
| `confusion_matrix.png` | row-normalised intent confusion matrix on the val split |
| `intent_and_force_preview.mp4` | live visualization: EMG + intent label + force gauge + hidden traces |
| `class_mapping.json` | DB2 restimulus label -> 5-class intent mapping (with rationale) |
| `training_summary.txt` | final val accuracy, val force-MSE, hyperparams, distribution |
| `force_validation_e3.png` | overlay: real force vs EMG proxy vs model force on E3 |
| `force_validation_summary.txt` | E3 correlation report with honest interpretation |
| `README.md` | this file |

## Model architecture (recap)

- Input window: 200 ms (400 samples @ 2 kHz) of 70-dim multimodal features
  (12 EMG envelope + 36 acc + 22 glove).
- Backbone: 2-layer LSTM, hidden size 128, dropout 0.2 between layers.
- Shared 128-dim **hidden state** from the last LSTM timestep — exposed for
  Stage 7 (Diffusion Policy) via `model.get_hidden_state(x)`.
- Heads:
  - Classification: linear `128 -> 5` (REST / REACHING / GRIPPING / STABILIZING / RELEASING).
  - Force regression: MLP `128 -> 64 -> 1` with sigmoid -> scalar in [0, 1].

`forward(x)` returns a `ModelOutput` dataclass with `logits`, `probs`,
`force`, and `hidden`.

## How Stage 3 / Stage 7 should consume the checkpoint

```python
import torch
from src.stage2_lstm.model import IntentForceLSTM

ckpt = torch.load("outputs/stage2/lstm_best.pt", weights_only=False)
model = IntentForceLSTM()
model.load_state_dict(ckpt["model_state"])
model.eval().cuda()

# x shape (B, 400, 70), float32. Concatenate EMG (12) + acc (36) + glove (22).
hidden = model.get_hidden_state(x)   # (B, 128) — the conditioning feature
out = model(x)                       # logits, probs, force, hidden
```

The 128-dim `hidden` vector is what Stage 7's diffusion policy conditions on,
NOT just the discrete intent label. The intent label is for human-readable
display; the hidden state carries the continuous force/grasp variation that
two grasps with the same intent class produce differently.

## Force-intensity caveat — read this

The force regression head was trained to predict a **proxy** of grip force,
not a measured force:

> force_proxy[t] = moving_average( mean across 12 channels of MVC-normalized EMG envelope )

We use it because DB2's E1+E2 blocks contain no measured grip force; only the
excluded E3 block does. This proxy is documented as such everywhere and is
*not* a calibration of any sensor. The trained head therefore produces a
learned readout of EMG-based muscle effort.

The honesty check is in `force_validation_e3.png` and `force_validation_summary.txt`,
which compare both the proxy and the trained model's prediction against E3's
real 6-channel measured force.

## Run summary

- best val intent-accuracy : **{summary['best_val_acc']*100:.2f}%**
- best val force-MSE       : **{summary['best_val_mse']:.4f}**
- epochs run               : **{summary['epochs_run']}**
- best epoch               : **{summary['best_epoch']}**
"""
    (out_dir / "README.md").write_text(txt, encoding="utf-8")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", default=DEFAULT_STAGE1_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--e3-mat", default=DEFAULT_E3_MAT)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-force", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--animate", action="store_true",
                    help="open the matplotlib visualization window")
    ap.add_argument("--save-mp4", action="store_true",
                    help="render the visualization MP4 (default: on)")
    ap.add_argument("--no-save-mp4", action="store_true",
                    help="skip MP4 rendering")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse existing checkpoint and only run eval/viz/E3")
    ap.add_argument("--skip-e3", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = Path(args.stage1_dir)

    t0 = time.time()

    # 1. class mapping ---------------------------------------------------
    print("\n=== Stage 2 — class mapping ===")
    print(cm.format_table())
    write_class_mapping_json(out_dir)

    # 2. dataset --------------------------------------------------------
    print("\n=== Stage 2 — dataset ===")
    arrays = ds.load_stage1_arrays(stage1_dir)
    print(f"loaded Stage 1 arrays; T = {arrays.n_samples:,}, "
          f"force proxy [{arrays.force_proxy.min():.3f}, "
          f"{arrays.force_proxy.max():.3f}], mean={arrays.force_proxy.mean():.3f}")
    train_ds, val_ds, stats = ds.build_splits(arrays)
    print(stats.report())

    # 3. train ----------------------------------------------------------
    ckpt_path = out_dir / "lstm_best.pt"
    info = None
    history = None
    if args.skip_train and ckpt_path.exists():
        print("\n=== Stage 2 — skipping training (loading checkpoint) ===")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        model = IntentForceLSTM()
        model.load_state_dict(ckpt["model_state"])
        info = {
            "best_val_acc": ckpt["best_val_acc"],
            "best_val_mse": ckpt["best_val_mse"],
            "best_epoch":   ckpt["best_epoch"],
            "epochs_run":   ckpt.get("hparams", {}).get("max_epochs", -1),
            "confusion_matrix": None,
        }
    else:
        print("\n=== Stage 2 — training ===")
        cfg = tr.TrainConfig(
            max_epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, lambda_force=args.lambda_force, patience=args.patience,
        )
        model, history, info = tr.train(train_ds, val_ds, stats, out_dir, cfg)
        # confusion matrix png + summary
        if info["confusion_matrix"] is not None:
            tr.save_confusion_matrix(
                np.asarray(info["confusion_matrix"]),
                out_dir / "confusion_matrix.png",
            )
            print(f"[stage2] saved confusion matrix -> "
                  f"{out_dir/'confusion_matrix.png'}")
        tr.write_training_summary(out_dir / "training_summary.txt", cfg, stats, info)
        print(f"[stage2] wrote training_summary.txt")

    print(f"\n[stage2] best val intent-accuracy = {info['best_val_acc']*100:.2f}%")
    print(f"[stage2] best val force-MSE       = {info['best_val_mse']:.6f}")

    # 4. visualization MP4 ----------------------------------------------
    save_mp4 = not args.no_save_mp4
    if save_mp4 or args.animate:
        print("\n=== Stage 2 — visualization ===")
        mp4_path = out_dir / "intent_and_force_preview.mp4" if save_mp4 else None
        viz.render_visualization(
            arrays, model, val_reps=ds.VAL_REPS,
            out_mp4=mp4_path, duration_s=18.0,
            show=args.animate,
        )

    # 5. E3 validation --------------------------------------------------
    if not args.skip_e3:
        print("\n=== Stage 2 — E3 force validation ===")
        raw = v3.load_e3(args.e3_mat)
        report = raw.report()
        print(report)

        mvc = np.load(stage1_dir / "mvc_per_channel.npy")
        env = v3.process_e3_emg(raw.emg, mvc)
        accc = v3.process_e3_acc(raw.acc)
        proxy = ds.build_force_proxy(env)
        real_norm = v3.build_e3_force_scalar(raw.force)
        # smooth real force lightly for comparison vs smoothed proxy
        real_smooth = v3._smooth(real_norm, ds.FS_HZ, window_ms=200.0)

        # model predictions (sliding)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ends, model_force = v3.run_model_on_e3(model, env, accc, device)

        # build a sample-aligned model_force signal by step-interpolating
        T = env.shape[0]
        model_force_aligned = np.zeros(T, dtype=np.float32)
        last = 0.0
        idx = 0
        for s in range(T):
            while idx < ends.shape[0] and ends[idx] <= s:
                last = float(model_force[idx])
                idx += 1
            model_force_aligned[s] = last
        # smooth a touch
        model_force_aligned = v3._smooth(model_force_aligned, ds.FS_HZ, window_ms=100.0)

        vr = v3.correlate(proxy, model_force_aligned, real_smooth)
        print(f"[e3] n compared = {vr.n_samples_compared:,}")
        print(f"[e3] proxy vs real: r={vr.pearson_proxy_vs_force:+.3f} "
              f"rho={vr.spearman_proxy_vs_force:+.3f}")
        print(f"[e3] model vs real: r={vr.pearson_model_vs_force:+.3f} "
              f"rho={vr.spearman_model_vs_force:+.3f}")

        t_seconds = np.arange(T) / ds.FS_HZ
        v3.plot_overlay(
            t_seconds, real_smooth, proxy, model_force_aligned,
            out_dir / "force_validation_e3.png",
        )
        v3.write_summary(
            out_dir / "force_validation_summary.txt",
            raw_report=report, vr=vr,
            notes=[
                "E3 has no glove channel; the 22 glove inputs were zero-filled "
                "for model inference (a degraded-input validation).",
                "E3 EMG was processed with the *exact* Stage 1 DSP chain and "
                "normalized by the Stage 1 per-channel MVC vector.",
                "Real force is the L2 magnitude across the 6 force sensors, "
                "min-max normalised over the E3 block.",
                "Proxy and model traces were lightly smoothed (200 ms / 100 ms) "
                "before comparison; correlations are reported sample-by-sample.",
            ],
        )
        print(f"[stage2] wrote force_validation_e3.png + "
              f"force_validation_summary.txt")

    # 6. README ----------------------------------------------------------
    write_readme(out_dir, info)
    print(f"[stage2] wrote README.md")

    print(f"\n[stage2] DONE in {time.time()-t0:.1f}s")
    print(f"[stage2] final val intent-accuracy = {info['best_val_acc']*100:.2f}%")
    print(f"[stage2] final val force-MSE       = {info['best_val_mse']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

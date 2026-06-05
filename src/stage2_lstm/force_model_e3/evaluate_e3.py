"""E3 force-model evaluation: metrics + overlay plot + scatter plot.

All metrics are reported on the HELD-OUT test repetition (rep 6).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from .preprocess_e3 import E3Processed
from .dataset_e3 import E3SplitMeta


@dataclass
class E3Metrics:
    r2: float                     # coefficient of determination
    pearson_r: float              # Pearson correlation
    rmse_normalized: float        # RMSE in [0,1] units
    rmse_real: float              # RMSE in calibrated force units
    mae_normalized: float
    mae_real: float
    mape_real: float              # mean abs % error in real units (clamped denom)
    n_samples: int


def compute_metrics(y_true_norm: np.ndarray, y_pred_norm: np.ndarray,
                    meta: E3SplitMeta) -> E3Metrics:
    y_true_norm = y_true_norm.astype(np.float64)
    y_pred_norm = y_pred_norm.astype(np.float64)
    n = y_true_norm.shape[0]

    err = y_pred_norm - y_true_norm
    rmse_n = float(np.sqrt(np.mean(err * err)))
    mae_n = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true_norm - y_true_norm.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rho = float(pearsonr(y_true_norm, y_pred_norm).statistic)

    scale = meta.force_hi - meta.force_lo
    real_true = y_true_norm * scale + meta.force_lo
    real_pred = y_pred_norm * scale + meta.force_lo
    rmse_r = float(np.sqrt(np.mean((real_true - real_pred) ** 2)))
    mae_r = float(np.mean(np.abs(real_true - real_pred)))
    denom = np.maximum(np.abs(real_true), 0.05 * scale)  # avoid divide-by-zero
    mape = float(np.mean(np.abs(real_true - real_pred) / denom) * 100.0)

    return E3Metrics(
        r2=r2, pearson_r=rho,
        rmse_normalized=rmse_n, rmse_real=rmse_r,
        mae_normalized=mae_n, mae_real=mae_r,
        mape_real=mape,
        n_samples=n,
    )


# ---------------------------------------------------------------------------

def plot_overlay_real_units(ys_norm: np.ndarray, ps_norm: np.ndarray,
                            meta: E3SplitMeta,
                            path: Path,
                            sample_rate: float = 2000.0,
                            stride_samples: int = 20,
                            segment_s: float = 12.0) -> None:
    """Time-domain overlay in REAL force units. Picks an active segment with
    the highest variance in ground truth.
    """
    scale = meta.force_hi - meta.force_lo
    real_true = ys_norm.astype(np.float32) * scale + meta.force_lo
    real_pred = ps_norm.astype(np.float32) * scale + meta.force_lo

    # window samples per step in the held-out test stream (stride = 20 samples).
    step_s = stride_samples / sample_rate
    n = real_true.shape[0]
    if n == 0:
        return
    duration_total = n * step_s
    seg_steps = min(n, int(segment_s / step_s))
    # find the most active window: highest std of real_true
    best_start, best_std = 0, -1.0
    stride = max(1, seg_steps // 4)
    for s in range(0, max(1, n - seg_steps), stride):
        v = float(real_true[s:s + seg_steps].std())
        if v > best_std:
            best_std, best_start = v, s
    sl = slice(best_start, best_start + seg_steps)
    t = (np.arange(seg_steps) * step_s) + (best_start * step_s)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, real_true[sl], color="#f87171", lw=2.0,
            label="real measured force (E3, magnitude of 6 sensors)")
    ax.plot(t, real_pred[sl], color="#a3e635", lw=1.6, alpha=0.95,
            label="model-predicted force (held-out rep 6)")
    ax.set_xlabel("time (s, within E3 stream)")
    ax.set_ylabel("force (calibrated units, ~Newtons)")
    ax.set_title("E3 dedicated force model — held-out test segment")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_scatter(ys_norm: np.ndarray, ps_norm: np.ndarray,
                 meta: E3SplitMeta, metrics: E3Metrics, path: Path,
                 max_points: int = 20000) -> None:
    scale = meta.force_hi - meta.force_lo
    real_true = ys_norm.astype(np.float32) * scale + meta.force_lo
    real_pred = ps_norm.astype(np.float32) * scale + meta.force_lo
    if real_true.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(real_true.shape[0], size=max_points,
                                              replace=False)
        real_true = real_true[idx]; real_pred = real_pred[idx]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(real_true, real_pred, s=3, alpha=0.25, color="#00e5ff")
    lo = min(real_true.min(), real_pred.min())
    hi = max(real_true.max(), real_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#f87171", lw=1.5, label="y = x")
    ax.set_xlabel("real measured force (calibrated units)")
    ax.set_ylabel("model-predicted force (calibrated units)")
    ax.set_title(
        f"E3 force prediction — held-out test\n"
        f"R²={metrics.r2:+.3f}  r={metrics.pearson_r:+.3f}  "
        f"RMSE={metrics.rmse_real:.3f}"
    )
    ax.grid(alpha=0.15); ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary(path: Path, raw_report: str, meta: E3SplitMeta,
                  metrics_rich: E3Metrics,
                  metrics_envelope: E3Metrics | None = None,
                  hparams: dict | None = None) -> None:
    lines = []
    lines.append("Novonus Stage 2 — E3 dedicated force model accuracy")
    lines.append("=" * 60)
    lines.append("")
    lines.append(raw_report)
    lines.append("")
    lines.append(meta.report())
    lines.append("")

    def fmt(name: str, m: E3Metrics) -> list[str]:
        return [
            f"{name}:",
            f"  R^2                         : {m.r2:+.4f}",
            f"  Pearson r                   : {m.pearson_r:+.4f}",
            f"  RMSE (normalized [0,1])     : {m.rmse_normalized:.4f}",
            f"  RMSE (real force units)     : {m.rmse_real:.4f}",
            f"  MAE  (normalized)           : {m.mae_normalized:.4f}",
            f"  MAE  (real force units)     : {m.mae_real:.4f}",
            f"  MAPE (real units, %)        : {m.mape_real:.2f}",
            f"  test samples                : {m.n_samples:,}",
        ]

    if metrics_envelope is not None:
        lines.extend(fmt("Envelope-only baseline (no rich features)", metrics_envelope))
        lines.append("")
    lines.extend(fmt("Rich-feature model (RMS+MAV+WL+VAR+MNF concatenated)", metrics_rich))
    lines.append("")
    if metrics_envelope is not None:
        d_r2 = metrics_rich.r2 - metrics_envelope.r2
        d_rmse = metrics_rich.rmse_real - metrics_envelope.rmse_real
        lines.append("Feature-set comparison (rich minus envelope-only):")
        lines.append(f"  Delta R^2          : {d_r2:+.4f}")
        lines.append(f"  Delta RMSE (real)  : {d_rmse:+.4f}")
        lines.append("")

    lines.append("Honest interpretation:")
    lines.append(f"  - The model is trained against REAL measured force (E3 force sensors),")
    lines.append(f"    not the EMG-amplitude proxy used elsewhere in Stage 2.")
    lines.append(f"  - Held-out test = repetition 6 of E3 (model never saw it during training).")
    lines.append(f"  - E3 has no glove modality; the model uses EMG envelope + accelerometer only.")
    lines.append(f"  - E3 is sustained force-task data from one subject (DB2_s1). Real-world")
    lines.append(f"    contact-rich performance will differ; this number demonstrates the")
    lines.append(f"    EMG-to-real-force prediction capability, not field accuracy.")
    if hparams is not None:
        lines.append("")
        lines.append("Hyperparameters:")
        for k, v in hparams.items():
            lines.append(f"  {k}: {v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Independent honesty check: does our EMG-amplitude force proxy — and the
trained force head — correlate with the *measured* grip force recorded in
DB2 block E3 (which we excluded from training because it has no glove)?

We use E3 strictly as a held-out validation block. We:
  1. Load the raw E3 .mat and print its structure.
  2. Process E3's EMG through the *identical* Stage 1 DSP chain, normalising
     by the per-channel MVC saved from Stage 1 (so the activation lives in the
     same scale as training).
  3. Compute our EMG-amplitude force proxy on the E3 envelope.
  4. Run the trained model on E3 windows, zero-filling the 22 glove channels
     (this is a degraded-input inference, documented as such).
  5. Reduce E3's 6-channel measured force to a scalar (vector L2 magnitude,
     min-max normalized over the block).
  6. Time-align (the proxy is sample-aligned to EMG; model predictions live at
     window-end times) and compute Pearson + Spearman correlation, plus MAE.
  7. Save a 3-trace overlay plot on a representative E3 segment and an honest
     written summary.

The validation does NOT modify the trained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, spearmanr
import torch

from . import dataset as ds
from ..stage1_emg import dsp
from .class_mapping import INTENT_NAMES
from .model import IntentForceLSTM


FS_HZ: float = 2000.0


@dataclass
class E3Raw:
    emg: np.ndarray              # (T, 12)
    acc: np.ndarray              # (T, 36)
    force: np.ndarray            # (T, 6)
    forcecal: np.ndarray         # (2, 6)
    restimulus: np.ndarray       # (T,)
    rerepetition: np.ndarray     # (T,)
    n_samples: int

    def report(self) -> str:
        lines = []
        lines.append(f"E3 raw structure (S1_E3_A1.mat):")
        lines.append(f"  emg            shape={self.emg.shape}    dtype={self.emg.dtype}")
        lines.append(f"  acc            shape={self.acc.shape}    dtype={self.acc.dtype}")
        lines.append(f"  force          shape={self.force.shape}   dtype={self.force.dtype}")
        lines.append(f"  forcecal       shape={self.forcecal.shape} dtype={self.forcecal.dtype}")
        lines.append(f"  restimulus     shape={self.restimulus.shape}")
        lines.append(f"  rerepetition   shape={self.rerepetition.shape}")
        lines.append(f"  fs (DB2 spec): {FS_HZ:.0f} Hz")
        uniq_lbl = np.unique(self.restimulus)
        lines.append(f"  restimulus uniq: {uniq_lbl.tolist()}")
        lines.append(f"  force per-channel min/max:")
        for c in range(self.force.shape[1]):
            lines.append(
                f"    ch{c}: min={self.force[:,c].min():.3f}  "
                f"max={self.force[:,c].max():.3f}  "
                f"mean={self.force[:,c].mean():.3f}"
            )
        lines.append(f"  forcecal:\n    {self.forcecal}")
        return "\n".join(lines)


def load_e3(mat_path: str | Path) -> E3Raw:
    m = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    emg = m["emg"].astype(np.float32, copy=False)
    acc = m["acc"].astype(np.float32, copy=False)
    force = m["force"].astype(np.float32, copy=False)
    forcecal = m["forcecal"].astype(np.float32, copy=False)
    restim = m["restimulus"].astype(np.int16, copy=False).ravel()
    rerep = m["rerepetition"].astype(np.int16, copy=False).ravel()
    n = min(emg.shape[0], acc.shape[0], force.shape[0],
            restim.shape[0], rerep.shape[0])
    return E3Raw(
        emg=emg[:n], acc=acc[:n], force=force[:n],
        forcecal=forcecal,
        restimulus=restim[:n], rerepetition=rerep[:n], n_samples=n,
    )


# ---------------------------------------------------------------------------

def process_e3_emg(raw_emg: np.ndarray, mvc_from_stage1: np.ndarray,
                   fs: float = FS_HZ) -> np.ndarray:
    """Run the *identical* Stage 1 EMG chain (HP-Notch-BP-Rect-RMS-MVC) but use
    the MVC vector saved from Stage 1 (so E3 lives in the training scale)."""
    x = dsp.highpass(raw_emg, fs)
    x = dsp.notch(x, fs, 60.0)
    x = dsp.notch(x, fs, 120.0)
    x = dsp.notch(x, fs, 180.0)
    x = dsp.bandpass(x, fs)
    x = dsp.rectify(x)
    x = dsp.rms_envelope(x, fs, window_ms=100.0)
    safe = np.where(mvc_from_stage1 < 1e-12, np.float32(1.0), mvc_from_stage1)
    return (x / safe).astype(np.float32, copy=False)


def process_e3_acc(raw_acc: np.ndarray, fs: float = FS_HZ) -> np.ndarray:
    return dsp.lowpass(raw_acc, fs, cutoff=5.0, order=2)


def build_e3_force_scalar(force_6: np.ndarray) -> np.ndarray:
    """Reduce the 6-channel measured force to a scalar effort signal.

    We use the per-sample L2 magnitude across the 6 channels (a sensor-agnostic
    'overall force' summary), then min-max normalize across the E3 block so the
    scale matches our [0, 1] proxy and the model's [0, 1] prediction.
    """
    mag = np.linalg.norm(force_6.astype(np.float32), axis=1)
    lo = float(np.percentile(mag, 1.0))
    hi = float(np.percentile(mag, 99.0))
    rng = max(hi - lo, 1e-9)
    norm = np.clip((mag - lo) / rng, 0.0, 1.0)
    return norm.astype(np.float32)


# ---------------------------------------------------------------------------

@torch.no_grad()
def run_model_on_e3(model: IntentForceLSTM,
                    e3_envelope: np.ndarray,
                    e3_acc_clean: np.ndarray,
                    device: torch.device,
                    stride: int = ds.STRIDE,
                    batch_size: int = 256
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Slide the trained model over E3 with zero-filled glove inputs.

    Returns (window_end_sample_indices, predicted_force[N]).
    """
    T = e3_envelope.shape[0]
    starts = np.arange(0, T - ds.WINDOW_LEN + 1, stride, dtype=np.int64)
    ends = starts + ds.WINDOW_LEN

    zeros_glove_full = np.zeros((T, 22), dtype=np.float32)

    preds = np.empty(starts.shape[0], dtype=np.float32)
    model.to(device).eval()

    for i in range(0, starts.shape[0], batch_size):
        chunk = starts[i:i + batch_size]
        batch = np.empty((chunk.shape[0], ds.WINDOW_LEN, 70), dtype=np.float32)
        for j, s in enumerate(chunk):
            e = s + ds.WINDOW_LEN
            batch[j] = np.concatenate(
                (e3_envelope[s:e], e3_acc_clean[s:e], zeros_glove_full[s:e]),
                axis=1,
            )
        xb = torch.from_numpy(batch).to(device)
        out = model(xb)
        preds[i:i + chunk.shape[0]] = out.force.cpu().numpy()
    return ends - 1, preds


# ---------------------------------------------------------------------------

def _smooth(x: np.ndarray, fs: float, window_ms: float = 200.0) -> np.ndarray:
    w = max(1, int(round(window_ms / 1000.0 * fs)))
    k = np.ones(w, dtype=np.float32) / w
    return np.convolve(x.astype(np.float32, copy=False), k, mode="same"
                       ).astype(np.float32, copy=False)


def _normalize01(x: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    rng = max(hi - lo, 1e-9)
    return np.clip((x - lo) / rng, 0.0, 1.0).astype(np.float32)


@dataclass
class ValidationResult:
    pearson_proxy_vs_force: float
    spearman_proxy_vs_force: float
    mae_proxy_vs_force: float
    pearson_model_vs_force: float
    spearman_model_vs_force: float
    mae_model_vs_force: float
    n_samples_compared: int


def correlate(proxy: np.ndarray, model_force_at_t: np.ndarray,
              real_force: np.ndarray) -> ValidationResult:
    n = min(len(proxy), len(model_force_at_t), len(real_force))
    a = proxy[:n].astype(np.float64)
    b = model_force_at_t[:n].astype(np.float64)
    c = real_force[:n].astype(np.float64)
    pp = float(pearsonr(a, c).statistic)
    sp = float(spearmanr(a, c).statistic)
    pm = float(pearsonr(b, c).statistic)
    sm = float(spearmanr(b, c).statistic)
    return ValidationResult(
        pearson_proxy_vs_force=pp,
        spearman_proxy_vs_force=sp,
        mae_proxy_vs_force=float(np.mean(np.abs(a - c))),
        pearson_model_vs_force=pm,
        spearman_model_vs_force=sm,
        mae_model_vs_force=float(np.mean(np.abs(b - c))),
        n_samples_compared=n,
    )


# ---------------------------------------------------------------------------

def plot_overlay(t_seconds: np.ndarray,
                 real_force: np.ndarray, proxy: np.ndarray,
                 model_force: np.ndarray,
                 path: Path, segment_s: tuple[float, float] | None = None
                 ) -> None:
    if segment_s is None:
        # pick a chunk where real force varies most
        win = int(15.0 * FS_HZ)
        if real_force.shape[0] > win:
            step = int(0.5 * FS_HZ)
            best_s, best_v = 0, -1.0
            for s in range(0, real_force.shape[0] - win, step):
                v = float(np.std(real_force[s:s + win]))
                if v > best_v:
                    best_v, best_s = v, s
            t0 = t_seconds[best_s]
            t1 = t_seconds[best_s + win - 1]
            segment_s = (t0, t1)
        else:
            segment_s = (float(t_seconds[0]), float(t_seconds[-1]))

    sl = (t_seconds >= segment_s[0]) & (t_seconds <= segment_s[1])
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_seconds[sl], real_force[sl], color="#f87171", lw=1.8,
            label="real measured force (E3, L2 magnitude, normalized)")
    ax.plot(t_seconds[sl], proxy[sl], color="#00e5ff", lw=1.2, alpha=0.9,
            label="EMG-amplitude proxy (training target)")
    ax.plot(t_seconds[sl], model_force[sl], color="#a3e635", lw=1.6, alpha=0.95,
            label="model force-head prediction (glove zero-filled)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalised force / proxy in [0, 1]")
    ax.set_title("Stage 2 — E3 force validation: real measured force vs EMG proxy "
                 "vs model-predicted force")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary(path: Path, raw_report: str, vr: ValidationResult,
                  notes: Iterable[str] = ()) -> None:
    lines = []
    lines.append("Novonus Stage 2 — E3 force validation")
    lines.append("=" * 50)
    lines.append("")
    lines.append(raw_report)
    lines.append("")
    lines.append("Aligned-sample correlations (block E3, full record):")
    lines.append(f"  n samples compared:           {vr.n_samples_compared:,}")
    lines.append("")
    lines.append("  EMG-amplitude proxy  vs real force:")
    lines.append(f"    Pearson  r = {vr.pearson_proxy_vs_force:+.3f}")
    lines.append(f"    Spearman r = {vr.spearman_proxy_vs_force:+.3f}")
    lines.append(f"    MAE        = {vr.mae_proxy_vs_force:.3f}")
    lines.append("")
    lines.append("  Model force-head  vs real force:")
    lines.append(f"    Pearson  r = {vr.pearson_model_vs_force:+.3f}")
    lines.append(f"    Spearman r = {vr.spearman_model_vs_force:+.3f}")
    lines.append(f"    MAE        = {vr.mae_model_vs_force:.3f}")
    lines.append("")
    lines.append("Interpretation:")
    interp = _interpret(vr)
    for line in interp:
        lines.append(f"  {line}")
    if notes:
        lines.append("")
        lines.append("Caveats:")
        for n in notes:
            lines.append(f"  - {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _interpret(vr: ValidationResult) -> list[str]:
    def label(r: float) -> str:
        a = abs(r)
        if a < 0.1: return "no correlation"
        if a < 0.3: return "weak correlation"
        if a < 0.5: return "moderate correlation"
        if a < 0.7: return "strong correlation"
        return "very strong correlation"
    out = []
    out.append(
        f"EMG-amplitude proxy shows {label(vr.pearson_proxy_vs_force)} "
        f"(Pearson r={vr.pearson_proxy_vs_force:+.2f}) with measured force."
    )
    out.append(
        f"Model force head shows {label(vr.pearson_model_vs_force)} "
        f"(Pearson r={vr.pearson_model_vs_force:+.2f}) with measured force."
    )
    if vr.pearson_model_vs_force > 0.5:
        out.append(
            "The model's force-head output tracks real measured grip force "
            "with statistically meaningful agreement, despite the glove input "
            "being zero-filled on this block. This supports the claim that the "
            "shared LSTM hidden state encodes muscle-effort information."
        )
    elif vr.pearson_model_vs_force > 0.3:
        out.append(
            "The model's force-head output shows partial agreement with real "
            "force; useful as an effort cue but not a measurement substitute."
        )
    else:
        out.append(
            "The model's force-head output does NOT correlate strongly with "
            "real force on E3. This is honest: the head was trained on an "
            "EMG-amplitude proxy from a different task block (no glove on E3, "
            "different motor tasks), so weak transfer is plausible."
        )
    return out

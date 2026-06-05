"""Preprocess E3 to model-ready arrays.

EMG: same DSP chain as Stage 1 (HP-Notch-BP-Rect-RMS-MVCnorm).
     We compute E3's OWN per-channel MVC proxy (99th percentile of the
     envelope) rather than reusing Stage 1's MVC, because E3 is a sustained
     force-task block (high effort) and Stage 1's MVC was computed on the
     E1+E2 movement set where peak voluntary activations differ.

ACC: low-pass 5 Hz Butterworth-2 (same as Stage 1).

Filtered (no rectify) EMG: also returned for rich frequency-domain features
that need the post-bandpass, pre-rectified signal.

Force target: per-sample L2 magnitude across the 6 force channels, then
min-max normalized to [0, 1] using 1st-/99th-percentile robust bounds.
The scaling factor (force_lo, force_hi) is saved so predictions can be
converted back to the original calibrated force units."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...stage1_emg import dsp
from .data_e3 import E3Block, FS_HZ


@dataclass
class E3Processed:
    emg_filtered: np.ndarray   # (T, 12) post HP+notch+BP, pre-rectify (for features)
    emg_envelope: np.ndarray   # (T, 12) MVC-normalized envelope (LSTM input stream)
    acc_clean: np.ndarray      # (T, 36) low-pass cleaned accelerometer
    force_scalar: np.ndarray   # (T,)    L2 magnitude across 6 force channels
    force_target: np.ndarray   # (T,)    force_scalar normalized into [0, 1]
    rerepetition: np.ndarray   # (T,)    rep id
    restimulus: np.ndarray     # (T,)    movement label
    mvc_e3: np.ndarray         # (12,)   per-channel MVC proxy used
    force_lo: float            # normalization low (1st pct of magnitude)
    force_hi: float            # normalization high (99th pct of magnitude)
    fs: float = FS_HZ

    @property
    def n_samples(self) -> int:
        return int(self.emg_envelope.shape[0])

    def unscale(self, normalized: np.ndarray) -> np.ndarray:
        """Convert [0,1]-normalized predictions back to calibrated force units."""
        return (np.asarray(normalized) * (self.force_hi - self.force_lo)
                + self.force_lo).astype(np.float32)


def _filtered_emg(raw_emg: np.ndarray, fs: float) -> np.ndarray:
    x = dsp.highpass(raw_emg, fs)
    x = dsp.notch(x, fs, 60.0)
    x = dsp.notch(x, fs, 120.0)
    x = dsp.notch(x, fs, 180.0)
    x = dsp.bandpass(x, fs)
    return x.astype(np.float32, copy=False)


def _envelope_from_filtered(filtered: np.ndarray, fs: float
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Continue the chain: rectify, RMS, MVC-normalize on E3's own MVC."""
    x = dsp.rectify(filtered)
    x = dsp.rms_envelope(x, fs, window_ms=100.0)
    mvc = np.percentile(x, 99.0, axis=0).astype(np.float32)
    safe = np.where(mvc < 1e-12, np.float32(1.0), mvc)
    env = (x / safe).astype(np.float32, copy=False)
    return env, mvc


def build_force_target(force_6: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """L2 magnitude across 6 force channels -> normalize to [0, 1]."""
    mag = np.linalg.norm(force_6.astype(np.float32), axis=1)
    lo = float(np.percentile(mag, 1.0))
    hi = float(np.percentile(mag, 99.0))
    rng = max(hi - lo, 1e-9)
    target = np.clip((mag - lo) / rng, 0.0, 1.0).astype(np.float32)
    return mag.astype(np.float32), target, lo, hi


def preprocess(block: E3Block, fs: float = FS_HZ, verbose: bool = True
               ) -> E3Processed:
    if verbose:
        print(f"[e3-prep] HP+notch+BP filtering EMG ({block.emg.shape})")
    filt = _filtered_emg(block.emg, fs)
    if verbose:
        print(f"[e3-prep] rectify + RMS 100ms + MVC normalize (E3 own MVC)")
    env, mvc = _envelope_from_filtered(filt, fs)
    if verbose:
        print(f"[e3-prep] LP 5 Hz accel ({block.acc.shape})")
    acc = dsp.lowpass(block.acc, fs, cutoff=5.0, order=2)
    if verbose:
        print(f"[e3-prep] force magnitude + [0,1] normalize (1-99 pct)")
    mag, tgt, lo, hi = build_force_target(block.force)
    if verbose:
        print(f"[e3-prep] force scale: lo={lo:.3f}  hi={hi:.3f}  (calibrated units)")

    return E3Processed(
        emg_filtered=filt,
        emg_envelope=env,
        acc_clean=acc.astype(np.float32, copy=False),
        force_scalar=mag,
        force_target=tgt,
        rerepetition=block.rerepetition,
        restimulus=block.restimulus,
        mvc_e3=mvc,
        force_lo=lo, force_hi=hi,
    )

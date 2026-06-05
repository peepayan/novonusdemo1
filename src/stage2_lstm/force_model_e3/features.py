"""Rich per-window EMG feature extraction.

For each window and each of the 12 EMG channels we compute:
  RMS  : sqrt(mean(x**2))                                    -- effort proxy
  MAV  : mean(|x|)                                           -- effort proxy
  WL   : sum(|x[t+1] - x[t]|)  waveform length               -- complexity
  VAR  : var(x)                                              -- effort proxy
  MNF  : mean spectral frequency from |X(f)|^2               -- fatigue / drive

5 features x 12 channels = 60-dim feature vector per window.
Computed in chunks to bound memory."""

from __future__ import annotations

import numpy as np


N_FEATURES_PER_CH: int = 5
FEATURE_NAMES: tuple[str, ...] = ("RMS", "MAV", "WL", "VAR", "MNF")


def _chunked_windows(filtered_emg: np.ndarray, starts: np.ndarray,
                     window_len: int, chunk: int = 4096):
    """Yield (start_subset, (n, window_len, 12)) chunks."""
    n = starts.shape[0]
    for i in range(0, n, chunk):
        sub = starts[i:i + chunk]
        # build (k, window_len, 12) using fancy indexing
        # offsets[None, :] + sub[:, None] -> (k, window_len)
        offsets = np.arange(window_len, dtype=np.int64)
        idx = sub[:, None] + offsets[None, :]
        chunk_x = filtered_emg[idx]   # (k, window_len, 12)
        yield i, sub, chunk_x


def extract_window_features(filtered_emg: np.ndarray,
                            starts: np.ndarray,
                            window_len: int,
                            fs: float,
                            chunk: int = 4096) -> np.ndarray:
    """Return (N, n_ch * 5) per-window features in fixed order:
       [RMS_ch0..ch11, MAV_ch0..ch11, WL_ch0..ch11, VAR_ch0..ch11, MNF_ch0..ch11]."""
    n = starts.shape[0]
    n_ch = filtered_emg.shape[1]
    out = np.empty((n, n_ch * N_FEATURES_PER_CH), dtype=np.float32)

    # rfft freq axis is fixed
    freqs = np.fft.rfftfreq(window_len, d=1.0 / fs).astype(np.float32)

    for i, sub, x in _chunked_windows(filtered_emg, starts, window_len, chunk):
        # x: (k, T, C)
        # RMS
        rms = np.sqrt(np.mean(x * x, axis=1, dtype=np.float32))
        # MAV
        mav = np.mean(np.abs(x), axis=1, dtype=np.float32)
        # WL
        wl = np.sum(np.abs(np.diff(x, axis=1)), axis=1, dtype=np.float32)
        # VAR
        var = np.var(x, axis=1, dtype=np.float32)
        # MNF via rfft
        X = np.fft.rfft(x, axis=1)                  # (k, F, C) complex
        power = (X.real ** 2 + X.imag ** 2).astype(np.float32)
        denom = power.sum(axis=1) + 1e-12           # (k, C)
        num = (power * freqs[None, :, None]).sum(axis=1)
        mnf = (num / denom).astype(np.float32)

        k = sub.shape[0]
        out[i:i + k, 0 * n_ch:1 * n_ch] = rms
        out[i:i + k, 1 * n_ch:2 * n_ch] = mav
        out[i:i + k, 2 * n_ch:3 * n_ch] = wl
        out[i:i + k, 3 * n_ch:4 * n_ch] = var
        out[i:i + k, 4 * n_ch:5 * n_ch] = mnf
    return out


def standardize(features: np.ndarray, train_mask: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score features using train-set mean/std only (no leakage)."""
    mu = features[train_mask].mean(axis=0).astype(np.float32)
    sd = features[train_mask].std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-9, np.float32(1.0), sd)
    z = ((features - mu) / sd).astype(np.float32)
    return z, mu, sd

"""EMG DSP chain + light cleanup for acc/glove.

EMG pipeline (in order, zero-phase via filtfilt where applicable):
  1. Butterworth high-pass, 4th order, 20 Hz
  2. Notch 60 / 120 / 180 Hz (mains + harmonics)
  3. Butterworth band-pass, 4th order, 20-450 Hz (capped to <Nyquist if fs forces it)
  4. Full-wave rectification |x|
  5. 100 ms RMS envelope (window = round(100ms * fs))
  6. Per-channel MVC normalization (99th-percentile proxy; no MVC trial in DB2)

Each stage is an independently testable function with axis=0 = time.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

# -------- shared utilities --------

def _filtfilt_per_channel(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """filtfilt channel-by-channel to keep peak memory ~ one column."""
    out = np.empty_like(x, dtype=np.float32)
    if x.ndim == 1:
        return signal.filtfilt(b, a, x).astype(np.float32, copy=False)
    for c in range(x.shape[1]):
        out[:, c] = signal.filtfilt(b, a, x[:, c]).astype(np.float32, copy=False)
    return out


# -------- EMG stages --------

def highpass(x: np.ndarray, fs: float, cutoff: float = 20.0, order: int = 4) -> np.ndarray:
    b, a = signal.butter(order, cutoff / (fs / 2.0), btype="high")
    return _filtfilt_per_channel(b, a, x)


def notch(x: np.ndarray, fs: float, freq: float, Q: float = 30.0) -> np.ndarray:
    b, a = signal.iirnotch(freq / (fs / 2.0), Q)
    return _filtfilt_per_channel(b, a, x)


def bandpass(x: np.ndarray, fs: float,
             low: float = 20.0, high: float = 450.0, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    max_safe = nyq * 0.98
    if high >= nyq:
        new_high = max_safe
        warnings.warn(
            f"bandpass upper cutoff {high} Hz >= Nyquist {nyq} Hz; "
            f"capping to {new_high} Hz",
            stacklevel=2,
        )
        high = new_high
    if not (0.0 < low < high < nyq):
        raise ValueError(f"bad band [{low},{high}] for fs={fs}")
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return _filtfilt_per_channel(b, a, x)


def rectify(x: np.ndarray) -> np.ndarray:
    return np.abs(x).astype(np.float32, copy=False)


def rms_envelope(x: np.ndarray, fs: float, window_ms: float = 100.0) -> np.ndarray:
    w = max(1, int(round((window_ms / 1000.0) * fs)))
    x2 = (x.astype(np.float32, copy=False)) ** 2
    mean_x2 = uniform_filter1d(x2, size=w, axis=0, mode="nearest")
    return np.sqrt(mean_x2).astype(np.float32, copy=False)


def mvc_normalize(x: np.ndarray, percentile: float = 99.0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each channel by its high-percentile envelope as an MVC proxy."""
    mvc = np.percentile(x, percentile, axis=0).astype(np.float32)
    safe = np.where(mvc < 1e-12, np.float32(1.0), mvc)
    return (x / safe).astype(np.float32, copy=False), mvc


def process_emg(raw_emg: np.ndarray, fs: float, verbose: bool = True
                ) -> tuple[np.ndarray, np.ndarray]:
    """Full EMG chain. Returns (clean_envelope (T,C), mvc_per_ch (C,))."""
    def _say(s):
        if verbose:
            print(f"  [emg] {s}")

    _say("1/6 high-pass 20 Hz Butter-4")
    x = highpass(raw_emg, fs)
    _say("2/6 notch 60 Hz")
    x = notch(x, fs, 60.0)
    _say("    notch 120 Hz")
    x = notch(x, fs, 120.0)
    _say("    notch 180 Hz")
    x = notch(x, fs, 180.0)
    _say("3/6 band-pass 20-450 Hz Butter-4")
    x = bandpass(x, fs)
    _say("4/6 rectify |x|")
    x = rectify(x)
    _say("5/6 RMS envelope 100 ms")
    x = rms_envelope(x, fs)
    _say("6/6 MVC normalization (99th pct per ch)")
    x, mvc = mvc_normalize(x)
    return x, mvc


# -------- light processing for the other modalities --------

def lowpass(x: np.ndarray, fs: float, cutoff: float = 5.0, order: int = 2) -> np.ndarray:
    b, a = signal.butter(order, cutoff / (fs / 2.0), btype="low")
    return _filtfilt_per_channel(b, a, x)


def process_acc(acc: np.ndarray, fs: float, cutoff: float = 5.0,
                verbose: bool = True) -> np.ndarray:
    if verbose:
        print(f"  [acc] low-pass {cutoff} Hz Butter-2 -> jitter removed")
    return lowpass(acc, fs, cutoff=cutoff, order=2)


def process_glove(glove: np.ndarray, fs: float, cutoff: float = 5.0,
                  verbose: bool = True) -> np.ndarray:
    """Smooth and min-max normalize per joint to ~[0,1] using robust percentiles."""
    if verbose:
        print(f"  [glove] low-pass {cutoff} Hz Butter-2 + per-channel robust 1-99% normalize")
    smoothed = lowpass(glove, fs, cutoff=cutoff, order=2)
    p1 = np.percentile(smoothed, 1.0, axis=0).astype(np.float32)
    p99 = np.percentile(smoothed, 99.0, axis=0).astype(np.float32)
    rng = np.where((p99 - p1) < 1e-9, np.float32(1.0), p99 - p1)
    out = (smoothed - p1) / rng
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

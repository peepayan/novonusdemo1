"""Stage 2 dataset: sliding windows over the Stage 1 multimodal arrays.

What each window is
-------------------
- Inputs : 200 ms ( = 400 samples @ 2000 Hz) of stacked EMG (12) + acc (36)
           + glove (22) -> shape (400, 70) float32.
- Intent : majority 5-class intent within the window (after restimulus -> intent
           mapping). REST dominates because the recording is mostly rest, so
           class-weighted cross-entropy is used in training.
- Force  : a scalar in [0, 1] read off a smoothed EMG-amplitude proxy signal.

Force-intensity proxy
---------------------
DB2 blocks E1+E2 contain no measured force. We use the mean of the 12-channel
MVC-normalized EMG envelope at each timestep, lightly smoothed by a 200 ms
moving average. This is an *EMG-amplitude force proxy*, not a calibrated force
measurement, and must always be described as such downstream. The per-window
target is the mean of the proxy across the window (documented choice; mean is
more stable than end-value and matches the "average effort during the window").

Split
-----
Repetition-based, per Ninapro protocol:
  - rep 0 (the tiny rest gap, ~1.7 s total) is dropped.
  - reps 1..5 -> train.
  - rep    6 -> val.
A window's fold is determined by the rep of its centre sample. Windows that
cross a rep boundary are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .class_mapping import (
    DB2_LABEL_TO_INTENT,
    INTENT_NAMES,
    N_INTENT_CLASSES,
    build_label_array_mapping,
)

FS_HZ: float = 2000.0
WINDOW_MS: float = 200.0
STRIDE_MS: float = 10.0
WINDOW_LEN: int = int(round(WINDOW_MS / 1000.0 * FS_HZ))   # 400
STRIDE: int = int(round(STRIDE_MS / 1000.0 * FS_HZ))       # 20
FORCE_SMOOTH_MS: float = 200.0
N_FEATURES: int = 12 + 36 + 22                              # 70

TRAIN_REPS: tuple[int, ...] = (1, 2, 3, 4, 5)
VAL_REPS: tuple[int, ...] = (6,)


# ---------------------------------------------------------------------------
# Loading + proxy construction
# ---------------------------------------------------------------------------

@dataclass
class Stage1Arrays:
    emg: np.ndarray         # (T, 12) float32
    acc: np.ndarray         # (T, 36) float32
    glove: np.ndarray       # (T, 22) float32
    labels: np.ndarray      # (T,)    int16  -- DB2 restimulus
    reps: np.ndarray        # (T,)    int16  -- DB2 rerepetition
    intent: np.ndarray      # (T,)    int8   -- mapped 5-class intent
    force_proxy: np.ndarray # (T,)    float32, smoothed mean EMG envelope

    @property
    def n_samples(self) -> int:
        return int(self.emg.shape[0])


def _smooth_uniform(x: np.ndarray, window: int) -> np.ndarray:
    """Causal-ish uniform moving average via convolution, same length out."""
    if window <= 1:
        return x.astype(np.float32, copy=False)
    k = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x.astype(np.float32, copy=False), k, mode="same").astype(
        np.float32, copy=False
    )


def build_force_proxy(emg_envelope: np.ndarray, fs: float = FS_HZ,
                      smooth_ms: float = FORCE_SMOOTH_MS) -> np.ndarray:
    """Scalar EMG-amplitude proxy: per-timestep mean across 12 envelope channels,
    smoothed by a `smooth_ms` moving average, clipped to [0, 1].

    NOTE: this is *not* a measured force. It is an EMG-amplitude proxy that
    correlates with overall muscle effort and is documented as such everywhere.
    """
    if emg_envelope.ndim != 2 or emg_envelope.shape[1] != 12:
        raise ValueError(f"expected (T, 12) EMG envelope, got {emg_envelope.shape}")
    mean_act = emg_envelope.mean(axis=1).astype(np.float32, copy=False)
    win = max(1, int(round(smooth_ms / 1000.0 * fs)))
    sm = _smooth_uniform(mean_act, win)
    return np.clip(sm, 0.0, 1.0).astype(np.float32, copy=False)


def load_stage1_arrays(stage1_dir: str | Path) -> Stage1Arrays:
    d = Path(stage1_dir)
    emg = np.load(d / "emg_envelope.npy").astype(np.float32, copy=False)
    acc = np.load(d / "accel_clean.npy").astype(np.float32, copy=False)
    glove = np.load(d / "glove_clean.npy").astype(np.float32, copy=False)
    labels = np.load(d / "labels.npy").astype(np.int16, copy=False)
    reps = np.load(d / "repetitions.npy").astype(np.int16, copy=False)

    T = emg.shape[0]
    for name, a in (("acc", acc), ("glove", glove),
                    ("labels", labels), ("reps", reps)):
        if a.shape[0] != T:
            raise ValueError(f"length mismatch: emg has {T}, {name} has {a.shape[0]}")

    # vectorized label -> intent
    table = np.asarray(build_label_array_mapping(int(labels.max())), dtype=np.int8)
    intent = table[labels.clip(min=0)]

    proxy = build_force_proxy(emg)

    return Stage1Arrays(
        emg=emg, acc=acc, glove=glove,
        labels=labels, reps=reps, intent=intent, force_proxy=proxy,
    )


# ---------------------------------------------------------------------------
# Sliding-window indexer (vectorized)
# ---------------------------------------------------------------------------

def _candidate_starts(T: int) -> np.ndarray:
    last_start = T - WINDOW_LEN
    if last_start < 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, last_start + 1, STRIDE, dtype=np.int64)


def _window_summary(starts: np.ndarray, reps: np.ndarray, intent: np.ndarray,
                    proxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each candidate start, return (window_rep, window_intent, window_force)
    using the END-1 sample's rep, the mode of intents in the window, and the
    *mean* of the force proxy over the window.

    The 'mode of intents' is computed with a fast bin-count over the window
    slice for each candidate, in chunks to keep memory bounded.
    """
    n = starts.shape[0]
    win_rep = np.empty(n, dtype=np.int16)
    win_intent = np.empty(n, dtype=np.int8)
    win_force = np.empty(n, dtype=np.float32)

    # Same-rep check: a window is valid only if rep is constant from start to end-1.
    # We'll mark windows that cross a rep boundary so the caller can drop them.
    same_rep = np.empty(n, dtype=bool)

    # cumulative sums for force proxy (mean per window in O(1))
    cs_proxy = np.concatenate(([0.0], np.cumsum(proxy, dtype=np.float64)))

    # per-class cumulative counts of intent, shape (T+1, K)
    K = N_INTENT_CLASSES
    one_hot = np.zeros((proxy.shape[0], K), dtype=np.int32)
    one_hot[np.arange(proxy.shape[0]), intent.astype(np.int64)] = 1
    cs_oh = np.concatenate(
        (np.zeros((1, K), dtype=np.int64), np.cumsum(one_hot, axis=0, dtype=np.int64)),
        axis=0,
    )

    end_idx = starts + WINDOW_LEN
    win_rep[:] = reps[end_idx - 1]
    same_rep[:] = (reps[starts] == reps[end_idx - 1])

    # mean force proxy per window
    win_force[:] = ((cs_proxy[end_idx] - cs_proxy[starts]) / WINDOW_LEN).astype(np.float32)

    # majority intent per window
    counts = cs_oh[end_idx] - cs_oh[starts]          # (n, K)
    win_intent[:] = counts.argmax(axis=1).astype(np.int8)

    return win_rep, win_intent, win_force, same_rep


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------

class WindowedNinaproDataset(Dataset):
    """Slice 200 ms windows from the Stage 1 arrays on the fly.

    Indices into the underlying arrays are precomputed; per-window features are
    materialised lazily by slicing the original arrays, so peak memory stays
    near the size of the (T, 12/36/22) arrays themselves rather than a fully
    expanded (N_windows, 400, 70) blob.
    """

    def __init__(self, arrays: Stage1Arrays, starts: np.ndarray,
                 intent: np.ndarray, force: np.ndarray):
        self.emg = arrays.emg
        self.acc = arrays.acc
        self.glove = arrays.glove
        self.starts = starts.astype(np.int64, copy=False)
        self.intent = intent.astype(np.int64, copy=False)
        self.force = force.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.starts.shape[0]

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        e = s + WINDOW_LEN
        # Concatenate per-modality slices along channel axis -> (400, 70)
        x = np.concatenate(
            (self.emg[s:e], self.acc[s:e], self.glove[s:e]),
            axis=1,
        ).astype(np.float32, copy=False)
        return (
            torch.from_numpy(x),
            torch.tensor(int(self.intent[idx]), dtype=torch.long),
            torch.tensor(float(self.force[idx]), dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Builder + splits
# ---------------------------------------------------------------------------

@dataclass
class SplitStats:
    n_train: int
    n_val: int
    train_class_counts: np.ndarray   # (K,)
    val_class_counts: np.ndarray     # (K,)
    train_class_weights: np.ndarray  # (K,) for nn.CrossEntropyLoss

    def report(self) -> str:
        lines = []
        lines.append(f"windows: train={self.n_train:,}  val={self.n_val:,}")
        lines.append("intent-class distribution (train | val):")
        for k, name in enumerate(INTENT_NAMES):
            t = int(self.train_class_counts[k])
            v = int(self.val_class_counts[k])
            tp = 100 * t / max(self.n_train, 1)
            vp = 100 * v / max(self.n_val, 1)
            lines.append(
                f"  {k} {name:<11s}  train={t:>8,d} ({tp:5.1f}%)   "
                f"val={v:>8,d} ({vp:5.1f}%)"
            )
        lines.append(
            "class weights (train, normalized to mean 1): "
            + ", ".join(f"{w:.3f}" for w in self.train_class_weights)
        )
        return "\n".join(lines)


def _compute_class_weights(counts: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights normalised to mean 1; 0-count classes get 0."""
    safe = counts.astype(np.float64).copy()
    nz = safe > 0
    inv = np.zeros_like(safe)
    inv[nz] = 1.0 / safe[nz]
    if inv[nz].sum() > 0:
        inv[nz] = inv[nz] / inv[nz].mean()
    return inv.astype(np.float32)


def build_splits(arrays: Stage1Arrays
                 ) -> tuple[WindowedNinaproDataset, WindowedNinaproDataset, SplitStats]:
    starts = _candidate_starts(arrays.n_samples)
    win_rep, win_intent, win_force, same_rep = _window_summary(
        starts, arrays.reps, arrays.intent, arrays.force_proxy,
    )

    # keep only windows that lie inside a single rep
    keep = same_rep
    starts, win_rep, win_intent, win_force = (
        starts[keep], win_rep[keep], win_intent[keep], win_force[keep],
    )

    train_mask = np.isin(win_rep, np.asarray(TRAIN_REPS, dtype=np.int16))
    val_mask = np.isin(win_rep, np.asarray(VAL_REPS, dtype=np.int16))

    train_ds = WindowedNinaproDataset(
        arrays, starts[train_mask], win_intent[train_mask], win_force[train_mask],
    )
    val_ds = WindowedNinaproDataset(
        arrays, starts[val_mask], win_intent[val_mask], win_force[val_mask],
    )

    train_counts = np.bincount(win_intent[train_mask].astype(np.int64),
                               minlength=N_INTENT_CLASSES)
    val_counts = np.bincount(win_intent[val_mask].astype(np.int64),
                             minlength=N_INTENT_CLASSES)
    weights = _compute_class_weights(train_counts)

    stats = SplitStats(
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        train_class_counts=train_counts,
        val_class_counts=val_counts,
        train_class_weights=weights,
    )
    return train_ds, val_ds, stats


def make_loaders(train_ds: WindowedNinaproDataset,
                 val_ds: WindowedNinaproDataset,
                 batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    # num_workers=0 on Windows (spawn-based multiprocessing breaks with arrays
    # referenced by Dataset). On Linux this can safely be raised.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader

"""E3 sliding-window dataset for the dedicated force regressor.

Input per window (LSTM stream): (400, 48) = 12 EMG envelope + 36 accel.
Optional per-window rich-feature vector (60 dim) is fetched alongside, so
the model can compare envelope-only vs envelope+features.

Target per window: real measured force, at the window end. Documented as
end-aligned (matches an instantaneous read-out at the window boundary).

Split: standard Ninapro repetition-based held-out protocol.
  reps 1..5 -> TRAIN.
  rep    6  -> TEST  (held out; the model never sees these reps).
  rep 0     -> dropped (tiny gap).
A window's rep is the rep of its end sample; windows crossing a rep
boundary are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .features import extract_window_features, standardize
from .preprocess_e3 import E3Processed

FS_HZ: float = 2000.0
WINDOW_MS: float = 200.0
STRIDE_MS: float = 10.0
WINDOW_LEN: int = int(round(WINDOW_MS / 1000.0 * FS_HZ))   # 400
STRIDE: int = int(round(STRIDE_MS / 1000.0 * FS_HZ))       # 20

N_LSTM_FEATURES: int = 12 + 36                              # EMG env + acc = 48

TRAIN_REPS: tuple[int, ...] = (1, 2, 3, 4, 5)
TEST_REPS: tuple[int, ...] = (6,)


@dataclass
class E3SplitMeta:
    n_train: int
    n_test: int
    feature_dim: int               # 0 if rich features disabled
    feat_mean: np.ndarray | None
    feat_std: np.ndarray | None
    force_lo: float
    force_hi: float

    def report(self) -> str:
        lines = []
        lines.append(f"E3 split sizes : train={self.n_train:,}   test={self.n_test:,}")
        lines.append(f"rich features  : {self.feature_dim} dim "
                     + ("(enabled)" if self.feature_dim > 0 else "(disabled)"))
        lines.append(f"force scaling  : lo={self.force_lo:.3f}  hi={self.force_hi:.3f}  "
                     "(calibrated units)")
        return "\n".join(lines)


class E3ForceWindows(Dataset):
    def __init__(self,
                 emg_env: np.ndarray,        # (T, 12)
                 acc_clean: np.ndarray,      # (T, 36)
                 force_target: np.ndarray,   # (T,) in [0,1]
                 starts: np.ndarray,         # (N,) int64
                 features: np.ndarray | None # (N, F) or None
                 ):
        self.emg_env = emg_env
        self.acc_clean = acc_clean
        self.force_target = force_target
        self.starts = starts.astype(np.int64, copy=False)
        self.features = features

    def __len__(self) -> int:
        return self.starts.shape[0]

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        e = s + WINDOW_LEN
        x = np.concatenate(
            (self.emg_env[s:e], self.acc_clean[s:e]),
            axis=1,
        ).astype(np.float32, copy=False)
        y = float(self.force_target[e - 1])
        feat = (torch.from_numpy(self.features[idx])
                if self.features is not None else torch.empty(0))
        return (
            torch.from_numpy(x),
            feat,
            torch.tensor(y, dtype=torch.float32),
        )


def _candidate_starts(T: int) -> np.ndarray:
    last = T - WINDOW_LEN
    if last < 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, last + 1, STRIDE, dtype=np.int64)


def build_splits(proc: E3Processed, use_rich_features: bool = True
                 ) -> tuple[E3ForceWindows, E3ForceWindows, E3SplitMeta]:
    starts = _candidate_starts(proc.n_samples)
    end_idx = starts + WINDOW_LEN
    rep_at_end = proc.rerepetition[end_idx - 1]
    rep_at_start = proc.rerepetition[starts]
    same_rep = rep_at_start == rep_at_end

    keep = same_rep
    starts = starts[keep]
    rep_at_end = rep_at_end[keep]

    train_arr = np.asarray(TRAIN_REPS, dtype=np.int16)
    test_arr = np.asarray(TEST_REPS, dtype=np.int16)
    train_mask = np.isin(rep_at_end, train_arr)
    test_mask = np.isin(rep_at_end, test_arr)

    # rich features (one vector per window) — computed on the FILTERED EMG
    feat_dim = 0
    feat_all = None
    feat_mu = feat_sd = None
    if use_rich_features:
        print(f"[e3-ds] extracting rich features over {len(starts):,} windows ...")
        feat_all = extract_window_features(
            proc.emg_filtered, starts, WINDOW_LEN, proc.fs,
        )
        feat_all, feat_mu, feat_sd = standardize(feat_all, train_mask)
        feat_dim = feat_all.shape[1]
        print(f"[e3-ds] feature shape = {feat_all.shape}")

    train_ds = E3ForceWindows(
        proc.emg_envelope, proc.acc_clean, proc.force_target,
        starts[train_mask],
        feat_all[train_mask] if feat_all is not None else None,
    )
    test_ds = E3ForceWindows(
        proc.emg_envelope, proc.acc_clean, proc.force_target,
        starts[test_mask],
        feat_all[test_mask] if feat_all is not None else None,
    )

    meta = E3SplitMeta(
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        feature_dim=feat_dim,
        feat_mean=feat_mu, feat_std=feat_sd,
        force_lo=proc.force_lo, force_hi=proc.force_hi,
    )
    return train_ds, test_ds, meta


def make_loaders(train_ds: E3ForceWindows, test_ds: E3ForceWindows,
                 batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    # num_workers=0 on Windows.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    return train_loader, test_loader

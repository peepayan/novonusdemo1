"""Sliding-window dataset over the cached features.

A training sample is one (sample_id, t) pair where t is a timestep
inside that sample's trajectory. The observation is the cached
DINOv2/LSTM/state values at t; the action target is the next 16
timesteps of action commands (padded at the trajectory tail).

All heavy lifting (DINOv2, LSTM) was done by ``cache_features.py`` —
this dataset just memory-maps the cached .npy files, so __getitem__ is
O(action_horizon) and the dataloader runs at GPU speed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..stage3_sim import physics_constants as pc


# Conditioning dimensions (frozen contract; obs_encoder + policy depend
# on these exact sizes).
DINO_DIM: int = 768
LSTM_DIM: int = 128
ROBOT_STATE_DIM: int = 20    # joint_pos(6)+joint_vel(6)+ee_pos(3)+ee_quat(4)+grip(1)
ROBOT_EMBED_DIM: int = 64
ACTION_DIM: int = 7          # 6 joint velocities + gripper
ACTION_HORIZON: int = 16
EXEC_HORIZON: int = 8


# ---------------------------------------------------------------------------
# Per-sample memmap loader
# ---------------------------------------------------------------------------

@dataclass
class CachedSample:
    sample_id: str
    stage: str                       # "stage4" | "stage5"
    dino: np.ndarray                 # (T, 768)
    lstm: np.ndarray                 # (T, 128)
    state: np.ndarray                # (T, 20)
    action: np.ndarray               # (T, 7)
    baseline_demo_idx: int | None    # only set for stage5


def load_cached_samples(index_path: Path, cache_dir: Path
                        ) -> list[CachedSample]:
    idx = json.loads(Path(index_path).read_text(encoding="utf-8"))
    out: list[CachedSample] = []
    for entry in idx["entries"]:
        sid = entry["id"]
        d = Path(cache_dir)
        try:
            dino = np.load(d / f"{sid}_dino.npy", mmap_mode="r")
            lstm = np.load(d / f"{sid}_lstm.npy", mmap_mode="r")
            state = np.load(d / f"{sid}_state.npy", mmap_mode="r")
            action = np.load(d / f"{sid}_action.npy", mmap_mode="r")
        except FileNotFoundError:
            continue
        out.append(CachedSample(
            sample_id=sid, stage=entry["stage"],
            dino=dino, lstm=lstm, state=state, action=action,
            baseline_demo_idx=entry.get("baseline_demo_idx"),
        ))
    return out


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------

class DemoActionDataset(Dataset):
    """Each item: observation at timestep t and action sequence [t, t+H].

    The observation pairs (dino, lstm, state) all live in the same
    timestep. The action target is the next ``action_horizon`` action
    commands; tail timesteps are clamped so we still get full-length
    sequences (zero-velocity hold at the trajectory end).
    """

    def __init__(self, samples: list[CachedSample],
                 action_horizon: int = ACTION_HORIZON,
                 include_emg: bool = True):
        self.samples = samples
        self.H = action_horizon
        self.include_emg = include_emg
        # build flat (sample_idx, t) index
        self.indices: list[tuple[int, int]] = []
        for si, s in enumerate(samples):
            T = int(s.action.shape[0])
            # allow every t in [0, T-1]; tail actions get clamped/padded
            for t in range(T):
                self.indices.append((si, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        si, t = self.indices[k]
        s = self.samples[si]
        T = int(s.action.shape[0])
        # action window [t, t+H], clamped at tail (repeat last action)
        end = min(t + self.H, T)
        a = np.zeros((self.H, ACTION_DIM), dtype=np.float32)
        a[:end - t] = np.asarray(s.action[t:end])
        if end - t < self.H:
            a[end - t:] = a[end - t - 1:end - t]    # hold last
        item = {
            "dino": torch.from_numpy(np.asarray(s.dino[t], dtype=np.float32)),
            "state": torch.from_numpy(np.asarray(s.state[t], dtype=np.float32)),
            "action": torch.from_numpy(a),
        }
        if self.include_emg:
            item["lstm"] = torch.from_numpy(
                np.asarray(s.lstm[t], dtype=np.float32))
        return item


# ---------------------------------------------------------------------------
# Action normaliser (essential for diffusion training stability)
# ---------------------------------------------------------------------------

@dataclass
class ActionNormalizer:
    mean: np.ndarray     # (ACTION_DIM,)
    std: np.ndarray      # (ACTION_DIM,)

    @classmethod
    def fit(cls, samples: list[CachedSample]) -> "ActionNormalizer":
        all_a = np.concatenate([np.asarray(s.action) for s in samples],
                               axis=0)
        m = all_a.mean(axis=0).astype(np.float32)
        s = all_a.std(axis=0).astype(np.float32)
        s = np.maximum(s, 1e-6)
        return cls(mean=m, std=s)

    def to(self, device: str) -> dict[str, torch.Tensor]:
        return {
            "mean": torch.from_numpy(self.mean).to(device),
            "std": torch.from_numpy(self.std).to(device),
        }


# ---------------------------------------------------------------------------
# Train/val split — deterministic, with at least some Stage 4 in val
# ---------------------------------------------------------------------------

def split_train_val(samples: list[CachedSample], seed: int = 0,
                    val_size: int = 20,
                    min_val_stage4: int = 6,
                    ) -> tuple[list[CachedSample], list[CachedSample]]:
    rng = np.random.default_rng(seed)
    stage4 = [s for s in samples if s.stage == "stage4"]
    stage5 = [s for s in samples if s.stage == "stage5"]
    rng.shuffle(stage4); rng.shuffle(stage5)
    val_s4 = stage4[:min_val_stage4]
    rem_val = val_size - len(val_s4)
    val_s5 = stage5[:rem_val] if rem_val > 0 else []
    val = val_s4 + val_s5
    train = [s for s in samples if s not in val]
    return train, val


__all__ = [
    "DINO_DIM", "LSTM_DIM", "ROBOT_STATE_DIM", "ROBOT_EMBED_DIM",
    "ACTION_DIM", "ACTION_HORIZON", "EXEC_HORIZON",
    "CachedSample", "load_cached_samples",
    "DemoActionDataset", "ActionNormalizer", "split_train_val",
]

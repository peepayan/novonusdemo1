"""Ninapro DB2 per-subject loader.

- Hard-pins EMG sampling rate to 2000 Hz (no `frequency` field in the .mat;
  this is the published DB2 spec; acc and glove are pre-upsampled to match).
- Loads exercise blocks E1 + E2 by default (E3 has no glove, breaks alignment).
- Returns numpy arrays with documented shapes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

EMG_FS: float = 2000.0  # Hz, from DB2 spec
EXPECTED_CHANNELS = {"emg": 12, "acc": 36, "glove": 22}


@dataclass
class SubjectData:
    subject: str
    fs: float
    emg: np.ndarray         # (T, 12) float32
    acc: np.ndarray         # (T, 36) float32
    glove: np.ndarray       # (T, 22) float32
    labels: np.ndarray      # (T,)    int16  -- restimulus
    reps: np.ndarray        # (T,)    int16  -- rerepetition
    block_ids: np.ndarray   # (T,)    int8   -- 1 for E1 samples, 2 for E2

    @property
    def n_samples(self) -> int:
        return int(self.emg.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs


def list_subjects(root: str | Path) -> list[str]:
    return sorted(p.name for p in Path(root).glob("DB2_s*") if p.is_dir())


def _find_block_mat(subj_dir: Path, subj_num: int, block: str) -> Path:
    pat = re.compile(rf"^S{subj_num}_{block}_.*\.mat$", re.IGNORECASE)
    cands = [p for p in subj_dir.glob("*.mat") if pat.match(p.name)]
    if not cands:
        raise FileNotFoundError(f"no .mat for {subj_dir.name} block {block}")
    return cands[0]


def _load_block(path: Path) -> dict[str, np.ndarray]:
    mat = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    # crop to common length (handles E3-style off-by-one if it ever appears)
    needed = ["emg", "acc", "glove", "restimulus", "rerepetition"]
    missing = [k for k in needed if k not in mat]
    if missing:
        raise KeyError(f"{path.name}: missing keys {missing}")
    n = min(mat[k].shape[0] for k in needed)
    return {
        "emg":   mat["emg"][:n].astype(np.float32, copy=False),
        "acc":   mat["acc"][:n].astype(np.float32, copy=False),
        "glove": mat["glove"][:n].astype(np.float32, copy=False),
        "labels": mat["restimulus"][:n].astype(np.int16, copy=False).ravel(),
        "reps":   mat["rerepetition"][:n].astype(np.int16, copy=False).ravel(),
    }


def load_subject(root: str | Path, subject: str,
                 blocks: tuple[str, ...] = ("E1", "E2")) -> SubjectData:
    subj_dir = Path(root) / subject
    if not subj_dir.is_dir():
        raise FileNotFoundError(subj_dir)
    m = re.match(r"DB2_s(\d+)", subject, re.IGNORECASE)
    if not m:
        raise ValueError(f"can't parse subject id from {subject!r}")
    subj_num = int(m.group(1))

    parts: list[dict[str, np.ndarray]] = []
    block_arrays: list[np.ndarray] = []
    for b in blocks:
        d = _load_block(_find_block_mat(subj_dir, subj_num, b))
        # validate channel counts
        for key, expected in EXPECTED_CHANNELS.items():
            got = d[key].shape[1]
            if got != expected:
                raise ValueError(f"{subject} {b}: expected {expected} {key} ch, got {got}")
        parts.append(d)
        bid = int(b[1:])  # E1 -> 1
        block_arrays.append(np.full(d["emg"].shape[0], bid, dtype=np.int8))

    cat = lambda key: np.concatenate([p[key] for p in parts], axis=0)
    return SubjectData(
        subject=subject,
        fs=EMG_FS,
        emg=cat("emg"),
        acc=cat("acc"),
        glove=cat("glove"),
        labels=cat("labels"),
        reps=cat("reps"),
        block_ids=np.concatenate(block_arrays),
    )


def summarize(d: SubjectData) -> str:
    uniq_lbl = np.unique(d.labels)
    uniq_reps = np.unique(d.reps)
    n_movements = int((uniq_lbl > 0).sum())
    return "\n".join([
        f"subject:           {d.subject}",
        f"sampling rate:     {d.fs:.0f} Hz (hard-pinned, DB2 spec)",
        f"total samples:     {d.n_samples:,}",
        f"duration:          {d.duration_s/60:.2f} min ({d.duration_s:.1f} s)",
        f"emg shape:         {d.emg.shape}  dtype={d.emg.dtype}",
        f"acc shape:         {d.acc.shape}  dtype={d.acc.dtype}",
        f"glove shape:       {d.glove.shape}  dtype={d.glove.dtype}",
        f"labels (restim):   {d.labels.shape}  classes={n_movements} movements + rest",
        f"labels range:      [{int(uniq_lbl.min())} .. {int(uniq_lbl.max())}]",
        f"repetitions:       unique={list(uniq_reps)} (0=rest gap)",
        f"blocks present:    {sorted(set(int(x) for x in np.unique(d.block_ids)))}",
    ])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\deepa\novonusdemo1\data\ninapro_db2")
    ap.add_argument("--subject", default="DB2_s1")
    args = ap.parse_args()

    print(f"subjects available in {args.root}:")
    for s in list_subjects(args.root):
        print(f"  - {s}")
    print()
    d = load_subject(args.root, args.subject)
    print(summarize(d))

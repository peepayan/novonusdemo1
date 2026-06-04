"""Inspect one Ninapro DB2 .mat file: print every key, shape, dtype.

Used to confirm sampling rate, channel counts, label set, and repetition encoding
BEFORE writing any DSP code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


def _mat_keys(mat: dict) -> list[str]:
    return [k for k in mat.keys() if not k.startswith("__")]


def _unique_summary(arr: np.ndarray, label: str, max_show: int = 80) -> str:
    flat = np.asarray(arr).ravel()
    uniq = np.unique(flat)
    head = ", ".join(str(x) for x in uniq[:max_show])
    more = "" if len(uniq) <= max_show else f", ... ({len(uniq)} total)"
    return f"{label}: n_unique={len(uniq)} -> [{head}{more}]"


def inspect_file(path: Path) -> dict:
    print(f"\n--- {path.name} ---")
    print(f"path: {path}")
    print(f"size: {path.stat().st_size / 1024**2:.1f} MB")

    mat = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    keys = _mat_keys(mat)
    print(f"keys ({len(keys)}): {keys}")

    summary: dict = {"path": str(path), "keys": keys, "arrays": {}}
    for k in keys:
        v = mat[k]
        if isinstance(v, np.ndarray):
            print(f"  {k:>16s}  shape={str(v.shape):<18s} dtype={v.dtype}")
            summary["arrays"][k] = {"shape": v.shape, "dtype": str(v.dtype)}
        else:
            print(f"  {k:>16s}  type={type(v).__name__}")
            summary["arrays"][k] = {"type": type(v).__name__}

    # Reported sampling rate if present
    if "frequency" in mat:
        summary["frequency_field"] = float(np.asarray(mat["frequency"]).ravel()[0])
        print(f"\n[freq field present] frequency = {summary['frequency_field']} Hz")
    elif "fs" in mat:
        summary["frequency_field"] = float(np.asarray(mat["fs"]).ravel()[0])
        print(f"\n[fs field present] fs = {summary['frequency_field']} Hz")

    # Movement labels
    for label_key in ("restimulus", "stimulus"):
        if label_key in mat:
            print()
            print(_unique_summary(mat[label_key], label_key))

    # Repetitions
    for rep_key in ("rerepetition", "repetition"):
        if rep_key in mat:
            print(_unique_summary(mat[rep_key], rep_key))

    # Infer fs from EMG length if frequency field missing
    if "emg" in mat and "restimulus" in mat:
        n_emg = mat["emg"].shape[0]
        n_lab = np.asarray(mat["restimulus"]).shape[0]
        print(f"\n[shape sanity] emg.shape[0]={n_emg}  restimulus.shape[0]={n_lab}  "
              f"aligned={n_emg == n_lab}")
        summary["n_samples_emg"] = int(n_emg)
        summary["n_samples_labels"] = int(n_lab)
        if "frequency" not in mat and "fs" not in mat:
            print("[warn] no explicit sampling-rate field; downstream must infer fs externally")

    # Per-modality sample-count comparison (DB2 sometimes samples acc/glove at lower rate)
    for k in ("emg", "acc", "glove", "inclin"):
        if k in mat:
            print(f"  rows[{k}]={mat[k].shape[0]}")

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=r"C:\Users\deepa\novonusdemo1\data\ninapro_db2",
        help="Root of extracted DB2 subject folders",
    )
    ap.add_argument("--subject", default="DB2_s1")
    args = ap.parse_args()

    subj_dir = Path(args.root) / args.subject
    if not subj_dir.exists():
        print(f"ERROR: subject dir not found: {subj_dir}", file=sys.stderr)
        return 1

    mats = sorted(subj_dir.glob("*.mat"))
    print(f"Found {len(mats)} .mat file(s) under {subj_dir}")
    for p in mats:
        print(f"  - {p.name}")

    if not mats:
        return 1

    # Inspect each block (E1/E2/E3) -- they differ in label content
    for p in mats:
        inspect_file(p)

    print("\n[inspect_data] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stage 1 entry point: copy-confirm data, load, DSP, light-process, save, viz."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import data_loader as dl
from . import dsp
from . import visualize as viz

DEFAULT_DATA_ROOT = r"C:\Users\deepa\novonusdemo1\data\ninapro_db2"
DEFAULT_OUT_DIR = r"C:\Users\deepa\novonusdemo1\outputs\stage1"


def _confirm_data(root: Path, subject: str) -> None:
    subj_dir = root / subject
    if not subj_dir.is_dir():
        sys.exit(f"ERROR: missing subject dir {subj_dir}")
    mats = sorted(subj_dir.glob("*.mat"))
    print(f"[data] root = {root}")
    print(f"[data] subject dir = {subj_dir}")
    print(f"[data] found {len(mats)} .mat file(s):")
    for p in mats:
        size = p.stat().st_size / 1024**2
        print(f"         {p.name}  ({size:.1f} MB)")


def _save_outputs(out_dir: Path,
                  subject: str,
                  fs: float,
                  envelope: np.ndarray,
                  accel_clean: np.ndarray,
                  glove_clean: np.ndarray,
                  labels: np.ndarray,
                  reps: np.ndarray,
                  block_ids: np.ndarray,
                  mvc: np.ndarray,
                  summary_text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "emg_envelope.npy": envelope,
        "accel_clean.npy": accel_clean,
        "glove_clean.npy": glove_clean,
        "labels.npy": labels.astype(np.int16, copy=False),
        "repetitions.npy": reps.astype(np.int16, copy=False),
        "block_ids.npy": block_ids.astype(np.int8, copy=False),
        "mvc_per_channel.npy": mvc.astype(np.float32, copy=False),
    }
    print(f"[save] writing to {out_dir}")
    for name, arr in paths.items():
        p = out_dir / name
        np.save(p, arr)
        print(f"  - {name:22s} shape={str(arr.shape):<16s} dtype={arr.dtype}  "
              f"{p.stat().st_size/1024**2:.1f} MB")

    meta = {
        "subject": subject,
        "fs_hz": float(fs),
        "channels": {"emg": 12, "acc": 36, "glove": 22},
        "label_field": "restimulus",
        "repetition_field": "rerepetition",
        "blocks_included": ["E1", "E2"],
        "files": {
            "emg_envelope.npy":   "float32 (T,12), HP-Notch-BP-Rect-RMS100ms-MVCnorm",
            "accel_clean.npy":    "float32 (T,36), LP 5 Hz Butter-2",
            "glove_clean.npy":    "float32 (T,22), LP 5 Hz Butter-2 + per-ch 1-99% normalize",
            "labels.npy":         "int16 (T,), restimulus 0=rest, 1..49=movement",
            "repetitions.npy":    "int16 (T,), rerepetition 0=gap, 1..6=rep idx",
            "block_ids.npy":      "int8 (T,), 1=E1, 2=E2",
            "mvc_per_channel.npy":"float32 (12,), 99th-pct MVC proxy used to normalize",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "summary.txt").write_text(summary_text + "\n")

    readme = f"""# Stage 1 outputs — subject {subject}

All arrays are time-aligned at fs = {fs:.0f} Hz with T = {envelope.shape[0]:,} samples
(blocks E1 + E2 concatenated). E3 omitted because it lacks the glove modality.

| file                 | shape            | dtype   | description |
|----------------------|------------------|---------|-------------|
| emg_envelope.npy     | (T, 12)          | float32 | clean MVC-normalized EMG envelope |
| accel_clean.npy      | (T, 36)          | float32 | LP-5Hz accelerometer |
| glove_clean.npy      | (T, 22)          | float32 | smoothed + per-channel normalized glove |
| labels.npy           | (T,)             | int16   | restimulus 0=rest, 1..49=movement |
| repetitions.npy      | (T,)             | int16   | rerepetition 0=gap, 1..6=rep idx |
| block_ids.npy        | (T,)             | int8    | 1=E1, 2=E2 |
| mvc_per_channel.npy  | (12,)            | float32 | 99th-pct MVC proxy used as denominator |
| meta.json            | -                | -       | sampling rate + channel layout |
| segment_raw_vs_clean.png | -            | -       | 10 s raw vs envelope, demo figure |

Stage 2 (LSTM intent classifier) should load `emg_envelope.npy` + `labels.npy`
+ optionally `accel_clean.npy` / `glove_clean.npy` and chunk into windows.
"""
    (out_dir / "README.md").write_text(readme)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--subject", default="DB2_s1")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--animate", action="store_true",
                    help="launch live oscilloscope window (interactive)")
    ap.add_argument("--save-mp4", default=None,
                    help="path to save oscilloscope as MP4 (needs ffmpeg)")
    ap.add_argument("--anim-duration", type=float, default=20.0,
                    help="seconds of recording to animate")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)

    t0 = time.time()
    _confirm_data(root, args.subject)

    print(f"\n[load] loading {args.subject} blocks E1+E2 ...")
    d = dl.load_subject(root, args.subject, blocks=("E1", "E2"))
    summary = dl.summarize(d)
    print(summary)
    print(f"[load] done in {time.time()-t0:.1f}s\n")

    t1 = time.time()
    print("[dsp] EMG chain ...")
    envelope, mvc = dsp.process_emg(d.emg, d.fs)
    print(f"[dsp] EMG envelope shape={envelope.shape} dtype={envelope.dtype}  "
          f"in {time.time()-t1:.1f}s")

    t2 = time.time()
    print("\n[dsp] light cleanup ...")
    accel_clean = dsp.process_acc(d.acc, d.fs)
    glove_clean = dsp.process_glove(d.glove, d.fs)
    print(f"[dsp] acc+glove done in {time.time()-t2:.1f}s\n")

    _save_outputs(out_dir, d.subject, d.fs,
                  envelope, accel_clean, glove_clean,
                  d.labels, d.reps, d.block_ids, mvc, summary)

    if not args.no_png:
        png_path = out_dir / "segment_raw_vs_clean.png"
        print(f"\n[png] saving static raw-vs-clean segment to {png_path}")
        viz.make_static_segment_png(d.emg, envelope, d.fs, png_path,
                                    channels=(0, 6), window_s=10.0)
        print(f"[png] wrote {png_path.stat().st_size/1024:.1f} KB")

    if args.save_mp4:
        print(f"\n[mp4] rendering oscilloscope video -> {args.save_mp4}")
        viz.run_live_oscilloscope(d.emg, envelope, accel_clean, d.fs,
                                  duration_s=args.anim_duration,
                                  save_mp4=args.save_mp4, show=False)

    if args.animate:
        print("\n[viz] launching live oscilloscope (close window to exit) ...")
        viz.run_live_oscilloscope(d.emg, envelope, accel_clean, d.fs,
                                  duration_s=args.anim_duration, show=True)

    print(f"\n[stage1] DONE in {time.time()-t0:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Aggregate statistics + visual summaries over a saved demo dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from ..stage3_sim import physics_constants as pc


def iter_demo_npz(demos_dir: Path) -> Iterable[Path]:
    for p in sorted(Path(demos_dir).glob("demo_*.npz")):
        yield p


def load_one(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def write_dataset_summary(demos_dir: Path, out_txt: Path) -> dict:
    rows = []
    grape_initials = []
    force_max = []
    force_mean = []
    durations = []
    phase_durations = {p: [] for p in (
        "REST_PRE", "REACHING", "GRIPPING", "STABILIZING", "RELEASING",
        "REST_POST")}
    grape_finals = []
    grape_errs = []
    contact_max = []

    for p in iter_demo_npz(demos_dir):
        z = load_one(p)
        meta = json.loads(str(z["metadata"]))
        T = int(z["joint_positions"].shape[0])
        durations.append(meta["duration_s"])
        for phase, cnt in meta["phase_frame_count"].items():
            phase_durations[phase].append(cnt / meta["render_fps"])
        af = z["applied_force"].astype(np.float32)
        force_max.append(float(af.max()))
        force_mean.append(float(af.mean()))
        grape_initials.append(z["grape_initial_pos"].astype(np.float32))
        grape_finals.append(np.asarray(meta["grape_final_xyz"],
                                       dtype=np.float32))
        grape_errs.append(float(meta["grape_xy_err_m"]))
        contact_max.append(float(meta["max_contact_force_N"]))
        rows.append((p.name, T, meta["duration_s"], force_max[-1],
                     meta["max_contact_force_N"], meta["grape_xy_err_m"]))

    n = len(rows)
    total_T = sum(r[1] for r in rows)
    lines = []
    lines.append("Novonus Stage 4 — demo dataset summary")
    lines.append("=" * 55)
    lines.append(f"demos saved              : {n}")
    lines.append(f"total timesteps          : {total_T:,}")
    if n:
        lines.append(f"avg duration             : {np.mean(durations):.2f} s")
        lines.append(f"min/max duration         : "
                     f"{np.min(durations):.2f} / {np.max(durations):.2f} s")
        lines.append("")
        lines.append("per-phase duration (s) [mean ± std, min..max]:")
        for phase, vals in phase_durations.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float32)
            lines.append(f"  {phase:<12s}  {arr.mean():.2f} ± {arr.std():.2f}"
                         f"   ({arr.min():.2f}..{arr.max():.2f})")
        lines.append("")
        fm = np.asarray(force_max)
        fme = np.asarray(force_mean)
        cm = np.asarray(contact_max)
        lines.append("applied grip force (N)  : "
                     f"max {fm.max():.2f}  mean {fme.mean():.2f}  "
                     f"min {fm.min():.2f}")
        lines.append("measured contact (N)    : "
                     f"max {cm.max():.2f}  median {np.median(cm):.2f}")
        lines.append(f"crush threshold (N)     : "
                     f"{pc.GRAPE_CRUSH_THRESHOLD_N:.1f}")
        lines.append("ALL DEMOS BELOW CRUSH    : "
                     f"{bool((cm < pc.GRAPE_CRUSH_THRESHOLD_N).all())}")
        lines.append(f"safe grip band          : "
                     f"{pc.GRAPE_SAFE_GRIP_MIN_N:.1f}-"
                     f"{pc.GRAPE_SAFE_GRIP_MAX_N:.1f} N")
        lines.append("")
        gi = np.stack(grape_initials, axis=0)
        gf = np.stack(grape_finals, axis=0)
        ge = np.asarray(grape_errs)
        lines.append("grape initial XY (m):")
        lines.append(f"  x: mean {gi[:,0].mean():.3f}  std {gi[:,0].std():.4f}"
                     f"   range {gi[:,0].min():.3f}..{gi[:,0].max():.3f}")
        lines.append(f"  y: mean {gi[:,1].mean():.3f}  std {gi[:,1].std():.4f}"
                     f"   range {gi[:,1].min():.3f}..{gi[:,1].max():.3f}")
        lines.append("grape final XY error from target (mm):")
        lines.append(f"  mean {ge.mean()*1000:.1f}   max {ge.max()*1000:.1f}"
                     f"   threshold 30.0")
        lines.append(f"  all in target zone (<= 30 mm) : "
                     f"{bool((ge <= 0.030).all())}")
        lines.append("")
        lines.append("EMG pairing               : synthetic, phase-aligned")
        lines.append("  source                  : outputs/stage1/*.npy "
                     "(val reps), Stage 2 LSTM-predicted class per window")
        lines.append("  GRIPPING force gate     : LSTM force head 0.2-0.4")
        lines.append("")
        lines.append("per-demo (filename, T, dur, max_applied_N, "
                     "max_contact_N, xy_err_mm):")
        for nm, T, du, fm_, cm_, ge_ in rows:
            lines.append(f"  {nm:<18s} T={T:>4d}  dur={du:5.2f}s  "
                         f"applied={fm_:4.2f}  contact={cm_:4.2f}  "
                         f"err={ge_*1000:5.1f}mm")
    out_txt = Path(out_txt); out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_demos": n,
        "total_timesteps": total_T,
        "max_applied_N": float(np.asarray(force_max).max()) if force_max else 0,
        "max_contact_N": float(np.asarray(contact_max).max()) if contact_max else 0,
        "all_below_crush": bool(
            (np.asarray(contact_max) < pc.GRAPE_CRUSH_THRESHOLD_N).all())
            if contact_max else True,
        "all_in_target_zone": bool(
            (np.asarray(grape_errs) <= 0.030).all()) if grape_errs else True,
        "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
    }


# ---------------------------------------------------------------------------

def make_demo_grid(demos_dir: Path, out_png: Path) -> Path:
    import imageio.v2 as iio
    paths = list(iter_demo_npz(demos_dir))
    n = len(paths)
    rows, cols = 5, 6
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.4),
                             facecolor="#0d111a")
    for i in range(rows * cols):
        ax = axs.flat[i]; ax.axis("off")
        if i >= n:
            continue
        z = np.load(paths[i], allow_pickle=False)
        frames_subdir = str(z["frames"])
        demo_dir = paths[i].parent / frames_subdir
        png_files = sorted(demo_dir.glob("*.png"))
        if not png_files:
            continue
        # pick a frame in the middle of STABILIZING when the arm is mid-lift
        meta = json.loads(str(z["metadata"]))
        bounds = meta.get("phase_frame_boundaries", {})
        sb = bounds.get("STABILIZING", [0, len(png_files)])
        mid = int(0.5 * (sb[0] + sb[1]))
        mid = max(0, min(mid, len(png_files) - 1))
        img = iio.imread(png_files[mid])
        ax.imshow(img)
        ax.set_title(paths[i].stem, color="#cccccc", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png

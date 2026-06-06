"""Stage 5 dataset artefacts: augmentation grid PNG, stats JSON,
summary TXT, README.md, and the unified dataset_index.json that Stage 7
will consume."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..stage3_sim import physics_constants as pc
from . import params as pmod
from . import verification as vmod


# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------

def load_scenario(npz_path: Path) -> dict[str, Any]:
    z = np.load(npz_path, allow_pickle=False)
    out: dict[str, Any] = {k: z[k] for k in z.files}
    out["augmentation_params"] = json.loads(str(out["augmentation_params"]))
    out["verification_results"] = json.loads(
        str(out["verification_results"]))
    out["metadata"] = json.loads(str(out["metadata"]))
    out["frames_mp4"] = str(out["frames_mp4"])
    return out


# ---------------------------------------------------------------------------
# Augmentation grid PNG (5x6 = 30 cells)
# ---------------------------------------------------------------------------

GRID_ROWS: int = 5
GRID_COLS: int = 6


def _mp4_middle_frame(mp4_path: Path) -> np.ndarray:
    """Pick a frame near the middle of the gripping/transit window — the
    most visually informative moment in the demo."""
    reader = iio.get_reader(str(mp4_path))
    try:
        # quick path: count frames, return ~midpoint
        try:
            n = reader.count_frames()
        except Exception:
            n = 412   # Stage 4 default — close enough
        target = max(0, n * 9 // 16)   # late-gripping / early-transit
        frame = reader.get_data(target)
    finally:
        reader.close()
    return np.asarray(frame)


def build_augmentation_grid(scen_dir: Path, frames_dir: Path,
                            out_png: Path,
                            seed: int = 0) -> Path:
    """One representative frame per baseline-demo origin, arranged 5x6.

    Sampling rule: prefer one scenario per *distinct* baseline_demo_idx.
    If we have fewer baselines covered than cells, fill remaining cells
    with random scenarios.
    """
    scen_paths = sorted(Path(scen_dir).glob("scenario_*.npz"))
    if not scen_paths:
        raise RuntimeError(f"no scenarios in {scen_dir}")

    rng = np.random.default_rng(seed)
    # group by baseline_demo_idx
    by_base: dict[int, list[Path]] = {}
    for p in scen_paths:
        z = np.load(p, allow_pickle=False)
        b = int(z["baseline_demo_idx"])
        by_base.setdefault(b, []).append(p)

    bases = sorted(by_base.keys())
    rng.shuffle(bases)
    picks: list[Path] = []
    for b in bases:
        picks.append(rng.choice(np.asarray(by_base[b])))
        if len(picks) >= GRID_ROWS * GRID_COLS:
            break
    # pad with random extra scenarios from any base if we have fewer
    # distinct bases than cells
    if len(picks) < GRID_ROWS * GRID_COLS:
        leftover = [p for p in scen_paths if p not in picks]
        rng.shuffle(np.asarray(leftover))
        picks.extend(leftover[:GRID_ROWS * GRID_COLS - len(picks)])

    plt.style.use("dark_background")
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS,
                             figsize=(GRID_COLS * 2.4, GRID_ROWS * 2.0),
                             facecolor="#0d111a")
    fig.suptitle(
        f"Novonus — Stage 5: physics-validated augmentations "
        f"({len(scen_paths)} passing scenarios)",
        color="#ffffff", fontsize=14, y=0.995)

    for ax, npz_path in zip(axes.ravel(), picks):
        s = load_scenario(npz_path)
        mp4 = Path(frames_dir) / s["frames_mp4"]
        try:
            frame = _mp4_middle_frame(mp4)
            ax.imshow(frame)
        except Exception as e:
            ax.text(0.5, 0.5, f"no frame\n({type(e).__name__})",
                    ha="center", va="center", color="#888",
                    transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        label = pmod.short_label(pmod.AugmentationParams(
            grape_xy_offset_m=tuple(s["augmentation_params"][
                "grape_xy_offset_m"]),
            grape_yaw_rad=s["augmentation_params"]["grape_yaw_rad"],
            grape_mass_kg=s["augmentation_params"]["grape_mass_kg"],
            grape_solref0=s["augmentation_params"]["grape_solref0"],
            grape_jaw_friction=s["augmentation_params"][
                "grape_jaw_friction"],
            approach_angle_rad=s["augmentation_params"]["approach_angle_rad"],
            light_ambient_scale=s["augmentation_params"][
                "light_ambient_scale"],
            light_dir_jitter_rad=tuple(s["augmentation_params"][
                "light_dir_jitter_rad"]),
            table_friction0=s["augmentation_params"]["table_friction0"],
        ))
        ax.set_title(f"base {int(s['baseline_demo_idx']):02d}\n{label}",
                     color="#cccccc", fontsize=8, pad=4)

    # blank any unused cells
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, facecolor="#0d111a")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Stats + summary
# ---------------------------------------------------------------------------

def compute_stats(scen_dir: Path, attempts_log_path: Path,
                  n_baselines: int) -> dict[str, Any]:
    scen_paths = sorted(Path(scen_dir).glob("scenario_*.npz"))
    passing: list[dict[str, Any]] = []
    for p in scen_paths:
        passing.append(load_scenario(p))

    log: list[dict[str, Any]] = []
    if Path(attempts_log_path).exists():
        try:
            log = json.loads(Path(attempts_log_path).read_text(
                encoding="utf-8"))
        except Exception:
            log = []

    raw = len(log)
    n_pass = len(passing)
    rejection_counter: Counter = Counter()
    for entry in log:
        if not entry.get("passed", False):
            for r in entry.get("reasons", []):
                rejection_counter[r] += 1

    # force stats over passing scenarios
    peak_force = np.array([float(np.asarray(s["contact_force"]).max())
                           for s in passing], dtype=np.float64)
    mean_force = np.array([float(np.asarray(s["contact_force"]).mean())
                           for s in passing], dtype=np.float64)
    max_overshoot = float(peak_force.max()) if peak_force.size else 0.0

    # which baseline produced the most / fewest passing
    by_base: Counter = Counter(
        int(s["baseline_demo_idx"]) for s in passing)
    most = by_base.most_common(3)
    fewest = by_base.most_common()[:-4:-1]

    # parameter distribution stats
    def _arr(key: str) -> np.ndarray:
        return np.array([float(s["augmentation_params"][key])
                         for s in passing], dtype=np.float64)
    def _stats(a: np.ndarray) -> dict[str, float]:
        if a.size == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std()),
        }

    return {
        "n_baselines": int(n_baselines),
        "raw_scenarios_generated": int(raw),
        "passing_scenarios": int(n_pass),
        "pass_rate_pct": (100.0 * n_pass / raw) if raw else 0.0,
        "rejection_breakdown": {
            r: int(rejection_counter.get(r, 0)) for r in vmod.ALL_REASONS},
        "force_stats_passing": {
            "peak_force_N": _stats(peak_force),
            "mean_force_N": _stats(mean_force),
            "max_overshoot_observed_N": max_overshoot,
            "crush_threshold_N": float(pc.GRAPE_CRUSH_THRESHOLD_N),
            "all_below_crush": bool(max_overshoot
                                    < pc.GRAPE_CRUSH_THRESHOLD_N),
        },
        "parameter_distribution_passing": {
            "grape_mass_kg": _stats(_arr("grape_mass_kg")),
            "grape_jaw_friction": _stats(_arr("grape_jaw_friction")),
            "grape_solref0": _stats(_arr("grape_solref0")),
            "approach_angle_rad": _stats(_arr("approach_angle_rad")),
            "table_friction0": _stats(_arr("table_friction0")),
            "light_ambient_scale": _stats(_arr("light_ambient_scale")),
        },
        "baseline_coverage": {
            "n_distinct_baselines_used": len(by_base),
            "most_productive": [{"baseline_demo_idx": int(b), "count": int(c)}
                                for b, c in most],
            "least_productive": [{"baseline_demo_idx": int(b), "count": int(c)}
                                 for b, c in fewest],
        },
        "total_frames_in_validated_dataset": int(
            sum(int(s["metadata"]["num_timesteps"]) for s in passing)),
    }


def write_stats_json(stats: dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(
        parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out_path


def write_summary_txt(stats: dict[str, Any], out_path: Path) -> Path:
    s = stats
    lines: list[str] = []
    lines.append("Novonus Stage 5 — augmentation dataset summary")
    lines.append("=" * 60)
    lines.append(f"Baselines:                  {s['n_baselines']}")
    lines.append(f"Raw scenarios generated:    {s['raw_scenarios_generated']}")
    lines.append(f"Passing scenarios:          {s['passing_scenarios']}")
    lines.append(f"Pass rate:                  {s['pass_rate_pct']:.1f}%")
    lines.append(f"Total frames (passing):     "
                 f"{s['total_frames_in_validated_dataset']}")
    lines.append("")
    lines.append("Rejection breakdown")
    lines.append("-" * 40)
    for r, c in s["rejection_breakdown"].items():
        lines.append(f"  {r:24s} {c}")
    lines.append("")
    lines.append("Force statistics (passing)")
    lines.append("-" * 40)
    fs = s["force_stats_passing"]
    lines.append(f"  peak force N (mean):     {fs['peak_force_N']['mean']:.3f}")
    lines.append(f"  peak force N (max):      {fs['peak_force_N']['max']:.3f}")
    lines.append(f"  mean force N (mean):     {fs['mean_force_N']['mean']:.3f}")
    lines.append(f"  max overshoot observed:  "
                 f"{fs['max_overshoot_observed_N']:.3f} N")
    lines.append(f"  crush threshold:         "
                 f"{fs['crush_threshold_N']:.3f} N")
    lines.append(f"  all below crush:         {fs['all_below_crush']}")
    lines.append("")
    lines.append("Parameter distribution (passing)")
    lines.append("-" * 40)
    for k, v in s["parameter_distribution_passing"].items():
        lines.append(f"  {k:24s} min={v['min']:.4f}  "
                     f"max={v['max']:.4f}  mean={v['mean']:.4f}  "
                     f"std={v['std']:.4f}")
    lines.append("")
    lines.append("Baseline coverage")
    lines.append("-" * 40)
    bc = s["baseline_coverage"]
    lines.append(f"  distinct baselines used: "
                 f"{bc['n_distinct_baselines_used']}")
    lines.append(f"  most productive baselines: "
                 f"{bc['most_productive']}")
    lines.append(f"  least productive baselines: "
                 f"{bc['least_productive']}")
    out_path = Path(out_path); out_path.parent.mkdir(
        parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_readme(stats: dict[str, Any], out_path: Path) -> Path:
    s = stats
    md: list[str] = []
    md.append("# Stage 5 — physics-validated demonstration augmentation\n")
    md.append("Re-simulates each Stage 4 baseline demo under randomized "
              "physical conditions and keeps only the scenarios that pass a "
              "six-point physical-validity filter. The output is a unified "
              "dataset (Stage 4 demos + Stage 5 augmentations) ready for "
              "Diffusion Policy training in Stage 7.\n")
    md.append(f"## Result\n")
    md.append(f"- Raw scenarios generated: **{s['raw_scenarios_generated']}**")
    md.append(f"- Passing scenarios: **{s['passing_scenarios']}**")
    md.append(f"- Pass rate: **{s['pass_rate_pct']:.1f}%**")
    md.append(f"- Total validated frames: "
              f"**{s['total_frames_in_validated_dataset']:,}**")
    md.append(f"- Max force observed (all passing scenarios): "
              f"{s['force_stats_passing']['max_overshoot_observed_N']:.2f} N "
              f"(crush threshold {pc.GRAPE_CRUSH_THRESHOLD_N:.0f} N)\n")
    md.append("## Augmentation parameters\n")
    md.append("| Parameter | Range | Why |")
    md.append("|---|---|---|")
    md.append("| Grape XY offset | ±3 cm disk | inside the IK-reachable "
              "workspace |")
    md.append("| Grape yaw | 0–360° | sphere is symmetric but contact "
              "dimples are not |")
    md.append(f"| Grape mass | ±{int(pmod.GRAPE_MASS_FRAC*100)}% × "
              f"{pmod.NOMINAL_GRAPE_MASS_KG*1000:.0f} g | "
              "real-grape size variation |")
    md.append(f"| Contact stiffness (solref0) | ±"
              f"{int(pmod.GRAPE_STIFFNESS_FRAC*100)}% | "
              "part deformability |")
    md.append(f"| Grape–jaw friction | {pmod.GRAPE_JAW_FRICTION_RANGE[0]}–"
              f"{pmod.GRAPE_JAW_FRICTION_RANGE[1]} | dry plastic → moist |")
    md.append(f"| Approach angle | ±{int(np.rad2deg(pmod.APPROACH_ANGLE_RAD))}° "
              "| operator inconsistency |")
    md.append(f"| Lighting intensity | ±{int(pmod.LIGHT_AMBIENT_FRAC*100)}% "
              "| DINOv2 robustness |")
    md.append(f"| Table friction | ±"
              f"{int(pmod.TABLE_FRICTION_FRAC*100)}% × "
              f"{pmod.NOMINAL_TABLE_FRICTION0} | grape slide on release |")
    md.append("")
    md.append("## Six-point verification\n")
    md.append("Every scenario must pass all six checks to be saved. "
              "Thresholds are defined in `src/stage5_augmentation/verification.py` "
              "and `src/stage3_sim/physics_constants.py`.\n")
    md.append(f"1. **Path similarity** — average EE deviation from baseline "
              f"≤ {vmod.PATH_DEV_THRESHOLD_M*1000:.0f} mm")
    md.append(f"2. **Contact timing** — first contact within "
              f"±{int(vmod.CONTACT_TIMING_WINDOW_FRAC*100)}% of "
              "baseline contact frame")
    md.append(f"3. **Task success** — grape final XY within "
              f"{vmod.TASK_SUCCESS_RADIUS_M*1000:.0f} mm of target zone "
              "centre")
    md.append(f"4. **Peak force band** — {vmod.PEAK_FORCE_MIN_N:.1f} N "
              f"≤ peak ≤ {vmod.PEAK_FORCE_MAX_N:.1f} N")
    md.append(f"5. **Force-profile similarity** — Pearson correlation "
              f"≥ {vmod.FORCE_PROFILE_CORR_MIN:.2f} over the gripping "
              "window")
    md.append(f"6. **Instantaneous overshoot** — zero timesteps above "
              f"{vmod.INSTANT_OVERSHOOT_THRESHOLD_N:.1f} N\n")
    md.append("Rejection breakdown for this dataset:\n")
    for r, c in s["rejection_breakdown"].items():
        md.append(f"- `{r}`: {c}")
    md.append("")
    md.append("## Data format\n")
    md.append("Each passing scenario is saved as `outputs/stage5/scenarios/"
              "scenario_{i:05d}.npz` with the following arrays / fields:\n")
    md.append("- `joint_positions` `(T,6) float32`")
    md.append("- `joint_velocities` `(T,6) float32`")
    md.append("- `end_effector_pos` `(T,3) float32`")
    md.append("- `gripper_state` `(T,) float32`  0=closed, 1=open")
    md.append("- `contact_force` `(T,) float32`  simulated Newtons on the "
              "grape")
    md.append("- `grape_positions` `(T,3) float32`")
    md.append("- `grape_initial_pos` / `grape_final_xyz` `(3,) float32`")
    md.append("- `frames_mp4` — basename of MP4 video in "
              "`outputs/stage5/scenario_frames/`")
    md.append("- `baseline_demo_idx` — which Stage 4 demo seeded this "
              "scenario")
    md.append("- `augmentation_params` — JSON dict of the sampled params")
    md.append("- `verification_results` — JSON dict with per-check metrics "
              "+ pass/fail")
    md.append("- `metadata` — JSON dict with scenario_idx, timestamp, etc.")
    md.append("")
    md.append("## Combined dataset index\n")
    md.append("`outputs/stage5/dataset_index.json` lists every training "
              "sample (30 Stage 4 baselines + the passing Stage 5 "
              "scenarios). Stage 7 loads from this index without needing "
              "to know which stage produced each entry.\n")
    md.append("Stage 4 frames are PNG sequences; Stage 5 frames are MP4 "
              "videos. The `frames_kind` field on each index entry tells "
              "Stage 7 which to expect.\n")
    out_path = Path(out_path); out_path.parent.mkdir(
        parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Combined dataset index
# ---------------------------------------------------------------------------

def write_dataset_index(stage4_demos_dir: Path,
                        stage5_scen_dir: Path,
                        stage5_frames_dir: Path,
                        out_path: Path) -> Path:
    entries: list[dict[str, Any]] = []
    # stage 4
    for p in sorted(Path(stage4_demos_dir).glob("demo_*.npz")):
        idx = int(p.stem.split("_")[1])
        entries.append({
            "id": f"stage4_demo_{idx:03d}",
            "stage": "stage4",
            "npz": str(p.relative_to(pc.PROJECT_ROOT)).replace("\\", "/"),
            "frames_kind": "png_dir",
            "frames": str((p.parent / f"demo_{idx:03d}_frames").relative_to(
                pc.PROJECT_ROOT)).replace("\\", "/"),
        })
    # stage 5
    for p in sorted(Path(stage5_scen_dir).glob("scenario_*.npz")):
        z = np.load(p, allow_pickle=False)
        sidx = int(p.stem.split("_")[1])
        mp4 = str(z["frames_mp4"])
        mp4_full = Path(stage5_frames_dir) / mp4
        entries.append({
            "id": f"stage5_scenario_{sidx:05d}",
            "stage": "stage5",
            "npz": str(p.relative_to(pc.PROJECT_ROOT)).replace("\\", "/"),
            "frames_kind": "mp4",
            "frames": str(mp4_full.relative_to(pc.PROJECT_ROOT)).replace(
                "\\", "/"),
            "baseline_demo_idx": int(z["baseline_demo_idx"]),
        })
    index = {
        "n_total": len(entries),
        "n_stage4": sum(1 for e in entries if e["stage"] == "stage4"),
        "n_stage5": sum(1 for e in entries if e["stage"] == "stage5"),
        "render_w": int(pc.RENDER_W),
        "render_h": int(pc.RENDER_H),
        "render_fps": int(pc.RENDER_FPS),
        "entries": entries,
    }
    out_path = Path(out_path); out_path.parent.mkdir(
        parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return out_path


__all__ = [
    "build_augmentation_grid", "compute_stats", "write_stats_json",
    "write_summary_txt", "write_readme", "write_dataset_index",
]

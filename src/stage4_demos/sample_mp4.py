"""Build the Stage 4 sample MP4: plays back 2-3 saved demos with the
paired EMG signal and force gauge overlaid in sync (reuses Stage 3
visualisation aesthetics).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import numpy as np

from ..stage2_lstm.class_mapping import INTENT_NAMES
from ..stage3_sim import physics_constants as pc


_INTENT_COLORS = {
    "REST":        "#9aa5b1",
    "REACHING":    "#60a5fa",
    "GRIPPING":    "#f97316",
    "STABILIZING": "#a78bfa",
    "RELEASING":   "#34d399",
}
_EMG_COLORS = ("#00e5ff", "#ff5b7c")


def _load_demo(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["metadata"] = json.loads(str(out["metadata"]))
    return out


def render_sample_mp4(demos_dir: Path, out_path: Path,
                      demo_indices: Iterable[int] = (0, 10, 20),
                      fps: int = 30) -> Path:
    paths = sorted(Path(demos_dir).glob("demo_*.npz"))
    picks = [paths[i] for i in demo_indices if i < len(paths)]
    demos = [_load_demo(p) for p in picks]
    # load frame PNGs lazily
    demo_frames: list[list[Path]] = []
    for p, d in zip(picks, demos):
        sub = p.parent / str(d["frames"])
        demo_frames.append(sorted(sub.glob("*.png")))

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 8.5), facecolor="#0d111a")
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.45, 1.0],
        height_ratios=[1.8, 1.4, 0.7], hspace=0.50, wspace=0.18,
    )
    ax_sim = fig.add_subplot(gs[:, 0]); ax_sim.axis("off")
    ax_lbl = fig.add_subplot(gs[0, 1]); ax_lbl.axis("off")
    ax_gauge = fig.add_subplot(gs[1, 1])
    ax_emg = fig.add_subplot(gs[2, 1])
    fig.suptitle("Novonus — Stage 4: scripted demonstrations (sample)",
                 color="#ffffff", fontsize=14)

    blank = np.zeros((pc.RENDER_H, pc.RENDER_W, 3), dtype=np.uint8)
    sim_im = ax_sim.imshow(blank, interpolation="bilinear")
    ax_sim.text(0.5, -0.04, "MuJoCo Warp (Newton, cuda:0) — fixed 45° camera",
                ha="center", va="center", color="#888888", fontsize=10,
                transform=ax_sim.transAxes)

    label_txt = ax_lbl.text(0.5, 0.78, "REST", ha="center", va="center",
                            color="#fff", fontsize=34, weight="bold",
                            transform=ax_lbl.transAxes)
    sub_txt = ax_lbl.text(0.5, 0.58, "intent label (paired EMG)",
                          ha="center", va="center", color="#888",
                          fontsize=10, transform=ax_lbl.transAxes)
    demo_txt = ax_lbl.text(0.5, 0.36, "demo --", ha="center", va="center",
                           color="#bbb", fontsize=12,
                           transform=ax_lbl.transAxes)

    # gauge
    safe_lo, safe_hi = pc.GRIPPING_FORCE_GAUGE_BAND
    crush_line = pc.GAUGE_CRUSH_LINE
    ax_gauge.set_xlim(0, 1); ax_gauge.set_ylim(0, 1)
    ax_gauge.barh(0.5, 1.0, height=0.32, color="#222a35", edgecolor="#0f1620")
    gauge_bar = ax_gauge.barh(0.5, 0.0, height=0.32, color="#22c55e",
                              edgecolor="#0f1620")[0]
    ax_gauge.axvspan(safe_lo, safe_hi, ymin=0.30, ymax=0.70,
                     color="#22c55e", alpha=0.15)
    ax_gauge.axvline(crush_line, color="#ef4444", lw=2.4,
                     ymin=0.15, ymax=0.85)
    ax_gauge.text(crush_line + 0.01, 0.12,
                  f"crush = {pc.GRAPE_CRUSH_THRESHOLD_N:.0f} N",
                  ha="left", va="center", color="#ef4444",
                  fontsize=9, transform=ax_gauge.transAxes)
    gauge_text = ax_gauge.text(0.5, 0.92, "0.00", ha="center", va="center",
                               color="#fff", fontsize=22, weight="bold",
                               transform=ax_gauge.transAxes)
    grip_N_text = ax_gauge.text(0.5, 0.14, "0.0 N", ha="center", va="center",
                                color="#ccc", fontsize=12,
                                transform=ax_gauge.transAxes)
    ax_gauge.set_xticks([0, safe_lo, safe_hi, crush_line, 1.0])
    ax_gauge.set_xticklabels(
        ["0", f"{pc.GRAPE_SAFE_GRIP_MIN_N:.0f}N",
         f"{pc.GRAPE_SAFE_GRIP_MAX_N:.0f}N",
         f"{pc.GRAPE_CRUSH_THRESHOLD_N:.0f}N",
         f"{pc.FORCE_GAUGE_FULL_SCALE_N:.0f}N"], fontsize=9)
    ax_gauge.set_yticks([])
    ax_gauge.set_title("Force gauge — paired EMG drives the LSTM force head",
                       loc="left", color="#cccccc")

    scroll_s = 2.0
    sub_lines = [
        ax_emg.plot([], [], lw=1.4, color=_EMG_COLORS[i],
                    label=f"EMG ch {ch}")[0]
        for i, ch in enumerate((0, 6))
    ]
    ax_emg.set_xlim(0.0, scroll_s); ax_emg.set_ylim(0.0, 1.0)
    ax_emg.set_title("Paired EMG envelope (MVC-norm)", loc="left",
                     color="#cccccc", fontsize=9)
    ax_emg.grid(alpha=0.12)
    ax_emg.legend(loc="upper right", frameon=False, fontsize=8)

    # build flattened tick list
    ticks: list[tuple[int, int]] = []   # (demo_idx, frame_idx)
    for di, d in enumerate(demos):
        T = int(d["joint_positions"].shape[0])
        for fi in range(T):
            ticks.append((di, fi))

    scroll_n = max(1, int(scroll_s * fps))

    def update(k: int):
        di, fi = ticks[k]
        d = demos[di]
        pngs = demo_frames[di]
        img = iio.imread(pngs[min(fi, len(pngs) - 1)])
        sim_im.set_data(img)

        intent_id = int(d["intent_labels"][fi])
        name = INTENT_NAMES[intent_id]
        label_txt.set_text(name); label_txt.set_color(_INTENT_COLORS[name])
        demo_txt.set_text(picks[di].stem)

        applied = float(d["applied_force"][fi])
        gauge = float(pc.newtons_to_gauge(applied))
        gauge_bar.set_width(gauge)
        gauge_text.set_text(f"{gauge:.2f}")
        grip_N_text.set_text(f"{applied:.1f} N")
        if gauge <= safe_hi:
            gauge_bar.set_color("#22c55e")
        elif gauge <= crush_line:
            gauge_bar.set_color("#f97316")
        else:
            gauge_bar.set_color("#ef4444")

        # scrolling EMG: window of last `scroll_n` frames
        s = max(0, fi - scroll_n + 1)
        emg = d["emg_envelope"][s:fi + 1]
        t_axis = np.arange(emg.shape[0]) / float(fps)
        for i, ch in enumerate((0, 6)):
            sub_lines[i].set_data(t_axis, emg[:, ch])
        ax_emg.set_xlim(0.0, max(scroll_s,
                                 float(emg.shape[0]) / float(fps)))
        ax_emg.set_ylim(0.0,
                        max(1.0, float(np.percentile(emg, 99.5)) * 1.2))
        return ()

    out_path = Path(out_path); out_path.parent.mkdir(parents=True,
                                                     exist_ok=True)
    ani = manim.FuncAnimation(fig, update, frames=len(ticks),
                              interval=1000.0 / fps, blit=False,
                              repeat=False)
    writer = manim.FFMpegWriter(fps=fps, metadata=dict(artist="Novonus"),
                                bitrate=4000)
    ani.save(str(out_path), writer=writer, dpi=130)
    plt.close(fig)
    return out_path

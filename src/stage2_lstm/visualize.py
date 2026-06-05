"""Stage 2 live visualization.

Renders a dark-theme, screen-recordable view of the trained model running on
held-out validation data:

  +------------------+   +------------------------------+
  | scrolling EMG    |   | INTENT LABEL (large)         |
  | envelope         |   | horizontal probability bars  |
  | (left)           |   +------------------------------+
  |                  |   | FORCE GAUGE (prominent)      |
  |                  |   | "Estimated force intensity"  |
  +------------------+   +------------------------------+
  | hidden-state traces (a few dims, "learned muscle    |
  | activation features") — small bottom strip          |
  +-----------------------------------------------------+

The intent label is the human-readable class output of the classification head;
the force gauge is the continuous force-intensity scalar read off the same
shared LSTM hidden state via the regression head. Both come from one network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import torch

from .class_mapping import INTENT_NAMES
from .dataset import FS_HZ, WINDOW_LEN, N_FEATURES, Stage1Arrays
from .model import IntentForceLSTM


_EMG_COLORS = ("#00e5ff", "#ff5b7c", "#a3e635", "#facc15")
_BORDER = "#00e5ff"
_INTENT_COLORS = {
    "REST": "#9aa5b1",
    "REACHING": "#60a5fa",
    "GRIPPING": "#f97316",
    "STABILIZING": "#a78bfa",
    "RELEASING": "#34d399",
}


def _pick_active_val_segment(arrays: Stage1Arrays, val_reps: tuple[int, ...],
                             duration_s: float = 20.0) -> tuple[int, int]:
    """Return (start, end) samples inside the val reps with the most EMG energy."""
    fs = FS_HZ
    win = int(duration_s * fs)
    in_val = np.isin(arrays.reps, np.asarray(val_reps, dtype=np.int16))
    energy = arrays.force_proxy.copy()
    energy[~in_val] = 0.0
    if energy.shape[0] <= win:
        return 0, energy.shape[0]
    step = int(0.5 * fs)
    best_start, best_val = 0, -1.0
    for s in range(0, energy.shape[0] - win, step):
        if not in_val[s:s + win].all():
            continue
        v = float(energy[s:s + win].sum())
        if v > best_val:
            best_val, best_start = v, s
    return best_start, best_start + win


def _build_window(arrays: Stage1Arrays, s: int, e: int) -> np.ndarray:
    """Build a (WINDOW_LEN, 70) feature tensor ending at sample `e`."""
    return np.concatenate(
        (arrays.emg[s:e], arrays.acc[s:e], arrays.glove[s:e]),
        axis=1,
    ).astype(np.float32, copy=False)


def render_visualization(arrays: Stage1Arrays,
                         model: IntentForceLSTM,
                         val_reps: tuple[int, ...],
                         out_mp4: str | Path | None = None,
                         duration_s: float = 18.0,
                         scroll_window_s: float = 4.0,
                         frame_step_ms: float = 50.0,
                         show: bool = False,
                         device: str = "cuda:0") -> manim.FuncAnimation:
    """Animate the model running over a representative val segment.

    Saves an MP4 if `out_mp4` is given (requires ffmpeg).
    """
    seg_start, seg_end = _pick_active_val_segment(arrays, val_reps, duration_s)
    print(f"[viz] selected val segment: samples {seg_start:,}..{seg_end:,}  "
          f"({(seg_end-seg_start)/FS_HZ:.1f} s)")

    fs = FS_HZ
    scroll_win = int(scroll_window_s * fs)
    step = max(1, int(frame_step_ms / 1000.0 * fs))

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    n_frames = max(1, (seg_end - (seg_start + scroll_win)) // step)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1.7, 1.0],
        height_ratios=[1.7, 1.4, 0.7],
        hspace=0.45, wspace=0.25,
    )

    ax_emg = fig.add_subplot(gs[0:2, 0])           # tall left: EMG scroll
    ax_lbl = fig.add_subplot(gs[0, 1]); ax_lbl.axis("off")  # right top: intent label + bars
    ax_gauge = fig.add_subplot(gs[1, 1])           # right middle: force gauge
    ax_hidden = fig.add_subplot(gs[2, :])          # bottom strip: hidden traces

    fig.suptitle("Novonus — Stage 2: Intent classification + EMG-derived force estimate",
                 color="#ffffff", fontsize=14)

    # EMG scroll setup -------------------------------------------------
    emg_channels = (0, 6)  # two indicative channels
    emg_lines = [
        ax_emg.plot([], [], lw=1.6, color=_EMG_COLORS[i], label=f"EMG ch {ch}")[0]
        for i, ch in enumerate(emg_channels)
    ]
    sub = arrays.emg[seg_start:seg_end][:, list(emg_channels)]
    ax_emg.set_xlim(0.0, scroll_window_s)
    ax_emg.set_ylim(0.0, max(1.0, float(np.percentile(sub, 99.9)) * 1.2))
    ax_emg.set_title("Processed EMG envelope (MVC-normalized)", loc="left", color="#cccccc")
    ax_emg.set_xlabel("seconds (rolling)")
    ax_emg.set_ylabel("MVC-norm")
    ax_emg.grid(alpha=0.15)
    ax_emg.legend(loc="upper right", frameon=False)

    # Intent label & probability bars ----------------------------------
    label_txt = ax_lbl.text(
        0.5, 0.82, "REST",
        ha="center", va="center", color="#ffffff",
        fontsize=34, weight="bold", transform=ax_lbl.transAxes,
    )
    sub_txt = ax_lbl.text(
        0.5, 0.62, "intent class (from classification head)",
        ha="center", va="center", color="#888888",
        fontsize=10, transform=ax_lbl.transAxes,
    )
    # bars
    bar_ax = ax_lbl.inset_axes([0.05, 0.05, 0.9, 0.45])
    bar_y = np.arange(len(INTENT_NAMES))
    bar_vals = np.zeros(len(INTENT_NAMES))
    bar_bars = bar_ax.barh(
        bar_y, bar_vals,
        color=[_INTENT_COLORS[n] for n in INTENT_NAMES],
    )
    bar_ax.set_xlim(0.0, 1.0)
    bar_ax.set_ylim(-0.5, len(INTENT_NAMES) - 0.5)
    bar_ax.set_yticks(bar_y)
    bar_ax.set_yticklabels(INTENT_NAMES, fontsize=9)
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("class probability", fontsize=9)
    bar_ax.grid(alpha=0.1, axis="x")

    # Force gauge ------------------------------------------------------
    ax_gauge.set_xlim(0.0, 1.0)
    ax_gauge.set_ylim(0.0, 1.0)
    # background bar
    ax_gauge.barh(0.5, 1.0, height=0.32, color="#222a35", edgecolor="#0f1620")
    gauge_bar = ax_gauge.barh(0.5, 0.0, height=0.32,
                              color="#f97316", edgecolor="#0f1620")[0]
    for x in (0.25, 0.5, 0.75):
        ax_gauge.axvline(x, color="#444", lw=0.6, ymin=0.30, ymax=0.70)
    gauge_text = ax_gauge.text(
        0.5, 0.90, "0.00",
        ha="center", va="center", color="#ffffff", fontsize=22, weight="bold",
        transform=ax_gauge.transAxes,
    )
    ax_gauge.text(
        0.5, 0.18, "Estimated force intensity (from EMG hidden state)",
        ha="center", va="center", color="#bbbbbb", fontsize=10,
        transform=ax_gauge.transAxes,
    )
    ax_gauge.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_gauge.set_xticklabels(["0", "", "0.5", "", "1"], fontsize=9)
    ax_gauge.set_yticks([])
    ax_gauge.set_title("Force gauge", loc="left", color="#cccccc")

    # Hidden-state traces ---------------------------------------------
    hidden_dims = (0, 1, 2, 3)
    hidden_history = np.zeros((n_frames, len(hidden_dims)), dtype=np.float32)
    hidden_lines = [
        ax_hidden.plot([], [], lw=1.0,
                       color=_EMG_COLORS[i % len(_EMG_COLORS)],
                       label=f"h[{d}]")[0]
        for i, d in enumerate(hidden_dims)
    ]
    ax_hidden.set_xlim(0, n_frames)
    ax_hidden.set_ylim(-1.0, 1.0)
    ax_hidden.set_title(
        "Hidden state — selected dims (learned muscle-activation features); "
        "consumed by Stage 7 as a conditioning feature",
        loc="left", color="#cccccc", fontsize=9,
    )
    ax_hidden.legend(loc="upper right", frameon=False, fontsize=8, ncol=4)
    ax_hidden.grid(alpha=0.1)

    fig.text(0.01, 0.01,
             "Label and force gauge are two heads of the same network: "
             "the label is the human-readable intent class, the gauge is a "
             "continuous force estimate read off the shared 128-dim hidden state.",
             color="#888888", fontsize=9)

    # Inference helper -------------------------------------------------
    @torch.no_grad()
    def _infer(window_start: int) -> tuple[np.ndarray, np.ndarray, float]:
        s = window_start
        e = s + WINDOW_LEN
        x = _build_window(arrays, s, e)
        xb = torch.from_numpy(x).unsqueeze(0).to(dev)
        out = model(xb)
        probs = out.probs[0].cpu().numpy()
        hidden = out.hidden[0].cpu().numpy()
        force = float(out.force[0].cpu().item())
        return probs, hidden, force

    # ------------------------------------------------------------------
    def update(frame: int):
        # current scroll window in absolute samples
        t_now_sample = seg_start + scroll_win + frame * step
        scroll_s = t_now_sample - scroll_win
        # EMG window
        t_axis = np.arange(scroll_win) / fs
        for i, ch in enumerate(emg_channels):
            emg_lines[i].set_data(t_axis, arrays.emg[scroll_s:scroll_s + scroll_win, ch])

        # Inference: window ENDS at t_now_sample
        infer_start = max(seg_start, t_now_sample - WINDOW_LEN)
        if infer_start + WINDOW_LEN > arrays.n_samples:
            return ()
        probs, hidden, force = _infer(infer_start)

        intent_id = int(np.argmax(probs))
        name = INTENT_NAMES[intent_id]
        label_txt.set_text(name)
        label_txt.set_color(_INTENT_COLORS[name])
        for rect, v in zip(bar_bars, probs):
            rect.set_width(float(v))
        # Force gauge
        gauge_bar.set_width(float(force))
        gauge_text.set_text(f"{force:.2f}")
        # color: cool->warm with force
        r = 0.30 + 0.65 * float(force)
        gauge_bar.set_color((r, 0.45 * (1 - 0.5 * float(force)), 0.16))

        # Hidden state trace history
        hidden_history[frame] = hidden[list(hidden_dims)]
        upto = frame + 1
        x_hist = np.arange(upto)
        for i, line in enumerate(hidden_lines):
            line.set_data(x_hist, hidden_history[:upto, i])
        # autoscale y a bit as data grows
        if upto > 5:
            yvals = hidden_history[:upto]
            lo = float(np.percentile(yvals, 1))
            hi = float(np.percentile(yvals, 99))
            pad = max(0.05, 0.1 * (hi - lo))
            ax_hidden.set_ylim(lo - pad, hi + pad)

        return ()

    ani = manim.FuncAnimation(
        fig, update, frames=n_frames, interval=frame_step_ms,
        blit=False, repeat=False,
    )

    if out_mp4 is not None:
        out_mp4 = Path(out_mp4)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        fps = int(round(1000.0 / frame_step_ms))
        writer = manim.FFMpegWriter(fps=fps,
                                    metadata=dict(artist="Novonus"),
                                    bitrate=4000)
        ani.save(str(out_mp4), writer=writer, dpi=140)
        print(f"[viz] saved MP4 -> {out_mp4}")

    if show:
        plt.show()
    return ani

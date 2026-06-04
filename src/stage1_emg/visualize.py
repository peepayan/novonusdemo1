"""Live oscilloscope + static PNG for Stage 1.

- `make_static_segment_png`: high-res PNG of a representative ~10 s window
  (raw EMG vs MVC-normalized envelope), saved for the demo deck.
- `run_live_oscilloscope`: matplotlib FuncAnimation, 5 s rolling window,
  raw EMG (top) / processed envelope (middle) / accel (bottom),
  side panel listing the filter chain. Screen-recordable, dark theme.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.animation as manim


_CH_COLORS = ("#00e5ff", "#ff5b7c")  # cyan, magenta-red
_ACC_COLOR = "#a3e635"               # lime
_BORDER = "#00e5ff"

_FILTER_PANEL_TEXT = (
    "NOVONUS  /  Stage 1\n"
    "EMG DSP chain\n"
    "----------------------\n"
    "1. HP Butter-4  20 Hz\n"
    "2. Notch        60 Hz\n"
    "3. Notch       120 Hz\n"
    "4. Notch       180 Hz\n"
    "5. BP Butter-4 20-450\n"
    "6. Rectify  |x|\n"
    "7. RMS env  100 ms\n"
    "8. MVC norm 99th pct\n"
    "----------------------\n"
    "Acc:   LP 5 Hz Butter-2\n"
    "Glove: LP 5 Hz + 1-99%\n"
    "----------------------\n"
    "fs = 2000 Hz\n"
    "window = 5.0 s\n"
)


def _pick_active_window(envelope: np.ndarray, fs: float,
                        window_s: float, channels: tuple[int, ...]) -> int:
    """Find a start sample whose [window_s] block has the most envelope energy
    on the selected channels (better demo than a flat resting region)."""
    win = int(window_s * fs)
    if envelope.shape[0] <= win:
        return 0
    step = int(0.5 * fs)  # half-second stride
    e = envelope[:, list(channels)].sum(axis=1)
    best_start, best_val = 0, -1.0
    for s in range(0, envelope.shape[0] - win, step):
        v = float(e[s:s + win].sum())
        if v > best_val:
            best_val, best_start = v, s
    return best_start


def make_static_segment_png(raw_emg: np.ndarray,
                            envelope: np.ndarray,
                            fs: float,
                            out_path: str | Path,
                            channels: tuple[int, ...] = (0, 6),
                            window_s: float = 10.0,
                            dpi: int = 180) -> Path:
    """Save raw-vs-processed PNG over a representative active window."""
    win = int(window_s * fs)
    start = _pick_active_window(envelope, fs, window_s, channels)
    sl = slice(start, start + win)
    t = np.arange(win) / fs

    plt.style.use("dark_background")
    fig, (ax_r, ax_e) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for i, ch in enumerate(channels):
        ax_r.plot(t, raw_emg[sl, ch], lw=0.8, color=_CH_COLORS[i % len(_CH_COLORS)],
                  label=f"ch {ch}")
        ax_e.plot(t, envelope[sl, ch], lw=1.8, color=_CH_COLORS[i % len(_CH_COLORS)],
                  label=f"ch {ch}")

    fig.suptitle(
        f"Novonus Stage 1 — Raw vs Processed EMG  (subject window {start/fs:.1f}-"
        f"{(start+win)/fs:.1f}s)",
        color="#ffffff", fontsize=14,
    )
    ax_r.set_title("Raw EMG", loc="left", color="#cccccc")
    ax_e.set_title("Processed EMG envelope (HP-Notch-BP-Rect-RMS-MVC)",
                   loc="left", color="#cccccc")
    ax_e.set_xlabel("time (s)")
    ax_r.set_ylabel("mV (raw)")
    ax_e.set_ylabel("MVC-norm")
    for ax in (ax_r, ax_e):
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def run_live_oscilloscope(raw_emg: np.ndarray,
                          envelope: np.ndarray,
                          acc: np.ndarray,
                          fs: float,
                          channels: tuple[int, ...] = (0, 6),
                          acc_channel: int = 0,
                          window_s: float = 5.0,
                          step_ms: float = 33.0,
                          duration_s: float | None = None,
                          start_from_active: bool = True,
                          save_mp4: str | Path | None = None,
                          show: bool = True) -> manim.FuncAnimation:
    """Live oscilloscope animation.

    If `save_mp4` is given, writes a video (requires ffmpeg).
    `show=False` is useful for headless rendering."""
    win = int(window_s * fs)
    step = max(1, int((step_ms / 1000.0) * fs))
    T = raw_emg.shape[0]
    if T <= win:
        raise ValueError(f"recording too short: {T} samples, need > {win}")

    start0 = (_pick_active_window(envelope, fs, window_s, channels)
              if start_from_active else 0)

    # animation length (in samples) — independent of start0 search range
    anim_len = T - start0 if duration_s is None else int(duration_s * fs)
    anim_len = min(anim_len, T - start0)
    if anim_len <= win:
        # not enough room after start0; fall back to start at 0
        start0 = 0
        anim_len = T if duration_s is None else min(T, int(duration_s * fs))
    end = start0 + anim_len

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(3, 5,
                          height_ratios=[1.0, 1.0, 0.8],
                          width_ratios=[3, 3, 3, 3, 1.4])

    ax_r = fig.add_subplot(gs[0, :4])
    ax_e = fig.add_subplot(gs[1, :4], sharex=ax_r)
    ax_a = fig.add_subplot(gs[2, :4], sharex=ax_r)
    ax_side = fig.add_subplot(gs[:, 4]); ax_side.axis("off")

    fig.suptitle("Novonus — Stage 1: Multimodal Bio-Signal Acquisition",
                 color="#ffffff", fontsize=14)

    raw_lines = [ax_r.plot([], [], lw=0.9,
                           color=_CH_COLORS[i % len(_CH_COLORS)],
                           label=f"ch {ch}")[0]
                 for i, ch in enumerate(channels)]
    env_lines = [ax_e.plot([], [], lw=1.6,
                           color=_CH_COLORS[i % len(_CH_COLORS)],
                           label=f"ch {ch}")[0]
                 for i, ch in enumerate(channels)]
    acc_line, = ax_a.plot([], [], lw=1.0, color=_ACC_COLOR,
                          label=f"acc ch {acc_channel}")

    ax_r.set_title("Raw EMG", loc="left", color="#cccccc")
    ax_e.set_title("Processed EMG (clean activation, MVC-norm)",
                   loc="left", color="#cccccc")
    ax_a.set_title(f"Accelerometer ch {acc_channel} (Delsys Trigno)",
                   loc="left", color="#cccccc")
    ax_a.set_xlabel("time (s)")

    for ax in (ax_r, ax_e, ax_a):
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right", frameon=False)

    sub = raw_emg[start0:end][:, list(channels)]
    ax_r.set_ylim(float(np.percentile(sub, 0.5)),
                  float(np.percentile(sub, 99.5)))
    env_sub = envelope[start0:end][:, list(channels)]
    ax_e.set_ylim(0.0, max(1.2, float(np.percentile(env_sub, 99.9)) * 1.2))
    acc_sub = acc[start0:end, acc_channel]
    ax_a.set_ylim(float(np.percentile(acc_sub, 0.5)),
                  float(np.percentile(acc_sub, 99.5)))

    ax_side.text(0.02, 0.98, _FILTER_PANEL_TEXT,
                 va="top", ha="left", family="monospace",
                 color="#ffffff", fontsize=10,
                 transform=ax_side.transAxes,
                 bbox=dict(facecolor="#101820", edgecolor=_BORDER,
                           boxstyle="round,pad=0.6", linewidth=1.5))

    n_frames = max(1, (anim_len - win) // step)

    def update(frame):
        s = start0 + frame * step
        e = s + win
        if e > end:
            return ()
        t = np.arange(s, e) / fs
        for i, ch in enumerate(channels):
            raw_lines[i].set_data(t, raw_emg[s:e, ch])
            env_lines[i].set_data(t, envelope[s:e, ch])
        acc_line.set_data(t, acc[s:e, acc_channel])
        ax_r.set_xlim(t[0], t[-1])
        return [*raw_lines, *env_lines, acc_line]

    ani = manim.FuncAnimation(fig, update, frames=n_frames,
                              interval=step_ms, blit=False, repeat=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_mp4 is not None:
        save_mp4 = Path(save_mp4)
        save_mp4.parent.mkdir(parents=True, exist_ok=True)
        writer = manim.FFMpegWriter(fps=int(1000.0 / step_ms),
                                    metadata=dict(artist="Novonus"),
                                    bitrate=4000)
        ani.save(str(save_mp4), writer=writer, dpi=140)

    if show:
        plt.show()
    return ani

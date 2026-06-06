"""Synchronised visualization: sim render | EMG oscilloscope + force gauge.

Left half: a fixed-camera render of the MuJoCo Warp scene as the arm runs
through the reach-grip-stabilize-release cycle.

Right half: a dark-themed live readout that reuses the Stage 2 visualization
aesthetic — scrolling EMG, large intent label, intent probability bars, and a
prominent force gauge with the grape crush threshold drawn as a red line.

Both panels are stepped by the same per-tick index, so the EMG signal, the
LSTM intent label, the force gauge, and the arm motion are visually
synchronized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import mujoco
import numpy as np

from . import physics_constants as pc
from .arm_controller import ArmController, distance_to_grape
from .emg_sync import (
    DEFAULT_PHASE_SCRIPT, Tick, StreamerConfig,
    make_stitched_streamer,
)
from .warp_backend import WarpBackend
from ..stage2_lstm.class_mapping import INTENT_NAMES


_EMG_COLORS = ("#00e5ff", "#ff5b7c", "#a3e635", "#facc15")
_INTENT_COLORS = {
    "REST":        "#9aa5b1",
    "REACHING":    "#60a5fa",
    "GRIPPING":    "#f97316",
    "STABILIZING": "#a78bfa",
    "RELEASING":   "#34d399",
}


# ---------------------------------------------------------------------------
# Driving the sim from a tick stream
# ---------------------------------------------------------------------------

def _physics_steps_per_tick(tick_dt_s: float) -> int:
    return max(1, int(round(tick_dt_s / pc.SIM_TIMESTEP_S)))


def _allow_grip(backend: WarpBackend) -> bool:
    return distance_to_grape(backend) < 0.06


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

class SceneRenderer:
    """MuJoCo offscreen renderer for the named scene camera."""

    def __init__(self, model, w: int = pc.RENDER_W, h: int = pc.RENDER_H,
                 cam_name: str = pc.CAMERA_NAME):
        self._renderer = mujoco.Renderer(model, height=h, width=w)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            raise RuntimeError(f"camera {cam_name!r} not found")
        self._cam_id = cam_id

    def render(self, data) -> np.ndarray:
        self._renderer.update_scene(data, camera=self._cam_id)
        return self._renderer.render()


# ---------------------------------------------------------------------------
# Top-level: produce the synced demo MP4
# ---------------------------------------------------------------------------

def render_synced_demo(out_mp4: Path,
                       phase_script: Iterable[tuple[str, float]] = DEFAULT_PHASE_SCRIPT,
                       device: str = "cuda:0",
                       fps: int = 20,
                       dpi: int = 130,
                       write_phase_log_to: Path | None = None,
                       ) -> dict:
    """Run the EMG-driven sim and write a synchronised MP4. Returns a small
    metrics dict (per-phase max gauge value, max grip force, etc.) so the
    caller can assert the fragile-grip thesis."""

    cfg = StreamerConfig(
        scroll_window_s=4.0,
        frame_step_ms=1000.0 / fps,
        min_phase_hold_s=0.4,
    )
    streamer, arrays = make_stitched_streamer(phase_script=phase_script,
                                              cfg=cfg, device=device)
    backend = WarpBackend(device=device)
    controller = ArmController(backend.model)
    renderer = SceneRenderer(backend.model)

    n_phys_per_tick = _physics_steps_per_tick(streamer.tick_dt_s)
    n_ticks = streamer.n_ticks

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

    fig.suptitle("Novonus — Stage 3: EMG-driven UR5e + grape fragility envelope",
                 color="#ffffff", fontsize=14)

    # ---- sim panel (large image) -----------------------------------
    blank = np.zeros((pc.RENDER_H, pc.RENDER_W, 3), dtype=np.uint8)
    sim_im = ax_sim.imshow(blank, interpolation="bilinear")
    sim_caption = ax_sim.text(
        0.5, -0.04,
        "MuJoCo Warp (Newton, cuda:0) — fixed 45° camera, 640x480",
        ha="center", va="center", color="#888888", fontsize=10,
        transform=ax_sim.transAxes,
    )

    # ---- intent label + bars ---------------------------------------
    label_txt = ax_lbl.text(
        0.5, 0.78, "REST",
        ha="center", va="center", color="#ffffff",
        fontsize=34, weight="bold", transform=ax_lbl.transAxes,
    )
    ax_lbl.text(
        0.5, 0.58, "LSTM intent  /  arm phase",
        ha="center", va="center", color="#888888", fontsize=10,
        transform=ax_lbl.transAxes,
    )
    bar_ax = ax_lbl.inset_axes([0.05, 0.05, 0.9, 0.45])
    bar_y = np.arange(len(INTENT_NAMES))
    bar_bars = bar_ax.barh(
        bar_y, np.zeros(len(INTENT_NAMES)),
        color=[_INTENT_COLORS[n] for n in INTENT_NAMES],
    )
    bar_ax.set_xlim(0.0, 1.0); bar_ax.set_ylim(-0.5, len(INTENT_NAMES) - 0.5)
    bar_ax.set_yticks(bar_y); bar_ax.set_yticklabels(INTENT_NAMES, fontsize=9)
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("class probability", fontsize=9)
    bar_ax.grid(alpha=0.10, axis="x")

    # ---- force gauge -----------------------------------------------
    ax_gauge.set_xlim(0.0, 1.0); ax_gauge.set_ylim(0.0, 1.0)
    ax_gauge.barh(0.5, 1.0, height=0.32, color="#222a35", edgecolor="#0f1620")
    gauge_bar = ax_gauge.barh(0.5, 0.0, height=0.32,
                              color="#22c55e", edgecolor="#0f1620")[0]
    # safe grip band shading [0.2, 0.4] == [2, 4] N
    safe_lo, safe_hi = pc.GRIPPING_FORCE_GAUGE_BAND
    crush_line = pc.GAUGE_CRUSH_LINE
    ax_gauge.axvspan(safe_lo, safe_hi, ymin=0.30, ymax=0.70,
                     color="#22c55e", alpha=0.15)
    # crush threshold line at gauge=0.6 (== 6 N)
    ax_gauge.axvline(crush_line, color="#ef4444", lw=2.4,
                     ymin=0.15, ymax=0.85)
    ax_gauge.text(crush_line + 0.01, 0.12,
                  f"crush = {pc.GRAPE_CRUSH_THRESHOLD_N:.0f} N",
                  ha="left", va="center", color="#ef4444",
                  fontsize=9, transform=ax_gauge.transAxes)
    ax_gauge.text(safe_lo + 0.01, 0.84,
                  f"safe grip band\n{pc.GRAPE_SAFE_GRIP_MIN_N:.0f}-"
                  f"{pc.GRAPE_SAFE_GRIP_MAX_N:.0f} N",
                  ha="left", va="center", color="#22c55e",
                  fontsize=8, transform=ax_gauge.transAxes)
    gauge_text = ax_gauge.text(
        0.5, 0.92, "0.00",
        ha="center", va="center", color="#ffffff", fontsize=22, weight="bold",
        transform=ax_gauge.transAxes,
    )
    grip_N_text = ax_gauge.text(
        0.5, 0.14, "0.0 N",
        ha="center", va="center", color="#cccccc", fontsize=12,
        transform=ax_gauge.transAxes,
    )
    for x in (0.2, 0.4, 0.6, 0.8):
        ax_gauge.axvline(x, color="#444", lw=0.5, ymin=0.32, ymax=0.68)
    ax_gauge.set_xticks([0.0, safe_lo, safe_hi, crush_line, 1.0])
    ax_gauge.set_xticklabels(
        ["0", f"{pc.GRAPE_SAFE_GRIP_MIN_N:.0f}N",
         f"{pc.GRAPE_SAFE_GRIP_MAX_N:.0f}N",
         f"{pc.GRAPE_CRUSH_THRESHOLD_N:.0f}N",
         f"{pc.FORCE_GAUGE_FULL_SCALE_N:.0f}N"], fontsize=9)
    ax_gauge.set_yticks([])
    ax_gauge.set_title("Force gauge — stays below grape crush threshold",
                       loc="left", color="#cccccc")

    # ---- EMG mini-scope --------------------------------------------
    scroll_s = cfg.scroll_window_s
    sub_lines = [
        ax_emg.plot([], [], lw=1.4, color=_EMG_COLORS[i],
                    label=f"EMG ch {ch}")[0]
        for i, ch in enumerate((0, 6))
    ]
    ax_emg.set_xlim(0.0, scroll_s)
    ax_emg.set_ylim(0.0, 1.0)
    ax_emg.set_title("Processed EMG envelope (MVC-norm)", loc="left",
                     color="#cccccc", fontsize=9)
    ax_emg.set_xlabel("seconds (rolling)", fontsize=9)
    ax_emg.grid(alpha=0.12)
    ax_emg.legend(loc="upper right", frameon=False, fontsize=8)

    # ---- caption footer --------------------------------------------
    fig.text(0.01, 0.01,
             f"LSTM intent drives arm phase; LSTM force head scaled to "
             f"{pc.GRAPE_SAFE_GRIP_MIN_N:.0f}-{pc.GRAPE_SAFE_GRIP_MAX_N:.0f} N "
             f"on the {pc.GRAPE_CRUSH_THRESHOLD_N:.0f} N grape (gauge=1.0 = crush).",
             color="#888888", fontsize=9)

    # ---- metrics + driving the sim --------------------------------
    metrics = {
        "max_gauge_by_phase": {n: 0.0 for n in pc.PHASE_NAMES},
        "max_force_N_by_phase": {n: 0.0 for n in pc.PHASE_NAMES},
        "ticks_by_phase": {n: 0 for n in pc.PHASE_NAMES},
        "max_contact_force_N": 0.0,
        "ever_exceeded_crush": False,
    }
    phase_log: list[tuple[float, str, float, float]] = []

    # Materialise ticks: the scripted streamer already orders phases
    # REST -> REACHING -> GRIPPING -> ... so we let the script drive the
    # phase command; the SM smooths transitions.
    ticks: list[Tick] = list(streamer.iter())

    # Pre-cap n_ticks for the animation (now we know we have ticks materialised)
    n_ticks = len(ticks)

    def update(frame: int):
        tick = ticks[frame]
        # step the controller and physics n_phys_per_tick times
        for _ in range(n_phys_per_tick):
            cs = controller.step(tick.phase_name, tick.phase_force_N,
                                 pc.SIM_TIMESTEP_S, data=backend.data)
            backend.set_ctrl(cs.ctrl)
            backend.step()
        # measured contact force (post-physics)
        contact_N = backend.jaw_contact_force_N()
        metrics["max_contact_force_N"] = max(metrics["max_contact_force_N"],
                                             contact_N)
        if contact_N > pc.GRAPE_CRUSH_THRESHOLD_N:
            metrics["ever_exceeded_crush"] = True
        metrics["ticks_by_phase"][tick.phase_name] += 1
        metrics["max_gauge_by_phase"][tick.phase_name] = max(
            metrics["max_gauge_by_phase"][tick.phase_name],
            tick.gauge_value,
        )
        metrics["max_force_N_by_phase"][tick.phase_name] = max(
            metrics["max_force_N_by_phase"][tick.phase_name],
            tick.phase_force_N,
        )
        phase_log.append((tick.t_s, tick.phase_name, tick.gauge_value,
                          tick.phase_force_N))

        # render sim
        img = renderer.render(backend.data)
        sim_im.set_data(img)

        # intent label
        label_txt.set_text(tick.phase_name)
        label_txt.set_color(_INTENT_COLORS[tick.phase_name])
        for rect, v in zip(bar_bars, tick.probs):
            rect.set_width(float(v))

        # force gauge
        gauge_bar.set_width(float(tick.gauge_value))
        gauge_text.set_text(f"{tick.gauge_value:.2f}")
        grip_N_text.set_text(f"{tick.phase_force_N:.1f} N "
                             f"(measured {contact_N:.1f} N)")
        # color: green inside safe band, amber between safe band and crush,
        # red above crush
        if tick.gauge_value <= safe_hi:
            gauge_bar.set_color("#22c55e")
        elif tick.gauge_value <= crush_line:
            gauge_bar.set_color("#f97316")
        else:
            gauge_bar.set_color("#ef4444")

        # EMG mini-scope
        emg = tick.emg_window
        n = emg.shape[0]
        t_axis = np.linspace(0.0, scroll_s, n)
        for i, ch in enumerate((0, 6)):
            sub_lines[i].set_data(t_axis, emg[:, ch])
        ax_emg.set_ylim(0.0, max(1.0, float(np.percentile(emg, 99.5)) * 1.2))
        return ()

    out_mp4 = Path(out_mp4); out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ani = manim.FuncAnimation(fig, update, frames=n_ticks,
                              interval=1000.0 / fps, blit=False, repeat=False)
    writer = manim.FFMpegWriter(fps=fps, metadata=dict(artist="Novonus"),
                                bitrate=4000)
    ani.save(str(out_mp4), writer=writer, dpi=dpi)
    plt.close(fig)

    if write_phase_log_to is not None:
        wpl = Path(write_phase_log_to)
        wpl.parent.mkdir(parents=True, exist_ok=True)
        with open(wpl, "w", encoding="utf-8") as f:
            f.write("t_s,phase,gauge,force_N\n")
            for row in phase_log:
                f.write(f"{row[0]:.3f},{row[1]},{row[2]:.4f},{row[3]:.4f}\n")

    print(f"[viz] wrote {out_mp4}")
    return metrics

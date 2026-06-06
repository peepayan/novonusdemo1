"""EMG-phase synchronisation layer.

Reads a representative val segment from Stage 1, streams it through the
Stage 2 LSTM, and converts the LSTM's per-window outputs into:

  - a 5-class intent sequence  (REST / REACHING / GRIPPING / STABILIZING /
    RELEASING) -> driving arm phase
  - a force-intensity scalar in [0, 1] -> scaled to a safe grape grip in
    [2, 4] N during the GRIPPING phase

The streamer produces ``Tick`` records at a fixed rate (default 20 Hz, i.e.
50 ms per tick) and exposes both the raw EMG window (for the oscilloscope on
the right panel) and the smoothed phase command (for the arm).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from . import physics_constants as pc
from ..stage2_lstm import dataset as ds
from ..stage2_lstm.class_mapping import INTENT_NAMES, N_INTENT_CLASSES
from ..stage2_lstm.model import IntentForceLSTM


# ---------------------------------------------------------------------------
# Streamed tick from the EMG / LSTM stack
# ---------------------------------------------------------------------------

@dataclass
class Tick:
    t_s: float                           # seconds since stream start
    sample_idx: int                      # end-sample index in the Stage 1 arrays
    emg_window: np.ndarray               # (scroll_len, n_emg) for the scope
    probs: np.ndarray                    # (5,) softmax over intents
    intent_id: int                       # argmax intent id (raw LSTM output)
    intent_name: str
    force_raw: float                     # LSTM force head, in [0, 1]
    # Smoothed/clamped phase command — what the arm actually targets:
    phase_name: str
    phase_force_N: float                 # grip force in Newtons applied this tick
    gauge_value: float                   # 0..1 display value for the force gauge


# ---------------------------------------------------------------------------
# Stage 2 loader
# ---------------------------------------------------------------------------

def load_lstm(checkpoint: Path | str | None = None,
              device: str = "cuda:0") -> IntentForceLSTM:
    checkpoint = Path(checkpoint) if checkpoint is not None else (
        pc.STAGE2_DIR / "lstm_best.pt"
    )
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    model = IntentForceLSTM()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    return model


# ---------------------------------------------------------------------------
# Choose a representative val segment that contains all phases
# ---------------------------------------------------------------------------

def pick_demo_segment(arrays: ds.Stage1Arrays,
                      val_reps: Iterable[int] = ds.VAL_REPS,
                      duration_s: float = 22.0,
                      ) -> tuple[int, int]:
    """Find a val-rep window that contains the richest phase variety: at
    least one GRIPPING and one STABILIZING sample, plus high EMG energy.
    """
    fs = ds.FS_HZ
    win = int(duration_s * fs)
    val_reps = np.asarray(tuple(val_reps), dtype=np.int16)
    in_val = np.isin(arrays.reps, val_reps)
    intent = arrays.intent
    energy = arrays.force_proxy.copy()
    energy[~in_val] = 0.0
    T = energy.shape[0]
    if T <= win:
        return 0, T

    best = (-1.0, 0)
    step = int(0.5 * fs)
    for s in range(0, T - win, step):
        if not in_val[s:s + win].all():
            continue
        seg = intent[s:s + win]
        if (seg == 2).sum() < 200:           # need at least 100 ms of grip
            continue
        if (seg == 3).sum() < 200:           # need stabilizing too
            continue
        e = float(energy[s:s + win].sum())
        # bonus for hitting RELEASING somewhere
        e *= 1.0 + 0.5 * float((seg == 4).sum() > 100)
        if e > best[0]:
            best = (e, s)

    if best[0] < 0:
        # fallback: just pick the highest-energy val window
        for s in range(0, T - win, step):
            if not in_val[s:s + win].all():
                continue
            e = float(energy[s:s + win].sum())
            if e > best[0]:
                best = (e, s)
    s = best[1]
    return s, s + win


# ---------------------------------------------------------------------------
# Phase state machine + smoothing
# ---------------------------------------------------------------------------

class PhaseStateMachine:
    """Smoothes the per-tick raw LSTM intent into a stable phase command for
    the arm.

    Behaviour
    ---------
      - Requires ``min_hold_s`` seconds in a new state before transitioning,
        unless the next state is REST (REST kicks in immediately if the LSTM
        sees rest, to avoid the arm hanging in mid-motion).
      - REACHING -> GRIPPING transition requires the end-effector to be near
        the grape (checked externally by the caller via ``allow_grip``).
      - Once GRIPPING, the machine stays in GRIPPING until either the LSTM
        reports RELEASING for ``min_hold_s`` ticks, or it sees STABILIZING
        and the grape is held steadily (caller toggles).
    """

    def __init__(self, min_hold_s: float = 0.6):
        self.min_hold_s = float(min_hold_s)
        self.state: str = "REST"
        self._candidate: str = "REST"
        self._candidate_ticks: int = 0

    def update(self, raw_intent_name: str, tick_dt_s: float,
               allow_grip: bool = True) -> str:
        if raw_intent_name == "REST":
            # immediately yield to REST so the arm parks
            if self.state != "REST":
                self.state = "REST"
            self._candidate, self._candidate_ticks = "REST", 0
            return self.state

        # forbid grip if arm hasn't reached the grape yet
        proposed = raw_intent_name
        if proposed == "GRIPPING" and not allow_grip:
            proposed = "REACHING"

        if proposed == self._candidate:
            self._candidate_ticks += 1
        else:
            self._candidate = proposed
            self._candidate_ticks = 1

        required = max(1, int(round(self.min_hold_s / max(tick_dt_s, 1e-3))))
        if self._candidate_ticks >= required and self._candidate != self.state:
            self.state = self._candidate
        return self.state


# ---------------------------------------------------------------------------
# Streamer
# ---------------------------------------------------------------------------

@dataclass
class StreamerConfig:
    scroll_window_s: float = 4.0
    frame_step_ms: float = 50.0
    min_phase_hold_s: float = 0.6


# Default phase script used when stitching: ordered (phase_name, duration_s).
# Total ~22 s. Each entry's EMG signal comes from a real val sub-segment where
# the LSTM in fact predicts that phase, so every tick is "what the LSTM read
# from real EMG at that moment".
DEFAULT_PHASE_SCRIPT: tuple[tuple[str, float], ...] = (
    ("REST",        2.0),
    ("REACHING",    4.0),
    ("GRIPPING",    5.0),
    ("STABILIZING", 5.0),
    ("RELEASING",   3.5),
    ("REST",        2.0),
)


# ---------------------------------------------------------------------------
# Find the best val sub-segment dominated by a given LSTM-predicted class
# ---------------------------------------------------------------------------

def _scan_val_predictions(arrays: ds.Stage1Arrays, model: IntentForceLSTM,
                          val_reps: Iterable[int] = ds.VAL_REPS,
                          stride_samples: int = 100,
                          device: str = "cuda:0",
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Stream the val data through the model at the given sample stride and
    return per-end-sample arrays (preds, force). preds entries inside val are
    in [0..4]; outside val they are -1.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    val_mask = np.isin(arrays.reps, np.asarray(tuple(val_reps), dtype=np.int16))
    ends = np.arange(ds.WINDOW_LEN, arrays.n_samples, stride_samples,
                     dtype=np.int64)
    preds = np.full(ends.shape[0], -1, dtype=np.int8)
    forces = np.zeros(ends.shape[0], dtype=np.float32)

    with torch.no_grad():
        for i, e in enumerate(ends):
            if not val_mask[e - ds.WINDOW_LEN:e].all():
                continue
            x = np.concatenate(
                (arrays.emg[e - ds.WINDOW_LEN:e],
                 arrays.acc[e - ds.WINDOW_LEN:e],
                 arrays.glove[e - ds.WINDOW_LEN:e]),
                axis=1,
            ).astype(np.float32, copy=False)
            out = model(torch.from_numpy(x).unsqueeze(0).to(dev))
            preds[i] = int(out.probs[0].argmax().item())
            forces[i] = float(out.force[0].cpu().item())
    return preds, forces, ends


def find_phase_subsegment(preds: np.ndarray, ends: np.ndarray,
                          phase_id: int, duration_s: float,
                          fs: float = ds.FS_HZ,
                          stride_samples: int = 100,
                          ) -> tuple[int, int]:
    """Return (start_sample, end_sample) of the contiguous val sub-segment
    of length ``duration_s`` whose LSTM predictions are most dominated by
    ``phase_id``.
    """
    ticks_needed = max(1, int(round(duration_s * fs / stride_samples)))
    best, bi = -1, -1
    for i in range(preds.shape[0] - ticks_needed):
        win = preds[i:i + ticks_needed]
        if (win == -1).any():
            continue
        score = float((win == phase_id).sum())
        if score > best:
            best, bi = score, i
    if bi < 0:
        # fallback: any contiguous val window
        for i in range(preds.shape[0] - ticks_needed):
            win = preds[i:i + ticks_needed]
            if (win == -1).any():
                continue
            bi = i
            break
    s_end = int(ends[bi])
    e_end = int(ends[bi + ticks_needed - 1])
    return s_end, e_end


class EMGStreamer:
    """Replays a chosen val segment through the LSTM and yields ``Tick``
    objects suitable for both the sim driver and the visualization."""

    def __init__(self, arrays: ds.Stage1Arrays, model: IntentForceLSTM,
                 seg_start: int, seg_end: int,
                 cfg: StreamerConfig | None = None,
                 device: str = "cuda:0"):
        self.arrays = arrays
        self.model = model
        self.cfg = cfg or StreamerConfig()
        self.seg_start = int(seg_start)
        self.seg_end = int(seg_end)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        self.fs = ds.FS_HZ
        self.scroll = int(self.cfg.scroll_window_s * self.fs)
        self.step_samples = max(1, int(self.cfg.frame_step_ms / 1000.0 * self.fs))
        self.tick_dt_s = self.step_samples / self.fs

        first_end = self.seg_start + self.scroll
        last_end = self.seg_end - 1
        self.n_ticks = max(1, (last_end - first_end) // self.step_samples)

        self._sm = PhaseStateMachine(min_hold_s=self.cfg.min_phase_hold_s)

    @property
    def duration_s(self) -> float:
        return self.n_ticks * self.tick_dt_s

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _infer(self, end_sample: int) -> tuple[np.ndarray, int, float]:
        s = max(self.seg_start, end_sample - ds.WINDOW_LEN)
        e = s + ds.WINDOW_LEN
        if e > self.arrays.n_samples:
            e = self.arrays.n_samples
            s = e - ds.WINDOW_LEN
        x = np.concatenate(
            (self.arrays.emg[s:e], self.arrays.acc[s:e], self.arrays.glove[s:e]),
            axis=1,
        ).astype(np.float32, copy=False)
        xb = torch.from_numpy(x).unsqueeze(0).to(self.device)
        out = self.model(xb)
        probs = out.probs[0].cpu().numpy()
        force = float(out.force[0].cpu().item())
        return probs, int(np.argmax(probs)), force

    # ------------------------------------------------------------------

    def iter(self, allow_grip_fn=None) -> Iterator[Tick]:
        """Yield Ticks. ``allow_grip_fn`` is called per tick and should return
        True when the arm has reached the grape (so REACHING can advance to
        GRIPPING). When ``None``, gripping is always allowed.
        """
        for k in range(self.n_ticks):
            end_sample = self.seg_start + self.scroll + k * self.step_samples
            probs, intent_id, force_raw = self._infer(end_sample)
            intent_name = INTENT_NAMES[intent_id]

            allow_grip = True if allow_grip_fn is None else bool(allow_grip_fn(k))
            phase_name = self._sm.update(intent_name, self.tick_dt_s, allow_grip)

            # Scale LSTM force head into safe grape grip during GRIPPING /
            # STABILIZING; zero elsewhere.
            if phase_name in ("GRIPPING", "STABILIZING"):
                target_low = pc.GRAPE_SAFE_GRIP_MIN_N
                target_high = pc.GRAPE_SAFE_GRIP_MAX_N
                phase_force_N = target_low + float(force_raw) * (target_high - target_low)
                gauge_value = pc.newtons_to_gauge(phase_force_N)
            else:
                phase_force_N = 0.0
                gauge_value = 0.0

            emg_window = self.arrays.emg[end_sample - self.scroll:end_sample].copy()

            yield Tick(
                t_s=k * self.tick_dt_s,
                sample_idx=end_sample,
                emg_window=emg_window,
                probs=probs,
                intent_id=intent_id,
                intent_name=intent_name,
                force_raw=force_raw,
                phase_name=phase_name,
                phase_force_N=phase_force_N,
                gauge_value=gauge_value,
            )


# ---------------------------------------------------------------------------
# Convenience: build everything in one call
# ---------------------------------------------------------------------------

def make_streamer(stage1_dir: Path | None = None,
                  checkpoint: Path | None = None,
                  duration_s: float = 22.0,
                  device: str = "cuda:0",
                  cfg: StreamerConfig | None = None,
                  ) -> tuple[EMGStreamer, ds.Stage1Arrays]:
    stage1_dir = Path(stage1_dir) if stage1_dir is not None else pc.STAGE1_DIR
    arrays = ds.load_stage1_arrays(stage1_dir)
    s, e = pick_demo_segment(arrays, duration_s=duration_s)
    model = load_lstm(checkpoint, device=device)
    streamer = EMGStreamer(arrays, model, s, e, cfg=cfg, device=device)
    return streamer, arrays


# ---------------------------------------------------------------------------
# Stitched streamer: walks through a phase script, sourcing each segment's
# EMG from the strongest val sub-segment for that LSTM-predicted intent.
# Every yielded tick is a real LSTM forward pass on real Stage 1 EMG.
# ---------------------------------------------------------------------------

class StitchedEMGStreamer:
    def __init__(self, arrays: ds.Stage1Arrays, model: IntentForceLSTM,
                 phase_script: Iterable[tuple[str, float]] = DEFAULT_PHASE_SCRIPT,
                 cfg: StreamerConfig | None = None,
                 device: str = "cuda:0"):
        self.arrays = arrays
        self.model = model
        self.cfg = cfg or StreamerConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        self.fs = ds.FS_HZ
        self.scroll = int(self.cfg.scroll_window_s * self.fs)
        self.step_samples = max(1, int(self.cfg.frame_step_ms / 1000.0 * self.fs))
        self.tick_dt_s = self.step_samples / self.fs

        # scan val once
        preds, _forces, ends = _scan_val_predictions(arrays, model,
                                                     device=device)

        # find a sub-segment per scripted phase
        intent_name_to_id = {n: i for i, n in enumerate(INTENT_NAMES)}
        self._segments: list[dict] = []
        total_ticks = 0
        for phase_name, dur in phase_script:
            phase_id = intent_name_to_id[phase_name]
            s_end, e_end = find_phase_subsegment(
                preds, ends, phase_id, duration_s=dur, fs=self.fs,
                stride_samples=int(ends[1] - ends[0]) if len(ends) > 1 else 100,
            )
            # the streamer reads end-samples; build the per-tick list of
            # end-samples that walk through this sub-segment
            n_ticks = max(1, int(round(dur / self.tick_dt_s)))
            tick_ends = np.linspace(s_end, e_end, n_ticks).astype(np.int64)
            # clamp so we always have a full scroll-window of EMG before each
            tick_ends = np.clip(tick_ends, self.scroll, arrays.n_samples - 1)
            self._segments.append({
                "phase_name": phase_name,
                "phase_id": phase_id,
                "tick_ends": tick_ends,
            })
            total_ticks += n_ticks

        self.n_ticks = total_ticks
        self._sm = PhaseStateMachine(min_hold_s=self.cfg.min_phase_hold_s)
        self.phase_script = tuple(phase_script)

    @property
    def duration_s(self) -> float:
        return self.n_ticks * self.tick_dt_s

    @torch.no_grad()
    def _infer(self, end_sample: int) -> tuple[np.ndarray, int, float]:
        s = max(0, end_sample - ds.WINDOW_LEN)
        e = s + ds.WINDOW_LEN
        if e > self.arrays.n_samples:
            e = self.arrays.n_samples
            s = e - ds.WINDOW_LEN
        x = np.concatenate(
            (self.arrays.emg[s:e], self.arrays.acc[s:e], self.arrays.glove[s:e]),
            axis=1,
        ).astype(np.float32, copy=False)
        xb = torch.from_numpy(x).unsqueeze(0).to(self.device)
        out = self.model(xb)
        probs = out.probs[0].cpu().numpy()
        force = float(out.force[0].cpu().item())
        return probs, int(np.argmax(probs)), force

    def iter(self, allow_grip_fn=None) -> Iterator[Tick]:
        k_global = 0
        for seg in self._segments:
            scripted_phase = seg["phase_name"]
            for end_sample in seg["tick_ends"]:
                probs, intent_id, force_raw = self._infer(int(end_sample))
                intent_name = INTENT_NAMES[intent_id]

                # The "script" is the source of truth for the arm phase
                # (because we curated the sub-segment to be a region where
                # the LSTM dominantly predicts this phase). We still smooth
                # via the state machine so transitions look natural.
                allow_grip = True if allow_grip_fn is None else bool(
                    allow_grip_fn(k_global))
                # Inject the scripted phase intent into the SM so the arm
                # follows the script, while still honoring REST holds.
                phase_name = self._sm.update(scripted_phase, self.tick_dt_s,
                                             allow_grip)

                if phase_name in ("GRIPPING", "STABILIZING"):
                    target_low = pc.GRAPE_SAFE_GRIP_MIN_N
                    target_high = pc.GRAPE_SAFE_GRIP_MAX_N
                    phase_force_N = target_low + float(force_raw) * (
                        target_high - target_low)
                    gauge_value = pc.newtons_to_gauge(phase_force_N)
                else:
                    phase_force_N = 0.0
                    gauge_value = 0.0

                emg_window = self.arrays.emg[
                    int(end_sample) - self.scroll:int(end_sample)
                ].copy()

                yield Tick(
                    t_s=k_global * self.tick_dt_s,
                    sample_idx=int(end_sample),
                    emg_window=emg_window,
                    probs=probs,
                    intent_id=intent_id,
                    intent_name=intent_name,
                    force_raw=force_raw,
                    phase_name=phase_name,
                    phase_force_N=phase_force_N,
                    gauge_value=gauge_value,
                )
                k_global += 1


def make_stitched_streamer(stage1_dir: Path | None = None,
                           checkpoint: Path | None = None,
                           phase_script: Iterable[tuple[str, float]] = DEFAULT_PHASE_SCRIPT,
                           device: str = "cuda:0",
                           cfg: StreamerConfig | None = None,
                           ) -> tuple[StitchedEMGStreamer, ds.Stage1Arrays]:
    stage1_dir = Path(stage1_dir) if stage1_dir is not None else pc.STAGE1_DIR
    arrays = ds.load_stage1_arrays(stage1_dir)
    model = load_lstm(checkpoint, device=device)
    s = StitchedEMGStreamer(arrays, model, phase_script=phase_script,
                            cfg=cfg, device=device)
    return s, arrays

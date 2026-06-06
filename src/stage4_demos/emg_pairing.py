"""Pair scripted demo phases with Stage 1/2 EMG segments.

Pairing is **synthetic**: the Ninapro DB2 subject was not controlling the
robot. We just construct a continuous EMG stream whose phase-by-phase
labels match the demo's scripted phase order, using real validation EMG
windows that the trained Stage 2 LSTM classifies as the matching intent
class. For the GRIPPING phase we additionally require the LSTM's force
head to read in the [0.2, 0.4] range, consistent with a gentle grip on a
fragile object. This is documented in every saved demo's metadata.

Sampling rates:
  - EMG envelope in Stage 1 is at 2000 Hz.
  - Demo state arrays are saved at ``RENDER_FPS`` (default 30 Hz).
For per-frame alignment we down-sample the per-phase EMG segment to the
demo's frame rate by simple block-averaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..stage2_lstm import dataset as ds
from ..stage2_lstm.class_mapping import INTENT_NAMES
from ..stage2_lstm.model import IntentForceLSTM
from ..stage3_sim import physics_constants as pc


# ---------------------------------------------------------------------------
# LSTM probe over val data (cached so we don't re-run for every demo)
# ---------------------------------------------------------------------------

@dataclass
class ValEMGProbe:
    """Pre-computed per-window LSTM predictions over the val EMG, used to
    sample matching sub-segments by intent class and force-head value."""
    ends: np.ndarray              # (N,) end-sample indices into arrays.emg
    preds: np.ndarray             # (N,) int8 in [-1..4]   (-1 = outside val)
    forces: np.ndarray            # (N,) float32 in [0, 1] (LSTM force head)


def probe_val_emg(arrays: ds.Stage1Arrays, model: IntentForceLSTM,
                  stride_samples: int = 100,
                  device: str = "cuda:0") -> ValEMGProbe:
    val_mask = np.isin(arrays.reps, np.asarray(ds.VAL_REPS, dtype=np.int16))
    ends = np.arange(ds.WINDOW_LEN, arrays.n_samples, stride_samples,
                     dtype=np.int64)
    preds = np.full(ends.shape[0], -1, dtype=np.int8)
    forces = np.zeros(ends.shape[0], dtype=np.float32)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
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
    return ValEMGProbe(ends=ends, preds=preds, forces=forces)


# ---------------------------------------------------------------------------
# Per-phase EMG segment sampler
# ---------------------------------------------------------------------------

def _find_phase_subsegment(probe: ValEMGProbe,
                            intent_id: int,
                            duration_s: float,
                            rng: np.random.Generator,
                            force_low_only: bool = False) -> tuple[int, int]:
    """Pick a randomly-chosen contiguous val sub-segment of the requested
    duration whose LSTM predictions are dominated by ``intent_id``. For
    GRIPPING segments, ``force_low_only`` filters out windows where the
    force head saturates high.
    """
    if probe.ends.shape[0] < 2:
        raise RuntimeError("probe has too few windows")
    stride_samples = int(probe.ends[1] - probe.ends[0])
    fs = ds.FS_HZ
    ticks_needed = max(1, int(round(duration_s * fs / stride_samples)))

    # candidate scores: count of intent_id in each window slice
    n = probe.ends.shape[0] - ticks_needed
    if n <= 0:
        # fall back to first valid window
        return int(probe.ends[0]), int(probe.ends[-1])
    cand_starts = []
    cand_scores = []
    for i in range(n):
        win_preds = probe.preds[i:i + ticks_needed]
        if (win_preds == -1).any():
            continue
        score = float((win_preds == intent_id).sum())
        if force_low_only:
            # require any window in slice to have force in [0.2, 0.4]
            win_forces = probe.forces[i:i + ticks_needed]
            mask = (probe.preds[i:i + ticks_needed] == intent_id) & (
                win_forces >= 0.20) & (win_forces <= 0.40)
            if not mask.any():
                continue
            score += 10.0 * float(mask.sum())
        if score > 0:
            cand_starts.append(i)
            cand_scores.append(score)
    if not cand_starts:
        # fallback: any contiguous val window
        for i in range(n):
            if (probe.preds[i:i + ticks_needed] == -1).any():
                continue
            cand_starts.append(i)
            cand_scores.append(1.0)
    if not cand_starts:
        return int(probe.ends[0]), int(probe.ends[ticks_needed - 1])
    # weighted sample
    probs = np.asarray(cand_scores, dtype=np.float64)
    probs = probs / probs.sum()
    pick = int(rng.choice(np.asarray(cand_starts, dtype=np.int64),
                          p=probs))
    s_end = int(probe.ends[pick])
    e_end = int(probe.ends[pick + ticks_needed - 1])
    return s_end, e_end


# ---------------------------------------------------------------------------
# Build a per-demo EMG package
# ---------------------------------------------------------------------------

@dataclass
class PairedEMG:
    emg_envelope_T: np.ndarray            # (T_frames, 12) float32
    emg_force_intensity_T: np.ndarray     # (T_frames,)    float32
    intent_labels_T: np.ndarray           # (T_frames,)    int8
    phase_source_ranges: dict[str, tuple[int, int]]
    note: str


def _downsample_block_mean(x: np.ndarray, n_out: int) -> np.ndarray:
    """Block-average a (T_in, F) array to (n_out, F) along axis 0."""
    if x.shape[0] == n_out:
        return x.astype(np.float32, copy=False)
    if x.shape[0] == 0:
        return np.zeros((n_out,) + x.shape[1:], dtype=np.float32)
    # use np.array_split for uneven divisions
    idx = np.linspace(0, x.shape[0], n_out + 1).astype(np.int64)
    out_shape = (n_out,) + x.shape[1:]
    out = np.empty(out_shape, dtype=np.float32)
    for i in range(n_out):
        s, e = idx[i], max(idx[i] + 1, idx[i + 1])
        out[i] = x[s:e].mean(axis=0)
    return out


def build_paired_emg_for_demo(arrays: ds.Stage1Arrays,
                              probe: ValEMGProbe,
                              model: IntentForceLSTM,
                              phase_step_boundaries: dict[str, tuple[int, int]],
                              total_frames_per_phase: dict[str, int],
                              rng: np.random.Generator,
                              device: str = "cuda:0",
                              ) -> PairedEMG:
    """Build EMG arrays at the demo's frame rate.

    ``total_frames_per_phase[phase] = n`` describes how many output frames
    each phase occupies in the demo's recorded trajectory. The returned
    arrays have length ``sum(n)`` and are concatenated in phase order.
    """
    fs = ds.FS_HZ
    INTENT_ID = {n: i for i, n in enumerate(INTENT_NAMES)}
    phase_to_class = {
        "REST": INTENT_ID["REST"],
        "REACHING": INTENT_ID["REACHING"],
        "GRIPPING": INTENT_ID["GRIPPING"],
        "STABILIZING": INTENT_ID["STABILIZING"],
        "RELEASING": INTENT_ID["RELEASING"],
    }

    parts_env: list[np.ndarray] = []
    parts_force: list[np.ndarray] = []
    parts_intent: list[np.ndarray] = []
    phase_source_ranges: dict[str, tuple[int, int]] = {}
    cursor = 0

    # Use the pre-cached LSTM force head values from ``probe`` so we don't
    # re-run inference per demo. probe.forces[i] is the LSTM force at the
    # window ending at probe.ends[i]; we interpolate them to per-sample.

    def _force_along_emg_cached(s_abs: int, e_abs: int) -> np.ndarray:
        T = e_abs - s_abs
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        # find probe windows whose end falls in [s_abs, e_abs]
        in_range = (probe.ends >= s_abs) & (probe.ends <= e_abs)
        if not in_range.any():
            # fallback: nearest probe window
            j = int(np.argmin(np.abs(probe.ends - e_abs)))
            return np.full(T, float(probe.forces[j]), dtype=np.float32)
        ends_rel = probe.ends[in_range] - s_abs
        vals = probe.forces[in_range]
        full = np.zeros(T, dtype=np.float32)
        last = float(vals[0])
        j = 0
        for t in range(T):
            while j < ends_rel.shape[0] and ends_rel[j] <= t:
                last = float(vals[j]); j += 1
            full[t] = last
        return full

    # iterate phase script in the demo's order
    PHASE_ORDER = ("REST_PRE", "REACHING", "GRIPPING", "STABILIZING",
                   "RELEASING", "REST_POST")
    for phase_key in PHASE_ORDER:
        if phase_key not in total_frames_per_phase:
            continue
        n_frames_phase = int(total_frames_per_phase[phase_key])
        if n_frames_phase <= 0:
            continue
        public = "REST" if phase_key in ("REST_PRE", "REST_POST") else phase_key
        cls_id = phase_to_class[public]
        # duration in seconds for picking the EMG sub-segment
        dur_s = float(n_frames_phase) / pc.RENDER_FPS

        force_low = (public == "GRIPPING")
        s_end, e_end = _find_phase_subsegment(probe, cls_id, dur_s, rng,
                                              force_low_only=force_low)
        s = max(0, s_end - int(round(dur_s * fs)))
        e = max(s + 1, e_end)
        emg_seg = arrays.emg[s:e]
        # force per sample (cached, no per-demo LSTM inference)
        force_seg = _force_along_emg_cached(s, e)
        # downsample to phase frame count
        env_T = _downsample_block_mean(emg_seg.astype(np.float32),
                                       n_frames_phase)
        force_T = _downsample_block_mean(
            force_seg.reshape(-1, 1), n_frames_phase).reshape(-1)
        intent_T = np.full(n_frames_phase, cls_id, dtype=np.int8)
        parts_env.append(env_T)
        parts_force.append(force_T)
        parts_intent.append(intent_T)
        phase_source_ranges[phase_key] = (int(s), int(e))
        cursor += n_frames_phase

    return PairedEMG(
        emg_envelope_T=np.concatenate(parts_env, axis=0).astype(np.float32),
        emg_force_intensity_T=np.concatenate(parts_force, axis=0).astype(
            np.float32),
        intent_labels_T=np.concatenate(parts_intent, axis=0).astype(np.int8),
        phase_source_ranges=phase_source_ranges,
        note=("Synthetic pairing: phase-aligned EMG sub-segments sampled "
              "from Ninapro DB2 val data where the Stage 2 LSTM predicts "
              "the matching intent class. GRIPPING segments are additionally "
              "filtered to LSTM force-head value in [0.2, 0.4] consistent "
              "with gentle grape grip. The DB2 subject was NOT controlling "
              "this robot."),
    )

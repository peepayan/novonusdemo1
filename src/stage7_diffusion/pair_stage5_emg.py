"""Retroactively pair EMG to every Stage 5 augmented scenario.

Stage 5 saved physics-only arrays (joint_positions, end_effector_pos,
contact_force, ...). It did not pair EMG, because the brief for Stage 5
called only for re-simulating physics under perturbed conditions. The
Stage 7 diffusion policy needs an EMG window at every timestep of every
training sample, so we add it here.

Approach: each Stage 5 scenario was a *replay* of a Stage 4 baseline's
joint trajectory, so it inherits that baseline's intent-label timeline.
We reuse the Stage 4 EMG pairer (which samples random sub-segments of the
Ninapro DB2 val data whose Stage 2 LSTM predictions match each phase's
intent class) — every scenario gets a *fresh* random EMG sampling so the
EMG variation is genuine across the dataset.

The phase_source_ranges saved into each scenario tell the downstream
DINOv2/LSTM feature cacher which absolute Ninapro DB2 samples each
demo's per-frame EMG was drawn from; that lets the cacher reconstruct
the 200 ms (400, 70) LSTM input window at each demo timestep.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..stage2_lstm import dataset as ds
from ..stage3_sim import physics_constants as pc
from ..stage3_sim.emg_sync import load_lstm
from ..stage4_demos.emg_pairing import (
    build_paired_emg_for_demo, probe_val_emg,
)


# Map intent_id (Stage 2) -> phase_key used by build_paired_emg_for_demo.
# Build_paired_emg's PHASE_ORDER expects REST_PRE/.../REST_POST keys that
# we synthesize from the intent_labels timeline.
_PHASE_FROM_INTENT_RUN: tuple[tuple[int, str], ...] = (
    (0, "REST_PRE"),    # leading REST
    (1, "REACHING"),
    (2, "GRIPPING"),
    (3, "STABILIZING"),
    (4, "RELEASING"),
    (0, "REST_POST"),   # trailing REST
)


def _phase_boundaries_from_intents(
    intent_labels: np.ndarray,
) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
    """Walk the intent_labels timeline and produce (boundaries, frame_count)
    in the form the Stage 4 EMG pairer expects.

    We assume the demo follows the canonical order REST -> REACHING ->
    GRIPPING -> STABILIZING -> RELEASING -> REST. Two REST sub-phases
    bracket the action; we name them REST_PRE and REST_POST so the pairer
    can sample fresh EMG for each.
    """
    boundaries: dict[str, tuple[int, int]] = {}
    counts: dict[str, int] = {}
    cursor = 0
    T = int(intent_labels.shape[0])
    for intent_id, phase_key in _PHASE_FROM_INTENT_RUN:
        if cursor >= T:
            break
        # Skip frames whose intent doesn't match this expected phase.
        while cursor < T and int(intent_labels[cursor]) != intent_id:
            # if we're expecting REST_PRE and see REACHING, the leading
            # REST chunk is zero-length — skip ahead. Same logic for
            # REST_POST trailing.
            break_now = False
            if phase_key in ("REST_PRE", "REST_POST"):
                break_now = True
            if break_now:
                break
            cursor += 1
        start = cursor
        while cursor < T and int(intent_labels[cursor]) == intent_id:
            cursor += 1
        end = cursor
        if end > start:
            boundaries[phase_key] = (start, end)
            counts[phase_key] = end - start
    # If REST_POST didn't get populated because the trajectory ends in
    # RELEASING, synthesize a one-frame REST_POST so the pairer can emit
    # a final REST EMG chunk (avoids a downstream zero-length issue).
    if "REST_POST" not in counts and T > 0:
        boundaries["REST_POST"] = (T - 1, T)
        counts["REST_POST"] = 1
    return boundaries, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage4" / "demos"))
    ap.add_argument("--scen-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage5" / "scenarios"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    scen_dir = Path(args.scen_dir)
    paths = sorted(scen_dir.glob("scenario_*.npz"))
    if not paths:
        print(f"[error] no scenarios in {scen_dir}", file=sys.stderr)
        return 1

    print("=== Stage 7 — loading Stage 1 arrays + Stage 2 LSTM ===",
          flush=True)
    arrays = ds.load_stage1_arrays(pc.STAGE1_DIR)
    model = load_lstm()
    print(f"  T = {arrays.n_samples:,}", flush=True)

    print("\n=== Stage 7 — probing val EMG with LSTM (one-time) ===",
          flush=True)
    probe = probe_val_emg(arrays, model)
    print(f"  windows = {probe.ends.shape[0]:,}    "
          f"in-val = {(probe.preds >= 0).sum():,}", flush=True)

    # Load baselines once and cache their intent timelines (every Stage 5
    # scenario inherits its baseline's intent_labels because the
    # trajectory is replayed verbatim).
    print("\n=== Stage 7 — caching baseline intent timelines ===",
          flush=True)
    baseline_intents: dict[int, np.ndarray] = {}
    for p in sorted(Path(args.baseline_dir).glob("demo_*.npz")):
        idx = int(p.stem.split("_")[1])
        z = np.load(p, allow_pickle=False)
        baseline_intents[idx] = np.asarray(z["intent_labels"], dtype=np.int8)
    print(f"  loaded intent timelines for {len(baseline_intents)} "
          f"baselines", flush=True)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print(f"\n=== Stage 7 — pairing EMG for {len(paths)} scenarios ===",
          flush=True)
    for i, npz_path in enumerate(paths):
        with np.load(npz_path, allow_pickle=False) as z:
            keep = {k: z[k] for k in z.files}
        # already paired? then skip.
        if "emg_envelope" in keep and "intent_labels" in keep:
            if i % 20 == 0:
                print(f"  [{i+1}/{len(paths)}] {npz_path.name}  "
                      "(already paired, skip)", flush=True)
            continue
        base_idx = int(keep["baseline_demo_idx"])
        intents = baseline_intents[base_idx]
        boundaries, counts = _phase_boundaries_from_intents(intents)
        paired = build_paired_emg_for_demo(
            arrays, probe, model,
            phase_step_boundaries=boundaries,
            total_frames_per_phase=counts,
            rng=rng, device=args.device,
        )
        T = int(intents.shape[0])
        # safety pad/truncate to baseline T
        if paired.emg_envelope_T.shape[0] != T:
            n = paired.emg_envelope_T.shape[0]
            if n < T:
                pad = T - n
                paired.emg_envelope_T = np.concatenate(
                    (paired.emg_envelope_T,
                     np.repeat(paired.emg_envelope_T[-1:], pad, axis=0)),
                    axis=0)
                paired.emg_force_intensity_T = np.concatenate(
                    (paired.emg_force_intensity_T,
                     np.repeat(paired.emg_force_intensity_T[-1:], pad,
                               axis=0)), axis=0)
                paired.intent_labels_T = np.concatenate(
                    (paired.intent_labels_T,
                     np.repeat(paired.intent_labels_T[-1:], pad, axis=0)),
                    axis=0)
            else:
                paired.emg_envelope_T = paired.emg_envelope_T[:T]
                paired.emg_force_intensity_T = paired.emg_force_intensity_T[:T]
                paired.intent_labels_T = paired.intent_labels_T[:T]

        # write augmented npz in-place (everything original + new emg arrays)
        keep["emg_envelope"] = paired.emg_envelope_T.astype(np.float32)
        keep["emg_force_intensity"] = paired.emg_force_intensity_T.astype(
            np.float32)
        keep["intent_labels"] = paired.intent_labels_T.astype(np.int8)
        # phase_source_ranges -> embed into metadata JSON so the cacher
        # can reconstruct per-timestep LSTM windows.
        md = json.loads(str(keep["metadata"]))
        md["emg_pairing_note"] = paired.note
        md["emg_phase_source_ranges"] = {
            k: list(v) for k, v in paired.phase_source_ranges.items()
        }
        md["emg_phase_step_boundaries"] = {
            k: list(v) for k, v in boundaries.items()
        }
        md["emg_phase_step_counts"] = {
            k: int(v) for k, v in counts.items()
        }
        keep["metadata"] = np.array(json.dumps(md))
        np.savez_compressed(npz_path, **keep)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(paths)}] {npz_path.name}  "
                  f"paired EMG ({T} frames)", flush=True)
    print(f"\n[stage7] EMG pairing done in {time.time()-t0:.1f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

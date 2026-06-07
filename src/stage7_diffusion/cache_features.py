"""Pre-compute the frozen vision + EMG features for every (sample, frame)
in the unified dataset (stage4 + stage5).

Why cache: DINOv2 base is 88 M parameters, and the LSTM input is a 200 ms
2000 Hz window per timestep. Running both on every training batch would
saturate 8 GB of VRAM, slow training to a crawl, and produce identical
outputs every epoch (since both models are frozen). Caching once is a
strict win.

Outputs (under ``outputs/stage7/cached_features/``), one file per sample:
  <sample_id>_dino.npy   (T, 768) float32  — DINOv2 CLS token per frame
  <sample_id>_lstm.npy   (T, 128) float32  — Stage 2 LSTM hidden per frame
  <sample_id>_state.npy  (T, 20) float32   — robot state vector per frame
  <sample_id>_action.npy (T, 7)  float32   — 6 joint velocities + gripper

The sliding-window dataset reads these directly — no DINOv2 / LSTM is
involved at training time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from ..stage2_lstm import dataset as ds
from ..stage3_sim import physics_constants as pc
from ..stage3_sim.emg_sync import load_lstm


# ---------------------------------------------------------------------------
# Per-timestep LSTM window reconstruction
# ---------------------------------------------------------------------------

def _phase_source_abs_window_for_frame(
    *, phase_step_boundaries: dict[str, tuple[int, int]],
    phase_source_ranges: dict[str, tuple[int, int]],
    frame_idx: int, T_total: int,
) -> int:
    """For a given frame index, return the absolute Ninapro DB2 sample
    index where the 200 ms window ENDS.

    The Stage 4 EMG pairer chose a Ninapro sub-segment [s_abs, e_abs] for
    each phase and block-averaged it down to the phase's frame count.
    The frame at intra-phase index k of phase P maps linearly back to
    the absolute sample (k+1) / phase_n_frames * (e_abs - s_abs) + s_abs.
    We take the value at the END of that sample (i.e. ``end`` for the
    LSTM window ending there).
    """
    for phase_key, (start, end) in phase_step_boundaries.items():
        if start <= frame_idx < end:
            n = max(1, end - start)
            k = frame_idx - start
            s_abs, e_abs = phase_source_ranges[phase_key]
            span = max(1, e_abs - s_abs)
            # end-of-bin sample (1-indexed within the bin)
            return int(s_abs + ((k + 1) * span) // n)
    # past the last phase boundary -> use last available
    last_key = list(phase_step_boundaries.keys())[-1]
    return int(phase_source_ranges[last_key][1])


def _lstm_window(arrays: ds.Stage1Arrays, end_sample: int) -> np.ndarray:
    """(400, 70) raw multimodal input window ending at end_sample.
    Pads with the leading sample if there is not enough leading context."""
    e = int(end_sample)
    s = e - ds.WINDOW_LEN
    if s < 0:
        # pad with the first sample
        n_pad = -s
        emg = np.concatenate([np.tile(arrays.emg[:1], (n_pad, 1)),
                              arrays.emg[:e]], axis=0)
        acc = np.concatenate([np.tile(arrays.acc[:1], (n_pad, 1)),
                              arrays.acc[:e]], axis=0)
        glove = np.concatenate([np.tile(arrays.glove[:1], (n_pad, 1)),
                                arrays.glove[:e]], axis=0)
    else:
        emg = arrays.emg[s:e]
        acc = arrays.acc[s:e]
        glove = arrays.glove[s:e]
    return np.concatenate([emg, acc, glove], axis=1).astype(
        np.float32, copy=False)


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------

_DINOV2_NAME: str = "facebook/dinov2-base"
_DINOV2_DIM: int = 768


def _load_dinov2(device: str = "cuda:0") -> tuple[AutoModel, AutoImageProcessor]:
    processor = AutoImageProcessor.from_pretrained(_DINOV2_NAME)
    model = AutoModel.from_pretrained(_DINOV2_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


@torch.no_grad()
def _dinov2_batch(model: AutoModel, processor: AutoImageProcessor,
                  frames: list[np.ndarray], device: str,
                  batch_size: int = 16) -> np.ndarray:
    """Run a list of HxWx3 uint8 frames through DINOv2; return (N, 768)."""
    out: list[np.ndarray] = []
    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size]
        pil = [Image.fromarray(f) for f in chunk]
        inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model(**inputs).last_hidden_state[:, 0]   # CLS token
        out.append(feats.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------

def _load_frames_for_sample(entry: dict[str, str]) -> list[np.ndarray]:
    """Load every frame for one dataset_index entry, regardless of source
    (PNG dir for stage4, MP4 for stage5)."""
    frames_path = pc.PROJECT_ROOT / entry["frames"]
    if entry["frames_kind"] == "png_dir":
        return [iio.imread(p) for p in sorted(Path(frames_path).glob("*.png"))]
    elif entry["frames_kind"] == "mp4":
        reader = iio.get_reader(str(frames_path))
        try:
            return [np.asarray(f) for f in reader]
        finally:
            reader.close()
    else:
        raise ValueError(f"unknown frames_kind: {entry['frames_kind']}")


def _build_robot_state_and_action(
    npz: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """robot_state: (T, 20) = joint_pos(6) + joint_vel(6) + ee_pos(3)
    + ee_quat(4) + gripper(1).  Stage 5 doesn't carry ee_quat; we fill
    with identity so the dim is identical for both stages.

    action: (T, 7) = joint_velocity(6) + gripper(1). We compute joint
    velocity as a first-difference of joint_positions (zero-pad the last
    step). Gripper is taken directly from gripper_state.
    """
    jp = np.asarray(npz["joint_positions"], dtype=np.float32)
    jv = np.asarray(npz["joint_velocities"], dtype=np.float32)
    ee = np.asarray(npz["end_effector_pos"], dtype=np.float32)
    if "end_effector_quat" in npz:
        eq = np.asarray(npz["end_effector_quat"], dtype=np.float32)
    else:
        eq = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                     (jp.shape[0], 1))
    gs = np.asarray(npz["gripper_state"], dtype=np.float32).reshape(-1, 1)
    state = np.concatenate([jp, jv, ee, eq, gs], axis=1)   # (T, 20)

    # action: first-difference of joint positions scaled by render_fps so
    # the units are rad/s, plus gripper command.
    dt = 1.0 / float(pc.RENDER_FPS)
    jv_cmd = np.zeros_like(jp)
    jv_cmd[:-1] = (jp[1:] - jp[:-1]) / dt
    action = np.concatenate([jv_cmd, gs], axis=1).astype(np.float32)
    return state.astype(np.float32), action


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage5" / "dataset_index.json"))
    ap.add_argument("--out-dir", default=str(
        pc.PROJECT_ROOT / "outputs" / "stage7" / "cached_features"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dino-batch", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="re-cache even if files exist")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    entries = index["entries"]
    print(f"=== Stage 7 — cache features for {len(entries)} samples ===",
          flush=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    print("[load] Stage 1 arrays + Stage 2 LSTM...", flush=True)
    arrays = ds.load_stage1_arrays(pc.STAGE1_DIR)
    lstm = load_lstm()
    lstm.to(device).eval()
    for p in lstm.parameters():
        p.requires_grad_(False)

    print(f"[load] DINOv2 {_DINOV2_NAME}...", flush=True)
    dino, dino_proc = _load_dinov2(device)

    t0 = time.time()
    for i, entry in enumerate(entries):
        sid = entry["id"]
        out_dino = out_dir / f"{sid}_dino.npy"
        out_lstm = out_dir / f"{sid}_lstm.npy"
        out_state = out_dir / f"{sid}_state.npy"
        out_action = out_dir / f"{sid}_action.npy"
        if (not args.force) and all(p.exists() for p in
                                    (out_dino, out_lstm, out_state, out_action)):
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(entries)}] {sid}  (cached, skip)",
                      flush=True)
            continue

        t_s = time.time()
        npz_path = pc.PROJECT_ROOT / entry["npz"]
        with np.load(npz_path, allow_pickle=False) as z:
            npz_dict = {k: z[k] for k in z.files}
        md = json.loads(str(npz_dict["metadata"]))

        T = int(np.asarray(npz_dict["joint_positions"]).shape[0])

        # ---- robot state + action ----
        state, action = _build_robot_state_and_action(npz_dict)
        np.save(out_state, state)
        np.save(out_action, action)

        # ---- LSTM hidden per timestep ----
        boundaries = (md.get("phase_frame_boundaries")
                      or md.get("emg_phase_step_boundaries"))
        source_ranges = (md.get("emg_phase_source_ranges")
                         or md.get("emg_phase_source_ranges"))
        if boundaries is None or source_ranges is None:
            print(f"  [warn] {sid} missing phase metadata, skipping",
                  flush=True)
            continue
        # Convert lists -> tuples for type hints; keep keys
        boundaries = {k: tuple(v) for k, v in boundaries.items()}
        source_ranges = {k: tuple(v) for k, v in source_ranges.items()}

        windows = np.empty((T, ds.WINDOW_LEN, ds.N_FEATURES),
                           dtype=np.float32)
        for t in range(T):
            end_sample = _phase_source_abs_window_for_frame(
                phase_step_boundaries=boundaries,
                phase_source_ranges=source_ranges,
                frame_idx=t, T_total=T)
            windows[t] = _lstm_window(arrays, end_sample)
        # batched LSTM hidden
        hidden_chunks: list[np.ndarray] = []
        bs = 64
        with torch.no_grad():
            for s in range(0, T, bs):
                x = torch.from_numpy(windows[s:s + bs]).to(device)
                h = lstm.get_hidden_state(x)
                hidden_chunks.append(h.cpu().numpy().astype(np.float32))
        lstm_hidden = np.concatenate(hidden_chunks, axis=0)
        assert lstm_hidden.shape == (T, 128), lstm_hidden.shape
        np.save(out_lstm, lstm_hidden)

        # ---- DINOv2 features per timestep ----
        frames = _load_frames_for_sample(entry)
        if len(frames) != T:
            # Stage 4 PNG dirs sometimes have one extra frame at end; align
            frames = frames[:T] if len(frames) > T else (
                frames + [frames[-1]] * (T - len(frames)))
        dino_feats = _dinov2_batch(
            dino, dino_proc, frames, device,
            batch_size=args.dino_batch)
        assert dino_feats.shape == (T, _DINOV2_DIM), dino_feats.shape
        np.save(out_dino, dino_feats)

        print(f"  [{i+1}/{len(entries)}] {sid}  "
              f"({T} frames, {time.time()-t_s:.1f}s)", flush=True)

    print(f"\n[stage7] feature cache done in {time.time()-t0:.1f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

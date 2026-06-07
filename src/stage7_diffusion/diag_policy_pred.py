"""Diagnostic: compare policy-predicted actions vs cached actions at the
very first observation of stage4_demo_000, and replay BOTH sequences
through the execute loop. Tells us if the policy itself is the problem
(predictions far from GT) or if the closed-loop execution drift kills
the policy (predictions close to GT but execution diverges)."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch

from ..stage3_sim import physics_constants as pc
from .dataset import load_cached_samples
from .execute import ExecutionSim, FrozenDINOv2, load_policy


def main() -> int:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cache_dir = pc.PROJECT_ROOT / "outputs" / "stage7" / "cached_features"
    samples = load_cached_samples(
        pc.PROJECT_ROOT / "outputs" / "stage5" / "dataset_index.json",
        cache_dir,
    )
    s = samples[0]    # stage4_demo_000
    print(f"target: {s.sample_id}", flush=True)

    # Build the exact observation at t=0 from cached features.
    dino0 = torch.from_numpy(np.asarray(s.dino[0], dtype=np.float32))[None].to(device)
    lstm0 = torch.from_numpy(np.asarray(s.lstm[0], dtype=np.float32))[None].to(device)
    state0 = torch.from_numpy(np.asarray(s.state[0], dtype=np.float32))[None].to(device)
    print(f"  state[0] = {state0.cpu().numpy().ravel()[:10]}...", flush=True)

    policy, normalizer = load_policy(
        pc.PROJECT_ROOT / "outputs" / "stage7" / "policy_emg_best.pt",
        device,
    )
    pred = policy.predict_action(
        dino_feat=dino0, state=state0, lstm_feat=lstm0,
        normalizer=normalizer,
    )[0].cpu().numpy()   # (H=16, 7)

    gt = np.asarray(s.action[:16], dtype=np.float32)
    print(f"\nGT action[:16, :6] mean abs = {np.abs(gt[:, :6]).mean():.4f}  "
          f"max = {np.abs(gt[:, :6]).max():.4f}", flush=True)
    print(f"PRED        mean abs = {np.abs(pred[:, :6]).mean():.4f}  "
          f"max = {np.abs(pred[:, :6]).max():.4f}", flush=True)
    print(f"L1 distance GT vs PRED (per-step, mean per joint):")
    for t in range(16):
        diff = np.abs(gt[t] - pred[t])
        print(f"  t={t:2d}  jv_diff={diff[:6].mean():.4f}  "
              f"grip_diff={diff[6]:.3f}  "
              f"gt_grip={gt[t,6]:.2f}  pred_grip={pred[t,6]:.2f}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

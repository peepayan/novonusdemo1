"""Diffusion-policy wrapper: encoder + DDPM-trained denoiser + DDIM sampler.

Composition:
  observation -> MultimodalObsEncoder -> conditioning vector (960 dims)
  noisy_action + step + cond -> ConditionalUnet1D -> predicted noise
At inference, DDIM (5 steps by default) inverts the diffusion process to
produce a clean 16-step action sequence in normalized action space.
The action normalizer (from the training set) is stored in the
checkpoint so the policy can de-normalize at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler

from .conditional_unet1d import ConditionalUnet1D
from .dataset import (
    ACTION_DIM, ACTION_HORIZON, ActionNormalizer,
)
from .obs_encoder import (
    EMG_COND_DIM, MultimodalObsEncoder, VISION_ONLY_COND_DIM,
)


DDPM_TRAIN_STEPS: int = 100
DDIM_INFERENCE_STEPS: int = 5


@dataclass
class PolicyConfig:
    include_emg: bool = True
    action_horizon: int = ACTION_HORIZON
    action_dim: int = ACTION_DIM
    ddpm_steps: int = DDPM_TRAIN_STEPS
    ddim_steps: int = DDIM_INFERENCE_STEPS
    unet_down_dims: tuple[int, ...] = (256, 512, 1024)


class DiffusionPolicy(nn.Module):
    """Full policy: observation encoder + diffusion denoiser."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MultimodalObsEncoder(include_emg=cfg.include_emg)
        cond_dim = (EMG_COND_DIM if cfg.include_emg
                    else VISION_ONLY_COND_DIM)
        self.denoiser = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=cond_dim,
            down_dims=cfg.unet_down_dims,
        )
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.ddpm_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon",
        )
        self.infer_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.ddpm_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True, set_alpha_to_one=True,
            steps_offset=0, prediction_type="epsilon",
        )

    # --- training loss ---------------------------------------------------

    def compute_loss(
        self, *,
        dino_feat: torch.Tensor,        # (B, 768)
        state: torch.Tensor,            # (B, 20)
        action_norm: torch.Tensor,      # (B, H, 7)  in normalized space
        lstm_feat: torch.Tensor | None = None,   # (B, 128)
    ) -> torch.Tensor:
        B = action_norm.shape[0]
        cond = self.encoder(
            dino_feat=dino_feat, state=state, lstm_feat=lstm_feat)
        # sample timestep + noise; predict the noise
        timesteps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps, (B,),
            device=action_norm.device, dtype=torch.long)
        noise = torch.randn_like(action_norm)
        noisy = self.train_scheduler.add_noise(
            action_norm, noise, timesteps)
        pred = self.denoiser(noisy, timesteps, global_cond=cond)
        return nn.functional.mse_loss(pred, noise)

    # --- inference -------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self, *,
        dino_feat: torch.Tensor,        # (B, 768)
        state: torch.Tensor,            # (B, 20)
        lstm_feat: torch.Tensor | None = None,   # (B, 128)
        normalizer: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Returns (B, H, action_dim) action sequence in **real units**.
        ``normalizer`` is the dict from ``ActionNormalizer.to(device)``."""
        B = dino_feat.shape[0]
        cond = self.encoder(
            dino_feat=dino_feat, state=state, lstm_feat=lstm_feat)
        x = torch.randn(
            B, self.cfg.action_horizon, self.cfg.action_dim,
            device=dino_feat.device)
        self.infer_scheduler.set_timesteps(self.cfg.ddim_steps)
        for t in self.infer_scheduler.timesteps:
            pred = self.denoiser(x, t, global_cond=cond)
            x = self.infer_scheduler.step(pred, t, x).prev_sample
        if normalizer is not None:
            x = x * normalizer["std"] + normalizer["mean"]
        return x


__all__ = [
    "PolicyConfig", "DiffusionPolicy",
    "DDPM_TRAIN_STEPS", "DDIM_INFERENCE_STEPS",
]

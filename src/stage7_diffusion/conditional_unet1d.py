"""1-D Conditional U-Net denoiser with FiLM conditioning.

Adapted directly from Chi et al., "Diffusion Policy: Visuomotor Policy
Learning via Action Diffusion" — specifically the CNN denoiser used by
the Stanford reference implementation
(``diffusion_policy.model.diffusion.conditional_unet1d``). Their code is
released under MIT (Stanford ARMLab); see
https://github.com/real-stanford/diffusion_policy.

Vendored here (rather than imported from the upstream package) because
the upstream repo's Hydra/robomimic/mujoco-py dependency stack does not
install cleanly on Windows + Python 3.13. The denoiser itself is a small
self-contained module — Conv1dBlock + Conditional residual + U-Net wrap —
so vendoring is the simplest path.
"""

from __future__ import annotations

import math
from typing import Union

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Standard transformer-style sinusoidal positional embedding.
    Used to encode the diffusion timestep into a feature vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        emb = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """FiLM-conditioned residual block: two Conv1dBlocks with a per-channel
    scale+bias derived from the conditioning vector inserted between them.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 cond_dim: int, kernel_size: int = 3, n_groups: int = 8,
                 cond_predict_scale: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('b c -> b c 1'),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# ---------------------------------------------------------------------------
# The U-Net denoiser
# ---------------------------------------------------------------------------

class ConditionalUnet1D(nn.Module):
    """1-D Conditional U-Net for diffusion-policy action denoising.

    Inputs (forward):
      sample:    (B, action_horizon, action_dim) noisy action sequence
      timestep:  scalar or (B,) diffusion step index
      global_cond: (B, cond_dim) fused observation conditioning vector
    Output:
      (B, action_horizon, action_dim) predicted noise
    """

    def __init__(self,
                 input_dim: int,
                 global_cond_dim: int,
                 diffusion_step_embed_dim: int = 128,
                 down_dims: tuple[int, ...] = (256, 512, 1024),
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 cond_predict_scale: bool = True):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                       kernel_size, n_groups,
                                       cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                       kernel_size, n_groups,
                                       cond_predict_scale),
        ])

        self.down_modules = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim,
                                           kernel_size, n_groups,
                                           cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim,
                                           kernel_size, n_groups,
                                           cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = i >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim,
                                           kernel_size, n_groups,
                                           cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim,
                                           kernel_size, n_groups,
                                           cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size,
                        n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, int, float],
                global_cond: torch.Tensor) -> torch.Tensor:
        # sample: (B, T, D) -> work in (B, D, T)
        sample = einops.rearrange(sample, 'b t d -> b d t')

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        gfeat = self.diffusion_step_encoder(timesteps)
        gfeat = torch.cat([gfeat, global_cond], dim=-1)

        x = sample
        h: list[torch.Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, gfeat)
            x = resnet2(x, gfeat)
            h.append(x)
            x = downsample(x)

        for m in self.mid_modules:
            x = m(x, gfeat)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, gfeat)
            x = resnet2(x, gfeat)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, 'b d t -> b t d')


__all__ = ["ConditionalUnet1D"]

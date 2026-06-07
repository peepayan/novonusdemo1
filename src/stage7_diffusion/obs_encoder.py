"""Multimodal observation encoder for the diffusion policy.

At *training* time, this module consumes cached DINOv2 + LSTM features
(produced by ``cache_features.py``) along with the live robot-state
vector, and fuses them into a single 960-dim conditioning vector that
the diffusion denoiser uses for FiLM conditioning.

At *inference* time, ``cache_features.RealtimeObsEncoder`` (used by
``execute.py``) wraps this with the real DINOv2 + LSTM modules so the
same fusion happens on raw observations.

Trainable parameters:
- ``robot_state_mlp`` (~5 k params) — the only thing trained here.
- The diffusion denoiser trains its own params separately; this encoder
  is only the conditioning-vector builder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dataset import (
    DINO_DIM, LSTM_DIM, ROBOT_STATE_DIM, ROBOT_EMBED_DIM,
)


# Final fused conditioning size: 768 + 128 + 64 = 960
EMG_COND_DIM: int = DINO_DIM + LSTM_DIM + ROBOT_EMBED_DIM
VISION_ONLY_COND_DIM: int = DINO_DIM + ROBOT_EMBED_DIM


class RobotStateMLP(nn.Module):
    """64-dim embedding of the 20-dim robot state."""

    def __init__(self, in_dim: int = ROBOT_STATE_DIM,
                 out_dim: int = ROBOT_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, ROBOT_EMBED_DIM), nn.ReLU(),
            nn.Linear(ROBOT_EMBED_DIM, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalObsEncoder(nn.Module):
    """Fuses DINOv2 + LSTM + robot-state into the conditioning vector.

    Both ``dino_feat`` and ``lstm_feat`` are passed in already-computed
    (from cache or from the realtime wrapper); only the robot-state MLP
    is trained inside this module.
    """

    def __init__(self, include_emg: bool = True):
        super().__init__()
        self.include_emg = include_emg
        self.robot_mlp = RobotStateMLP()
        self.out_dim = (EMG_COND_DIM if include_emg
                        else VISION_ONLY_COND_DIM)

    def forward(self, *,
                dino_feat: torch.Tensor,
                state: torch.Tensor,
                lstm_feat: torch.Tensor | None = None,
                ) -> torch.Tensor:
        robot_emb = self.robot_mlp(state)
        if self.include_emg:
            assert lstm_feat is not None, "EMG-conditioned encoder needs lstm_feat"
            return torch.cat([dino_feat, lstm_feat, robot_emb], dim=-1)
        return torch.cat([dino_feat, robot_emb], dim=-1)


__all__ = [
    "EMG_COND_DIM", "VISION_ONLY_COND_DIM",
    "RobotStateMLP", "MultimodalObsEncoder",
]

"""Dedicated EMG-to-force regressor.

Backbone : 2-layer LSTM, hidden 128, dropout 0.2, batch_first
Optional : concat a per-window standardized rich-feature vector to the
           shared 128-dim hidden state -> MLP head 128(+F) -> 64 -> 1 -> sigmoid.

`use_features=False` -> "envelope only" baseline (no rich features), used
to demonstrate the gain from richer features.

Hidden state is exposed for downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ForceOutput:
    force: torch.Tensor    # (B,) in [0,1]
    hidden: torch.Tensor   # (B, 128)


class E3ForceLSTM(nn.Module):
    def __init__(self, n_lstm_features: int = 48,
                 n_rich_features: int = 0,
                 hidden_size: int = 128,
                 n_layers: int = 2,
                 lstm_dropout: float = 0.2,
                 head_hidden: int = 64):
        super().__init__()
        self.n_lstm_features = n_lstm_features
        self.n_rich_features = n_rich_features
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=n_lstm_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=lstm_dropout if n_layers > 1 else 0.0,
        )

        head_in = hidden_size + n_rich_features
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(head_hidden, 1),
            nn.Sigmoid(),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[2] != self.n_lstm_features:
            raise ValueError(
                f"expected (B, T, {self.n_lstm_features}), got {tuple(x.shape)}"
            )
        outputs, (_h, _c) = self.lstm(x)
        return outputs[:, -1, :]

    def get_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)

    def forward(self, x: torch.Tensor,
                feats: torch.Tensor | None = None) -> ForceOutput:
        h = self._encode(x)
        if self.n_rich_features > 0:
            if feats is None or feats.numel() == 0:
                raise ValueError("rich features expected but not provided")
            if feats.shape[1] != self.n_rich_features:
                raise ValueError(
                    f"feats has {feats.shape[1]} dims, expected {self.n_rich_features}"
                )
            joint = torch.cat([h, feats], dim=1)
        else:
            joint = h
        force = self.head(joint).squeeze(-1)
        return ForceOutput(force=force, hidden=h)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

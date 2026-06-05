"""Multimodal LSTM with a shared backbone, an intent-classification head, and
a force-intensity regression head.

Architecture
------------
input (B, T=400, F=70)
  |
  +-- LSTM 2-layer, hidden=128, dropout=0.2 between layers, batch_first
  |
  hidden state h  (B, 128)     <-- last-timestep top-layer output
  |       |
  |       +-- MLP 128 -> 64 -> 1, sigmoid  -->  force scalar in [0, 1]
  |                                              (a *learned readout* of the
  |                                               muscle-activation encoding)
  |
  +-- Linear 128 -> n_classes  --> intent logits

Important notes
---------------
- ``get_hidden_state(x)`` returns the 128-dim vector cleanly so Stage 7
  (Diffusion Policy) can consume it as a conditioning feature.
- The force head branches off the SAME shared hidden state as the classifier,
  not off the raw input. Since the LSTM is driven mostly by EMG (12 of the 70
  input channels are MVC-normalised EMG envelope), the force prediction is a
  learned readout of the EMG-shaped representation. It is *not* a measured
  force — it is a learned estimate of an EMG-amplitude force proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .class_mapping import N_INTENT_CLASSES


@dataclass
class ModelOutput:
    logits: torch.Tensor      # (B, K)  raw class logits for the loss
    probs: torch.Tensor       # (B, K)  softmax probabilities for display
    force: torch.Tensor       # (B,)    force-intensity scalar in [0, 1]
    hidden: torch.Tensor      # (B, H)  shared 128-dim representation


class IntentForceLSTM(nn.Module):
    def __init__(self, n_features: int = 70, hidden_size: int = 128,
                 n_layers: int = 2, lstm_dropout: float = 0.2,
                 n_classes: int = N_INTENT_CLASSES,
                 force_hidden: int = 64):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        # backbone shared between both heads
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=lstm_dropout if n_layers > 1 else 0.0,
        )

        # intent-classification head
        self.classifier = nn.Linear(hidden_size, n_classes)

        # force-intensity regression head
        # Reads the same shared hidden state and projects to a scalar in [0,1].
        # This makes the predicted force a *learned function of the
        # EMG-driven representation*, by construction.
        self.force_head = nn.Sequential(
            nn.Linear(hidden_size, force_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(force_hidden, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LSTM and return the last-timestep top-layer hidden state."""
        if x.dim() != 3 or x.shape[2] != self.n_features:
            raise ValueError(
                f"expected (B, T, {self.n_features}), got {tuple(x.shape)}"
            )
        # outputs: (B, T, H), h_n: (n_layers, B, H)
        outputs, (h_n, _c_n) = self.lstm(x)
        # Use the actual last timestep of the top layer's output, which equals
        # h_n[-1] for batch_first sequences without packing — explicit for clarity.
        return outputs[:, -1, :]

    def get_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Public hook for Stage 7: returns the shared 128-dim representation."""
        return self._encode(x)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self._encode(x)
        logits = self.classifier(h)
        probs = torch.softmax(logits, dim=-1)
        force = self.force_head(h).squeeze(-1)
        return ModelOutput(logits=logits, probs=probs, force=force, hidden=h)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

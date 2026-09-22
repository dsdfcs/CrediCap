from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@dataclass
class FeatureFusionOutput:
    score: torch.Tensor
    correction: torch.Tensor
    gate: torch.Tensor
    raw_direction: torch.Tensor


class FeatureFusionCalibrator(nn.Module):
    """A bounded correction head trained outside the evaluation benchmark.

    The no-feedback and feedback branches use this exact same architecture.
    The only experimental difference is whether the 8-D feedback vector is
    real or identically zero.  That makes the feedback ablation interpretable.
    """

    def __init__(
        self,
        base_dim: int,
        feedback_dim: int,
        hidden_dim: int,
        dropout: float,
        maximum_correction: float,
    ) -> None:
        super().__init__()
        self.base_dim = int(base_dim)
        self.feedback_dim = int(feedback_dim)
        self.hidden_dim = int(hidden_dim)
        self.maximum_correction = float(maximum_correction)

        self.base_encoder = nn.Sequential(
            nn.LayerNorm(base_dim),
            nn.Linear(base_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.feedback_encoder = nn.Sequential(
            nn.Linear(feedback_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Multiplication exposes sample-specific agreement between the frozen
        # M1/M2 diagnosis and the structured feedback diagnosis.
        fusion_dim = hidden_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.direction = nn.Linear(hidden_dim, 1)
        self.reliability = nn.Linear(hidden_dim, 1)
        zero_linear(self.direction)
        zero_linear(self.reliability)
        nn.init.constant_(self.reliability.bias, -1.5)

    def forward(
        self,
        anchor: torch.Tensor,
        base_features: torch.Tensor,
        feedback_features: torch.Tensor,
    ) -> FeatureFusionOutput:
        base = self.base_encoder(base_features.float())
        feedback = self.feedback_encoder(feedback_features.float())
        fused = self.fusion(
            torch.cat(
                [base, feedback, base * feedback, (base - feedback).abs()],
                dim=-1,
            )
        )
        raw_direction = torch.tanh(self.direction(fused).squeeze(-1))
        gate = torch.sigmoid(self.reliability(fused).squeeze(-1))
        correction = self.maximum_correction * gate * raw_direction
        score = (anchor.float() + correction).clamp(0.0, 1.0)
        return FeatureFusionOutput(
            score=score,
            correction=correction,
            gate=gate,
            raw_direction=raw_direction,
        )


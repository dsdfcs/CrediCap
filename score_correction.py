from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


BASE_DIM = 21
FEEDBACK_DIM = 7


@dataclass
class ScoreCorrectionOutput:
    correction: torch.Tensor
    direction_logits: torch.Tensor
    direction_value: torch.Tensor
    magnitude: torch.Tensor


@dataclass
class ScoreCorrectionResult:
    base: ScoreCorrectionOutput
    feedback: ScoreCorrectionOutput


class DirectionMagnitudeHead(nn.Module):
    """Turn a hidden state into a bounded signed correction.

    Direction classes are LOWER, KEEP and HIGHER.  The continuous direction is
    P(HIGHER)-P(LOWER), so the model may correct either an overestimate or an
    underestimate.  Magnitude is learned separately to avoid forcing every
    directional decision to use the full correction budget.
    """

    def __init__(self, hidden_dim: int, maximum_correction: float) -> None:
        super().__init__()
        if maximum_correction <= 0.0:
            raise ValueError("maximum_correction must be positive")
        self.maximum_correction = float(maximum_correction)
        self.direction = nn.Linear(hidden_dim, 3)
        self.magnitude_layer = nn.Linear(hidden_dim, 1)
        # Small non-zero weights preserve an almost-identity start while still
        # connecting every input field to the output during the self-check.
        nn.init.xavier_uniform_(self.direction.weight, gain=0.01)
        nn.init.zeros_(self.direction.bias)
        nn.init.xavier_uniform_(self.magnitude_layer.weight, gain=0.01)
        nn.init.constant_(self.magnitude_layer.bias, -1.0)

    def forward(self, hidden: torch.Tensor) -> ScoreCorrectionOutput:
        logits = self.direction(hidden)
        probability = torch.softmax(logits, dim=-1)
        direction_value = probability[:, 2] - probability[:, 0]
        magnitude = torch.sigmoid(self.magnitude_layer(hidden).squeeze(-1))
        correction = self.maximum_correction * direction_value * magnitude
        return ScoreCorrectionOutput(
            correction=correction,
            direction_logits=logits,
            direction_value=direction_value,
            magnitude=magnitude,
        )


class DirectionMagnitudeCorrector(nn.Module):
    """Two-stage signed error correction with an explicit feedback residual.

    The base branch sees only the frozen M1+CDED-M2 context.  The feedback
    branch sees the same context plus every actual feedback field, its expected
    value, the innovation and its magnitude.  The final feedback contribution
    is a signed residual; it is not tied to the old v8 correction direction.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        maximum_base_correction: float = 0.050,
        maximum_feedback_correction: float = 0.050,
    ) -> None:
        super().__init__()
        if hidden_dim < 32 or hidden_dim % 2:
            raise ValueError("hidden_dim must be an even integer >= 32")
        token_dim = hidden_dim // 2
        self.base_encoder = nn.Sequential(
            nn.LayerNorm(BASE_DIM),
            nn.Linear(BASE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.base_head = DirectionMagnitudeHead(
            hidden_dim, maximum_base_correction
        )

        self.field_identity = nn.Parameter(
            torch.empty(FEEDBACK_DIM, token_dim)
        )
        nn.init.normal_(self.field_identity, mean=0.0, std=0.02)
        self.field_encoder = nn.Sequential(
            nn.Linear(4, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.field_mixer = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )
        self.feedback_pool = nn.Sequential(
            nn.LayerNorm(token_dim * 3),
            nn.Linear(token_dim * 3, hidden_dim),
            nn.GELU(),
        )
        # Context, feedback, their interaction and their absolute difference.
        self.feedback_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.feedback_head = DirectionMagnitudeHead(
            hidden_dim, maximum_feedback_correction
        )

    @staticmethod
    def validate_inputs(
        base_features: torch.Tensor,
        actual_feedback: torch.Tensor,
        expected_feedback: torch.Tensor,
        anchor: torch.Tensor,
    ) -> None:
        count = base_features.shape[0]
        if base_features.shape != (count, BASE_DIM):
            raise ValueError(
                f"Expected base features [N,{BASE_DIM}], "
                f"got {tuple(base_features.shape)}"
            )
        if actual_feedback.shape != (count, FEEDBACK_DIM):
            raise ValueError(
                f"Expected actual feedback [N,{FEEDBACK_DIM}], "
                f"got {tuple(actual_feedback.shape)}"
            )
        if expected_feedback.shape != (count, FEEDBACK_DIM):
            raise ValueError(
                f"Expected expected feedback [N,{FEEDBACK_DIM}], "
                f"got {tuple(expected_feedback.shape)}"
            )
        if anchor.shape != (count,):
            raise ValueError(f"Expected anchor [N], got {tuple(anchor.shape)}")

    def forward_base(
        self,
        base_features: torch.Tensor,
    ) -> tuple[torch.Tensor, ScoreCorrectionOutput]:
        context = self.base_encoder(base_features.float())
        return context, self.base_head(context)

    def forward_feedback(
        self,
        context: torch.Tensor,
        base_output: ScoreCorrectionOutput,
        actual_feedback: torch.Tensor,
        expected_feedback: torch.Tensor,
        anchor: torch.Tensor,
    ) -> ScoreCorrectionOutput:
        actual = actual_feedback.float()
        expected = expected_feedback.float()
        innovation = actual - expected
        field_input = torch.stack(
            [actual, expected, innovation, innovation.abs()], dim=-1
        )
        tokens = self.field_encoder(field_input)
        tokens = self.field_mixer(tokens + self.field_identity.unsqueeze(0))
        feedback_state = self.feedback_pool(
            torch.cat(
                [
                    tokens.mean(dim=1),
                    tokens.amax(dim=1),
                    tokens.amin(dim=1),
                ],
                dim=1,
            )
        )
        scalar_context = torch.stack(
            [
                anchor.float(),
                base_output.correction,
                base_output.direction_value,
                base_output.magnitude,
            ],
            dim=1,
        )
        fused = self.feedback_fusion(
            torch.cat(
                [
                    context,
                    feedback_state,
                    context * feedback_state,
                    (context - feedback_state).abs(),
                    scalar_context,
                ],
                dim=1,
            )
        )
        return self.feedback_head(fused)

    def forward(
        self,
        base_features: torch.Tensor,
        actual_feedback: torch.Tensor,
        expected_feedback: torch.Tensor,
        anchor: torch.Tensor,
    ) -> ScoreCorrectionResult:
        self.validate_inputs(
            base_features,
            actual_feedback,
            expected_feedback,
            anchor,
        )
        context, base_output = self.forward_base(base_features)
        feedback_output = self.forward_feedback(
            context,
            base_output,
            actual_feedback,
            expected_feedback,
            anchor,
        )
        return ScoreCorrectionResult(base=base_output, feedback=feedback_output)

    def base_parameters(self):
        yield from self.base_encoder.parameters()
        yield from self.base_head.parameters()

    def feedback_parameters(self):
        yield self.field_identity
        yield from self.field_encoder.parameters()
        yield from self.field_mixer.parameters()
        yield from self.feedback_pool.parameters()
        yield from self.feedback_fusion.parameters()
        yield from self.feedback_head.parameters()

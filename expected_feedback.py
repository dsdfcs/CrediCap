from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class FeedbackExpectationNet(nn.Module):
    """Predict what Qwen feedback M1/M2 already makes unsurprising."""

    def __init__(self, base_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(base_dim),
            nn.Linear(base_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, normalized_base: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(normalized_base.float()))


@dataclass
class ExpectedFeedbackOutput:
    score: torch.Tensor
    correction: torch.Tensor
    innovation_delta: torch.Tensor
    gate: torch.Tensor
    novelty: torch.Tensor
    agreement: torch.Tensor


class ExpectedFeedbackAdapter(nn.Module):
    """Score only feedback information not predictable from frozen M1/M2.

    Let e = actual_feedback - expected_feedback(M1/M2).  Per-sample centering
    first removes a Qwen severity/style shift shared by all seven fields.  The
    score direction is then 0.5 * (V(h,e) - V(h,-e)).  This antisymmetric
    construction is exactly zero when feedback supplies no conditional
    innovation and blocks a feedback-independent Polaris calibration shift.
    """

    STRUCTURAL_STRENGTH = 4
    CONSENSUS_SUPPORT = 5
    DISSENT_SUPPORT = 6
    CONSENSUS_DISSENT_GAP = 7
    SUPPORT_DISPERSION = 8
    REFERENCE_DISAGREEMENT = 9

    def __init__(
        self,
        base_dim: int,
        hidden_dim: int,
        maximum_correction: float,
    ) -> None:
        super().__init__()
        self.maximum_correction = float(maximum_correction)
        self.base_encoder = nn.Sequential(
            nn.LayerNorm(base_dim),
            nn.Linear(base_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.innovation_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.odd_value = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(18, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.xavier_uniform_(self.odd_value[-1].weight, gain=0.02)
        nn.init.zeros_(self.odd_value[-1].bias)
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, -1.0)

    @staticmethod
    def _joint(base: torch.Tensor, innovation: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                base,
                innovation,
                base * innovation,
                (base - innovation).abs(),
            ],
            dim=1,
        )

    @staticmethod
    def _feedback_summary(values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        positive = values[:, 0:4].mean(dim=1)
        negative = values[:, 4:6].mean(dim=1)
        signed = positive - negative
        certainty = 1.0 - values[:, 6].clamp(0.0, 1.0)
        return positive, negative, signed, certainty

    def _gate_features(
        self,
        raw_base: torch.Tensor,
        actual: torch.Tensor,
        expected: torch.Tensor,
        innovation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_positive, actual_negative, actual_signed, certainty = (
            self._feedback_summary(actual)
        )
        expected_positive, expected_negative, expected_signed, _ = (
            self._feedback_summary(expected)
        )
        innovation_positive = innovation[:, 0:4].mean(dim=1)
        innovation_negative = innovation[:, 4:6].mean(dim=1)
        innovation_signed = innovation_positive - innovation_negative
        novelty = innovation.abs().mean(dim=1)

        structural = torch.tanh(raw_base[:, self.STRUCTURAL_STRENGTH])
        consensus = torch.tanh(raw_base[:, self.CONSENSUS_SUPPORT])
        dissent = torch.tanh(raw_base[:, self.DISSENT_SUPPORT])
        gap = torch.tanh(raw_base[:, self.CONSENSUS_DISSENT_GAP])
        dispersion = torch.tanh(
            raw_base[:, self.SUPPORT_DISPERSION].abs()
        )
        disagreement = torch.tanh(
            raw_base[:, self.REFERENCE_DISAGREEMENT].abs()
        )
        signed_m12 = torch.tanh(consensus - dissent + 0.5 * gap)
        agreement = torch.exp(-1.5 * (actual_signed - signed_m12).abs())
        expected_agreement = torch.exp(
            -1.5 * (expected_signed - signed_m12).abs()
        )
        novelty_reliability = torch.exp(
            -3.0 * torch.relu(novelty - 0.45)
        )

        features = torch.stack(
            [
                actual_positive,
                actual_negative,
                actual_signed,
                certainty,
                expected_positive,
                expected_negative,
                expected_signed,
                innovation_positive,
                innovation_negative,
                innovation_signed,
                novelty,
                structural,
                consensus,
                dissent,
                gap,
                disagreement,
                agreement,
                expected_agreement,
            ],
            dim=1,
        )
        evidence_confidence = torch.sigmoid(
            2.0 * structural - dispersion - disagreement
        )
        prior = (
            0.10
            + 0.25 * certainty
            + 0.30 * agreement
            + 0.20 * evidence_confidence
            + 0.15 * novelty_reliability
        ).clamp(0.0, 1.0)
        return features, prior, novelty, agreement

    def forward(
        self,
        anchor: torch.Tensor,
        normalized_base: torch.Tensor,
        raw_base: torch.Tensor,
        feedback: torch.Tensor,
        expected_feedback: torch.Tensor,
    ) -> ExpectedFeedbackOutput:
        anchor = anchor.float()
        normalized_base = normalized_base.float()
        raw_base = raw_base.float()
        feedback = feedback.float()
        expected_feedback = expected_feedback.float().clamp(0.0, 1.0)
        valid = feedback[:, 7].clamp(0.0, 1.0)
        actual = feedback[:, :7]
        innovation = (actual - expected_feedback) * valid[:, None]
        innovation = innovation - innovation.mean(dim=1, keepdim=True)

        base = self.base_encoder(normalized_base)
        positive = self.innovation_encoder(innovation)
        negative = self.innovation_encoder(-innovation)
        positive_value = self.odd_value(self._joint(base, positive)).squeeze(-1)
        negative_value = self.odd_value(self._joint(base, negative)).squeeze(-1)
        innovation_delta = torch.tanh(
            0.5 * (positive_value - negative_value)
        )

        gate_features, prior, novelty, agreement = self._gate_features(
            raw_base,
            actual,
            expected_feedback,
            innovation,
        )
        learned_gate = torch.sigmoid(
            self.gate_network(gate_features).squeeze(-1)
        )
        gate = valid * prior * learned_gate
        correction = self.maximum_correction * gate * innovation_delta
        score = (anchor + correction).clamp(0.0, 1.0)
        return ExpectedFeedbackOutput(
            score=score,
            correction=correction,
            innovation_delta=innovation_delta,
            gate=gate,
            novelty=novelty,
            agreement=agreement,
        )

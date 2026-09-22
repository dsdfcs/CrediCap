from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def safe_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1.0e-4, 1.0 - 1.0e-4))


@dataclass
class ReferenceCredibilityOutput:
    score: torch.Tensor
    stage1_score: torch.Tensor
    reference_weights: torch.Tensor
    router_weights: torch.Tensor


class ReferenceTrustModule(nn.Module):
    """M1-A: estimate trust for every reference and aggregate trusted evidence."""

    FEATURE_DIM = 13

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.shared_projection.weight)
        self.trust = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.state = nn.Sequential(
            nn.Linear(hidden_dim * 3 + self.FEATURE_DIM, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta = nn.Linear(hidden_dim, 1)
        zero_linear(self.delta)

    @staticmethod
    def masked_stats(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = mask.to(values.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (values * weights).sum(dim=1) / count
        variance = (
            ((values - mean[:, None]) ** 2) * weights
        ).sum(dim=1) / count
        minimum = values.masked_fill(~mask, torch.inf).min(dim=1).values
        maximum = values.masked_fill(~mask, -torch.inf).max(dim=1).values
        minimum = torch.where(torch.isfinite(minimum), minimum, mean)
        maximum = torch.where(torch.isfinite(maximum), maximum, mean)
        return mean, variance.sqrt(), minimum, maximum

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = F.normalize(self.shared_projection(batch["image"]), dim=-1)
        candidate = F.normalize(
            self.shared_projection(batch["candidate"]), dim=-1
        )
        references = F.normalize(
            self.shared_projection(batch["references"]), dim=-1
        )
        mask = batch["reference_mask"].bool()

        image_reference = torch.einsum("bd,brd->br", image, references)
        candidate_reference = torch.einsum(
            "bd,brd->br", candidate, references
        )
        pairwise_reference = torch.einsum(
            "brd,bsd->brs", references, references
        )
        pair_mask = mask[:, :, None] & mask[:, None, :]
        diagonal = torch.eye(
            pairwise_reference.shape[1],
            dtype=torch.bool,
            device=pairwise_reference.device,
        )[None]
        peer_mask = pair_mask & ~diagonal
        peer_count = peer_mask.to(pairwise_reference.dtype).sum(dim=2)
        peer_count = peer_count.clamp_min(1.0)
        peer_consensus = (
            pairwise_reference.masked_fill(~peer_mask, 0.0).sum(dim=2)
            / peer_count
        )

        trust_input = torch.stack(
            [image_reference, peer_consensus, candidate_reference], dim=-1
        )
        trust_logits = self.trust(trust_input).squeeze(-1)
        trust_logits = trust_logits.masked_fill(~mask, -1.0e4)
        reference_weights = torch.softmax(trust_logits, dim=1)
        audited_reference = torch.einsum(
            "br,brd->bd", reference_weights, references
        )

        image_candidate = (image * candidate).sum(dim=-1)
        audited_candidate = (audited_reference * candidate).sum(dim=-1)
        audited_image = (audited_reference * image).sum(dim=-1)
        cr_mean, cr_std, cr_min, cr_max = self.masked_stats(
            candidate_reference, mask
        )
        ir_mean, ir_std, ir_min, ir_max = self.masked_stats(
            image_reference, mask
        )
        base = batch["baseline"].clamp(1.0e-4, 1.0 - 1.0e-4)
        scalar = torch.stack(
            [
                base,
                image_candidate,
                audited_candidate,
                audited_image,
                cr_mean,
                cr_std,
                cr_min,
                cr_max,
                ir_mean,
                ir_std,
                ir_min,
                ir_max,
                batch["candidate_length"],
            ],
            dim=-1,
        )
        interaction = torch.cat(
            [image, candidate, audited_reference], dim=-1
        )
        hidden = self.state(torch.cat([interaction, scalar], dim=-1))
        correction = torch.tanh(self.delta(hidden).squeeze(-1))
        score = torch.sigmoid(safe_logit(base) + correction)
        return score, reference_weights, hidden, scalar


class MultiEvidenceRouter(nn.Module):
    """M1-B: combine RefFLEUR, visual, reference and interaction evidence."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        scalar_dim = ReferenceTrustModule.FEATURE_DIM
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim + scalar_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.interaction_score = nn.Linear(hidden_dim, 1)
        self.image_scale = nn.Parameter(torch.tensor(6.0))
        self.image_bias = nn.Parameter(torch.tensor(-1.5))
        self.reference_scale = nn.Parameter(torch.tensor(6.0))
        self.reference_bias = nn.Parameter(torch.tensor(-1.5))
        self.route_gain = nn.Parameter(torch.tensor(0.0))
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + scalar_dim + 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim + scalar_dim + 8 + 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        zero_linear(self.interaction_score)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.router[-1].bias.data[1] = 2.0
        zero_linear(self.residual[-1])

    def forward(
        self,
        baseline: torch.Tensor,
        stage1_score: torch.Tensor,
        stage1_hidden: torch.Tensor,
        scalar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.interaction(torch.cat([stage1_hidden, scalar], dim=-1))
        learned_score = torch.sigmoid(
            self.interaction_score(hidden).squeeze(-1)
        )
        image_score = torch.sigmoid(
            self.image_scale * scalar[:, 1] + self.image_bias
        )
        reference_score = torch.sigmoid(
            self.reference_scale * scalar[:, 2] + self.reference_bias
        )
        experts = torch.stack(
            [
                baseline,
                stage1_score,
                image_score,
                reference_score,
                learned_score,
            ],
            dim=-1,
        )
        disagreement = torch.stack(
            [
                experts.mean(dim=-1),
                experts.std(dim=-1, unbiased=False),
                experts.min(dim=-1).values,
                experts.max(dim=-1).values,
                (stage1_score - image_score).abs(),
                (stage1_score - reference_score).abs(),
                (image_score - reference_score).abs(),
                (stage1_score - learned_score).abs(),
            ],
            dim=-1,
        )
        router_input = torch.cat([hidden, scalar, disagreement], dim=-1)
        router_weights = torch.softmax(self.router(router_input), dim=-1)
        routed_logit = (
            router_weights * safe_logit(experts)
        ).sum(dim=-1)
        residual_input = torch.cat([router_input, experts], dim=-1)
        residual = 0.75 * torch.tanh(
            self.residual(residual_input).squeeze(-1)
        )
        routed_difference = torch.tanh(
            routed_logit - safe_logit(stage1_score)
        )
        evidence_strength = torch.sigmoid(disagreement[:, 1] * 4.0) - 0.5
        correction = (
            residual
            + torch.tanh(self.route_gain)
            * evidence_strength
            * routed_difference
        )
        score = torch.sigmoid(safe_logit(stage1_score) + correction)
        return score, router_weights, hidden


class ReferenceCredibilityModel(nn.Module):
    """Final M1 only. No Stage-3 class or parameter exists in this model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.reference_trust = ReferenceTrustModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.evidence_router = MultiEvidenceRouter(hidden_dim, dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> ReferenceCredibilityOutput:
        stage1_score, weights, hidden, scalar = self.reference_trust(batch)
        score, router, _ = self.evidence_router(
            batch["baseline"], stage1_score, hidden, scalar
        )
        return ReferenceCredibilityOutput(
            score=score.float(),
            stage1_score=stage1_score.float(),
            reference_weights=weights.float(),
            router_weights=router.float(),
        )

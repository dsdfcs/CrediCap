from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from credicap.reference_credibility import ReferenceCredibilityModel


def unit_cosine(value: torch.Tensor) -> torch.Tensor:
    return (0.5 * (value.float() + 1.0)).clamp(0.0, 1.0)


def masked_normalize(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = weights.float().masked_fill(~mask, 0.0)
    return values / values.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


def normalized_entropy(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = masked_normalize(weights, mask)
    entropy = -(
        values.clamp_min(1.0e-8) * values.clamp_min(1.0e-8).log()
    ).sum(dim=1)
    denominator = mask.sum(dim=1).clamp_min(2).float().log()
    return (entropy / denominator).clamp(0.0, 1.0)


@dataclass
class EvidenceDecompositionOutput:
    score: torch.Tensor
    module1_score: torch.Tensor
    stage1_score: torch.Tensor
    hidden_change_rms: torch.Tensor
    structural_strength: torch.Tensor
    consensus_support: torch.Tensor
    dissent_support: torch.Tensor
    consensus_dissent_gap: torch.Tensor
    support_dispersion: torch.Tensor
    reference_disagreement: torch.Tensor
    trust_entropy: torch.Tensor
    consensus_entropy: torch.Tensor
    dissent_entropy: torch.Tensor
    consensus_mass: torch.Tensor
    dissent_mass: torch.Tensor
    reference_weights: torch.Tensor
    router_weights: torch.Tensor


class ConsensusDissentEvidenceDecomposition(nn.Module):
    """CDED-v7: split M1-trusted references into consensus and dissent."""

    STRUCTURAL_DIM = 25

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        rank_dim: int = 48,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.rank_dim = int(rank_dim)
        self.contrast_proj = nn.Sequential(
            nn.Linear(input_dim * 3, rank_dim),
            nn.LayerNorm(rank_dim),
            nn.GELU(),
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(self.STRUCTURAL_DIM, rank_dim),
            nn.LayerNorm(rank_dim),
            nn.GELU(),
        )
        self.adapter = nn.Sequential(
            nn.Linear(rank_dim * 2, rank_dim),
            nn.LayerNorm(rank_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rank_dim, hidden_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def decompose(
        self,
        batch: Dict[str, torch.Tensor],
        reference_weights: torch.Tensor,
        stage1_score: torch.Tensor,
        scalar: torch.Tensor,
    ):
        candidate = F.normalize(batch["candidate"].float(), dim=-1)
        references = F.normalize(batch["references"].float(), dim=-1)
        mask = batch["reference_mask"].bool()
        trust = masked_normalize(reference_weights, mask)

        support = unit_cosine(
            torch.einsum("bd,bnd->bn", candidate, references)
        ).masked_fill(~mask, 0.0)
        support_mean = (trust * support).sum(dim=1)
        support_dispersion = (
            trust * (support - support_mean[:, None]).square()
        ).sum(dim=1).sqrt()

        pair_similarity = unit_cosine(
            torch.einsum("bid,bjd->bij", references, references)
        )
        pair_mask = mask[:, :, None] & mask[:, None, :]
        diagonal = torch.eye(
            mask.shape[1], dtype=torch.bool, device=mask.device
        )[None]
        peer_mask = pair_mask & ~diagonal
        peer_trust = trust[:, None, :] * peer_mask.float()
        peer_consensus = (
            pair_similarity * peer_trust
        ).sum(dim=2) / peer_trust.sum(dim=2).clamp_min(1.0e-8)
        peer_consensus = peer_consensus.masked_fill(~mask, 0.0).clamp(0.0, 1.0)

        consensus_raw = trust * (0.15 + 0.85 * peer_consensus).square()
        consensus_weights = masked_normalize(consensus_raw, mask)
        support_deviation = (support - support_mean[:, None]).abs()
        dissent_raw = trust * (
            0.10 + (1.0 - peer_consensus) + 0.75 * support_deviation
        )
        dissent_raw = dissent_raw.masked_fill(~mask, 0.0)
        dissent_sum = dissent_raw.sum(dim=1, keepdim=True)
        dissent_weights = torch.where(
            dissent_sum > 1.0e-7,
            dissent_raw / dissent_sum.clamp_min(1.0e-8),
            trust,
        )

        consensus_prototype = F.normalize(
            (consensus_weights[:, :, None] * references).sum(dim=1), dim=-1
        )
        dissent_prototype = F.normalize(
            (dissent_weights[:, :, None] * references).sum(dim=1), dim=-1
        )
        consensus_support = (consensus_weights * support).sum(dim=1)
        dissent_support = (dissent_weights * support).sum(dim=1)
        consensus_dissent_gap = consensus_support - dissent_support
        reference_disagreement = (
            trust * (1.0 - peer_consensus)
        ).sum(dim=1)
        trust_entropy = normalized_entropy(trust, mask)
        consensus_entropy = normalized_entropy(consensus_weights, mask)
        dissent_entropy = normalized_entropy(dissent_weights, mask)
        consensus_mass = (trust * peer_consensus).sum(dim=1)
        dissent_mass = (trust * (1.0 - peer_consensus)).sum(dim=1)
        structural_strength = (
            0.20
            + 0.80
            * (
                0.55 * reference_disagreement
                + 0.45 * support_dispersion
            ).clamp(0.0, 1.0)
        ).clamp(0.20, 1.0)

        decomposition_scalars = torch.stack(
            [
                stage1_score.float(),
                batch["baseline"].float(),
                support_mean,
                consensus_support,
                dissent_support,
                consensus_dissent_gap,
                support_dispersion,
                reference_disagreement,
                trust_entropy,
                consensus_entropy,
                dissent_entropy,
                consensus_mass - dissent_mass,
            ],
            dim=-1,
        )
        if scalar.shape[-1] != 13:
            raise RuntimeError(
                f"Expected 13-D M1 scalar contract, got {scalar.shape[-1]}"
            )
        structural = torch.cat(
            [decomposition_scalars, scalar.float()], dim=-1
        )
        contrast = torch.cat(
            [
                candidate - consensus_prototype,
                candidate - dissent_prototype,
                consensus_prototype - dissent_prototype,
            ],
            dim=-1,
        )
        return (
            contrast,
            structural,
            structural_strength,
            consensus_support,
            dissent_support,
            consensus_dissent_gap,
            support_dispersion,
            reference_disagreement,
            trust_entropy,
            consensus_entropy,
            dissent_entropy,
            consensus_mass,
            dissent_mass,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        stage1_score: torch.Tensor,
        stage1_hidden: torch.Tensor,
        scalar: torch.Tensor,
        reference_weights: torch.Tensor,
    ):
        decomposed = self.decompose(
            batch, reference_weights, stage1_score, scalar
        )
        contrast, structural = decomposed[:2]
        structural_strength = decomposed[2]
        latent = torch.cat(
            [self.contrast_proj(contrast), self.scalar_proj(structural)],
            dim=-1,
        )
        raw_delta = torch.tanh(self.adapter(latent).float())
        base_rms = stage1_hidden.float().pow(2).mean(
            dim=1, keepdim=True
        ).sqrt().clamp_min(0.10)
        hidden_delta = (
            0.30 * base_rms * structural_strength[:, None] * raw_delta
        )
        enhanced_hidden = stage1_hidden.float() + hidden_delta
        hidden_change_rms = (
            hidden_delta.pow(2).mean(dim=1) + 1.0e-12
        ).sqrt()
        return (enhanced_hidden, hidden_change_rms, *decomposed[2:])


class IntegratedEvidenceScorer(nn.Module):
    VARIANTS = ("m1", "m12")

    def __init__(
        self,
        module1: ReferenceCredibilityModel,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        variant: str,
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.module1 = module1
        self.module2 = ConsensusDissentEvidenceDecomposition(
            input_dim, hidden_dim, dropout
        )
        self.module1.requires_grad_(False)

    def forward(self, batch: Dict[str, torch.Tensor]) -> EvidenceDecompositionOutput:
        self.module1.eval()
        self.module2.eval()
        with torch.no_grad():
            stage1_score, reference_weights, stage1_hidden, scalar = (
                self.module1.reference_trust(batch)
            )
            module1_score, original_router, _ = (
                self.module1.evidence_router(
                    batch["baseline"],
                    stage1_score,
                    stage1_hidden,
                    scalar,
                )
            )
            decomposed = self.module2(
                batch,
                stage1_score,
                stage1_hidden,
                scalar,
                reference_weights,
            )
            enhanced_hidden = decomposed[0]
            module2_score, module2_router, _ = (
                self.module1.evidence_router(
                    batch["baseline"],
                    stage1_score,
                    enhanced_hidden,
                    scalar,
                )
            )

        if self.variant == "m1":
            score = module1_score
            router = original_router
            hidden_change_rms = torch.zeros_like(score)
        else:
            score = module2_score
            router = module2_router
            hidden_change_rms = decomposed[1]

        return EvidenceDecompositionOutput(
            score=score.float(),
            module1_score=module1_score.float(),
            stage1_score=stage1_score.float(),
            hidden_change_rms=hidden_change_rms.float(),
            structural_strength=decomposed[2].float(),
            consensus_support=decomposed[3].float(),
            dissent_support=decomposed[4].float(),
            consensus_dissent_gap=decomposed[5].float(),
            support_dispersion=decomposed[6].float(),
            reference_disagreement=decomposed[7].float(),
            trust_entropy=decomposed[8].float(),
            consensus_entropy=decomposed[9].float(),
            dissent_entropy=decomposed[10].float(),
            consensus_mass=decomposed[11].float(),
            dissent_mass=decomposed[12].float(),
            reference_weights=reference_weights.float(),
            router_weights=router.float(),
        )

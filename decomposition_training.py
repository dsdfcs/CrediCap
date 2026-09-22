from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from credicap.reference_training import ReferenceCredibilityTrainingModel


def _unit_cosine(x: torch.Tensor) -> torch.Tensor:
    return (0.5 * (x.float() + 1.0)).clamp(0.0, 1.0)


def _masked_normalize(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = weights.float().masked_fill(~mask, 0.0)
    return w / w.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


def _normalized_entropy(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = _masked_normalize(weights, mask)
    ent = -(w.clamp_min(1.0e-8) * w.clamp_min(1.0e-8).log()).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(2).float().log()
    return (ent / denom).clamp(0.0, 1.0)


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
    """Module 2 v7: decompose trusted references into consensus and dissent evidence.

    M1 already answers *which references are trustworthy*.  CDED keeps that trust
    fixed and deterministically decomposes the trusted set into two complementary
    evidence views:

      1) consensus core: trusted references that agree with the rest of the set;
      2) dissent/risk: trusted references that are structurally peripheral and/or
         candidate-specific outliers.

    There is deliberately NO learned reference attention, NO Stage-1 score rewrite,
    and NO post-hoc score residual.  A small low-rank adapter converts the
    consensus-vs-dissent decomposition into a hidden-evidence update.  The final
    score is produced only by the original frozen M1 evidence router.
    """

    STRUCTURAL_DIM = 25  # 12 decomposition scalars + the 13-D M1 scalar contract

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, rank_dim: int = 48) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.rank_dim = int(rank_dim)

        # Three explicit contrast vectors: candidate-core, candidate-dissent,
        # and consensus-core minus dissent prototype.
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

        # Identity initialization: M1 is reproduced exactly before training.
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
        trust = _masked_normalize(reference_weights, mask)

        support = _unit_cosine(torch.einsum("bd,bnd->bn", candidate, references))
        support = support.masked_fill(~mask, 0.0)
        support_mean = (trust * support).sum(dim=1)
        support_disp = (trust * (support - support_mean[:, None]).square()).sum(dim=1).sqrt()

        rr = _unit_cosine(torch.einsum("bid,bjd->bij", references, references))
        pair_mask = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device).unsqueeze(0)
        peer_mask = pair_mask & ~eye

        # M1 trust-weighted peer agreement for each reference.
        peer_trust = trust[:, None, :] * peer_mask.float()
        peer_norm = peer_trust.sum(dim=2).clamp_min(1.0e-8)
        peer_consensus = (rr * peer_trust).sum(dim=2) / peer_norm
        peer_consensus = peer_consensus.masked_fill(~mask, 0.0).clamp(0.0, 1.0)

        # Consensus weights are deterministic: high M1 trust + high peer agreement.
        consensus_raw = trust * (0.15 + 0.85 * peer_consensus).square()
        consensus_w = _masked_normalize(consensus_raw, mask)

        # Dissent/risk emphasizes trusted-but-peripheral references and candidate-
        # specific support deviations.  This is not a learned attention mechanism.
        support_dev = (support - support_mean[:, None]).abs()
        dissent_raw = trust * (0.10 + (1.0 - peer_consensus) + 0.75 * support_dev)
        dissent_raw = dissent_raw.masked_fill(~mask, 0.0)
        # If a set is perfectly coherent, retain a numerically stable weak dissent view.
        fallback = trust
        dissent_sum = dissent_raw.sum(dim=1, keepdim=True)
        dissent_w = torch.where(
            dissent_sum > 1.0e-7,
            dissent_raw / dissent_sum.clamp_min(1.0e-8),
            fallback,
        )

        consensus_proto = F.normalize((consensus_w[:, :, None] * references).sum(dim=1), dim=-1)
        dissent_proto = F.normalize((dissent_w[:, :, None] * references).sum(dim=1), dim=-1)

        consensus_support = (consensus_w * support).sum(dim=1)
        dissent_support = (dissent_w * support).sum(dim=1)
        gap = consensus_support - dissent_support

        reference_disagreement = (trust * (1.0 - peer_consensus)).sum(dim=1)
        trust_entropy = _normalized_entropy(trust, mask)
        consensus_entropy = _normalized_entropy(consensus_w, mask)
        dissent_entropy = _normalized_entropy(dissent_w, mask)

        consensus_mass = (trust * peer_consensus).sum(dim=1)
        dissent_mass = (trust * (1.0 - peer_consensus)).sum(dim=1)

        # Analytic structural strength prevents a learned gate from collapsing to a
        # constant ~0.5 as happened in v6.  Coherent sets are changed less; genuinely
        # disputed sets receive more representation capacity.
        structural_strength = (
            0.20 + 0.80 * (0.55 * reference_disagreement + 0.45 * support_disp).clamp(0.0, 1.0)
        ).clamp(0.20, 1.0)

        decomposition_scalars = torch.stack(
            [
                stage1_score.float(),
                batch["baseline"].float(),
                support_mean,
                consensus_support,
                dissent_support,
                gap,
                support_disp,
                reference_disagreement,
                trust_entropy,
                consensus_entropy,
                dissent_entropy,
                consensus_mass - dissent_mass,
            ],
            dim=-1,
        )
        if scalar.shape[-1] != 13:
            raise RuntimeError(f"Expected 13-D M1 scalar contract, got {scalar.shape[-1]}")
        structural = torch.cat([decomposition_scalars, scalar.float()], dim=-1)

        contrast = torch.cat(
            [candidate - consensus_proto, candidate - dissent_proto, consensus_proto - dissent_proto],
            dim=-1,
        )
        return (
            contrast,
            structural,
            structural_strength,
            consensus_support,
            dissent_support,
            gap,
            support_disp,
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
        (
            contrast,
            structural,
            structural_strength,
            consensus_support,
            dissent_support,
            gap,
            support_disp,
            reference_disagreement,
            trust_entropy,
            consensus_entropy,
            dissent_entropy,
            consensus_mass,
            dissent_mass,
        ) = self.decompose(batch, reference_weights, stage1_score, scalar)

        z = torch.cat([self.contrast_proj(contrast), self.scalar_proj(structural)], dim=-1)
        raw_delta = torch.tanh(self.adapter(z).float())
        base_rms = stage1_hidden.float().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(0.10)
        # Representation-only update.  No direct stage1/final-score correction exists.
        hidden_delta = 0.30 * base_rms * structural_strength[:, None] * raw_delta
        enhanced_hidden = stage1_hidden.float() + hidden_delta
        hidden_change_rms = (hidden_delta.pow(2).mean(dim=1) + 1.0e-12).sqrt()

        return (
            enhanced_hidden,
            hidden_change_rms,
            structural_strength,
            consensus_support,
            dissent_support,
            gap,
            support_disp,
            reference_disagreement,
            trust_entropy,
            consensus_entropy,
            dissent_entropy,
            consensus_mass,
            dissent_mass,
        )


class ConsensusDissentTrainingModel(nn.Module):
    VARIANTS = ("m1", "m12")

    def __init__(
        self,
        module1: ReferenceCredibilityTrainingModel,
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
        self.module2 = ConsensusDissentEvidenceDecomposition(input_dim, hidden_dim, dropout)
        for p in self.module1.parameters():
            p.requires_grad_(False)

    def set_trainable_module2(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.module2.parameters():
            p.requires_grad_(True)

    def _stage1(self, batch):
        self.module1.eval()
        with torch.no_grad():
            return self.module1.reference_trust(batch)

    def _original_m1(self, batch, stage1_score, stage1_hidden, scalar):
        self.module1.eval()
        with torch.no_grad():
            return self.module1.evidence_router(batch["baseline"], stage1_score, stage1_hidden, scalar)

    def forward(self, batch: Dict[str, torch.Tensor]) -> EvidenceDecompositionOutput:
        stage1_score, reference_weights, stage1_hidden, scalar = self._stage1(batch)
        module1_score, original_router_weights, _ = self._original_m1(
            batch, stage1_score, stage1_hidden, scalar
        )

        with torch.no_grad():
            decomp = self.module2.decompose(batch, reference_weights, stage1_score, scalar)
        (
            _, _, structural_strength, consensus_support, dissent_support, gap,
            support_disp, reference_disagreement, trust_entropy, consensus_entropy,
            dissent_entropy, consensus_mass, dissent_mass,
        ) = decomp

        if self.variant == "m1":
            score = module1_score.float()
            hidden_change_rms = torch.zeros_like(score)
            router_weights = original_router_weights
        else:
            (
                enhanced_hidden,
                hidden_change_rms,
                structural_strength,
                consensus_support,
                dissent_support,
                gap,
                support_disp,
                reference_disagreement,
                trust_entropy,
                consensus_entropy,
                dissent_entropy,
                consensus_mass,
                dissent_mass,
            ) = self.module2(batch, stage1_score, stage1_hidden, scalar, reference_weights)
            # Final score still comes ONLY from the original frozen M1 router.
            score, router_weights, _ = self.module1.evidence_router(
                batch["baseline"], stage1_score, enhanced_hidden, scalar
            )

        return EvidenceDecompositionOutput(
            score=score.float(),
            module1_score=module1_score.float(),
            stage1_score=stage1_score.float(),
            hidden_change_rms=hidden_change_rms.float(),
            structural_strength=structural_strength.float(),
            consensus_support=consensus_support.float(),
            dissent_support=dissent_support.float(),
            consensus_dissent_gap=gap.float(),
            support_dispersion=support_disp.float(),
            reference_disagreement=reference_disagreement.float(),
            trust_entropy=trust_entropy.float(),
            consensus_entropy=consensus_entropy.float(),
            dissent_entropy=dissent_entropy.float(),
            consensus_mass=consensus_mass.float(),
            dissent_mass=dissent_mass.float(),
            reference_weights=reference_weights.float(),
            router_weights=router_weights.float(),
        )

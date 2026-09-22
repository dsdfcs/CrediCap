#!/usr/bin/env python3
"""True structural ablation of the RefFLEUR initial score ``s_0``.

No substitute prior is introduced: no CLIP cosine score, no constant, and no
hidden copy of RefFLEUR.  The RCE score head predicts an absolute score from
image/candidate/reference evidence; its four-way router excludes the RefFLEUR
expert.  CDED and the signed-feedback stage retain their mechanisms while all
s_0-dependent inputs are removed.  Every changed-dimensional component is
retrained on Polaris train and selected on Polaris validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from credicap import pipeline
from credicap import train_reference
from credicap import training_utils
from credicap import verification_utils
from credicap.expected_feedback import FeedbackExpectationNet
from credicap.score_correction import DirectionMagnitudeCorrector


FORMAT = "credicap-true-drop-s0-v1"
FEATURE_FORMAT = "credicap-true-drop-s0-features-v1"
CHECKPOINT_FORMAT = "credicap-true-drop-s0-signed-error-v1"
SCALAR_DIM = 12
BASE_DIM = 19
FEATURE_NAMES = (
    "module1_score", "m1_stage1_score", "m2_hidden_change_rms",
    "structural_strength", "consensus_support", "dissent_support",
    "consensus_dissent_gap", "support_dispersion",
    "reference_disagreement", "trust_entropy", "consensus_entropy",
    "dissent_entropy", "consensus_mass", "dissent_mass",
    "router_stage1", "router_image", "router_reference",
    "router_interaction", "candidate_length",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: Path, device: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_torch(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def safe_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1.0e-4, 1.0 - 1.0e-4))


def make_batch(split: dict, indices, device: torch.device) -> dict[str, torch.Tensor]:
    """Construct a model batch without ever reading split['baseline']."""
    index = torch.as_tensor(indices, dtype=torch.long)
    result = {}
    for key in ("image", "candidate", "references", "reference_mask", "candidate_length"):
        value = split[key].index_select(0, index).to(device, non_blocking=True)
        result[key] = value.float() if value.is_floating_point() else value
    return result


def masked_normalize(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = weights.float().masked_fill(~mask, 0.0)
    return values / values.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


def normalized_entropy(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = masked_normalize(weights, mask)
    entropy = -(values.clamp_min(1.0e-8) * values.clamp_min(1.0e-8).log()).sum(1)
    denominator = mask.sum(1).clamp_min(2).float().log()
    return (entropy / denominator).clamp(0.0, 1.0)


@dataclass
class ModelOutput:
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


class DropS0ReferenceTrust(nn.Module):
    """RCE without s0: absolute score from multimodal/reference evidence."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.shared_projection.weight)
        self.trust = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.state = nn.Sequential(
            nn.Linear(hidden_dim * 3 + SCALAR_DIM, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.absolute_score = nn.Linear(hidden_dim, 1)
        zero_linear(self.absolute_score)

    @staticmethod
    def stats(values: torch.Tensor, mask: torch.Tensor):
        weights = mask.to(values.dtype)
        count = weights.sum(1).clamp_min(1.0)
        mean = (values * weights).sum(1) / count
        std = ((((values - mean[:, None]) ** 2) * weights).sum(1) / count).sqrt()
        minimum = values.masked_fill(~mask, torch.inf).min(1).values
        maximum = values.masked_fill(~mask, -torch.inf).max(1).values
        minimum = torch.where(torch.isfinite(minimum), minimum, mean)
        maximum = torch.where(torch.isfinite(maximum), maximum, mean)
        return mean, std, minimum, maximum

    def forward(self, batch: dict[str, torch.Tensor]):
        image = F.normalize(self.shared_projection(batch["image"]), dim=-1)
        candidate = F.normalize(self.shared_projection(batch["candidate"]), dim=-1)
        references = F.normalize(self.shared_projection(batch["references"]), dim=-1)
        mask = batch["reference_mask"].bool()
        image_reference = torch.einsum("bd,brd->br", image, references)
        candidate_reference = torch.einsum("bd,brd->br", candidate, references)
        pairwise = torch.einsum("brd,bsd->brs", references, references)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        diagonal = torch.eye(pairwise.shape[1], dtype=torch.bool, device=pairwise.device)[None]
        peer_mask = pair_mask & ~diagonal
        peer_count = peer_mask.to(pairwise.dtype).sum(2).clamp_min(1.0)
        peer_consensus = pairwise.masked_fill(~peer_mask, 0.0).sum(2) / peer_count
        trust_input = torch.stack([image_reference, peer_consensus, candidate_reference], -1)
        logits = self.trust(trust_input).squeeze(-1).masked_fill(~mask, -1.0e4)
        weights = torch.softmax(logits, 1)
        audited = torch.einsum("br,brd->bd", weights, references)
        ic = (image * candidate).sum(-1)
        ac = (audited * candidate).sum(-1)
        ai = (audited * image).sum(-1)
        cr = self.stats(candidate_reference, mask)
        ir = self.stats(image_reference, mask)
        scalar = torch.stack([ic, ac, ai, *cr, *ir, batch["candidate_length"]], -1)
        if scalar.shape[1] != SCALAR_DIM:
            raise RuntimeError(f"drop-s0 scalar contract is {SCALAR_DIM}, got {scalar.shape[1]}")
        hidden = self.state(torch.cat([image, candidate, audited, scalar], -1))
        score = torch.sigmoid(self.absolute_score(hidden).squeeze(-1))
        return score, weights, hidden, scalar


class DropS0Router(nn.Module):
    """Four experts: stage-1, image, trusted-reference, learned interaction."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim + SCALAR_DIM, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.interaction_score = nn.Linear(hidden_dim, 1)
        self.image_scale = nn.Parameter(torch.tensor(6.0))
        self.image_bias = nn.Parameter(torch.tensor(-1.5))
        self.reference_scale = nn.Parameter(torch.tensor(6.0))
        self.reference_bias = nn.Parameter(torch.tensor(-1.5))
        self.route_gain = nn.Parameter(torch.tensor(0.0))
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + SCALAR_DIM + 8, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim + SCALAR_DIM + 8 + 4, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        zero_linear(self.interaction_score)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.router[-1].bias.data[0] = 2.0
        zero_linear(self.residual[-1])

    def forward(self, stage1: torch.Tensor, stage1_hidden: torch.Tensor, scalar: torch.Tensor):
        hidden = self.interaction(torch.cat([stage1_hidden, scalar], -1))
        learned = torch.sigmoid(self.interaction_score(hidden).squeeze(-1))
        image_score = torch.sigmoid(self.image_scale * scalar[:, 0] + self.image_bias)
        reference_score = torch.sigmoid(self.reference_scale * scalar[:, 1] + self.reference_bias)
        experts = torch.stack([stage1, image_score, reference_score, learned], -1)
        disagreement = torch.stack([
            experts.mean(-1), experts.std(-1, unbiased=False), experts.min(-1).values,
            experts.max(-1).values, (stage1-image_score).abs(),
            (stage1-reference_score).abs(), (image_score-reference_score).abs(),
            (stage1-learned).abs(),
        ], -1)
        router_input = torch.cat([hidden, scalar, disagreement], -1)
        router = torch.softmax(self.router(router_input), -1)
        routed_logit = (router * safe_logit(experts)).sum(-1)
        residual = 0.75 * torch.tanh(self.residual(torch.cat([router_input, experts], -1)).squeeze(-1))
        routed_difference = torch.tanh(routed_logit - safe_logit(stage1))
        strength = torch.sigmoid(disagreement[:, 1] * 4.0) - 0.5
        score = torch.sigmoid(safe_logit(stage1) + residual + torch.tanh(self.route_gain) * strength * routed_difference)
        return score, router, hidden


class DropS0M1(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, stage: int = 2) -> None:
        super().__init__()
        self.stage = stage
        self.reference_trust = DropS0ReferenceTrust(input_dim, hidden_dim, dropout)
        self.evidence_router = DropS0Router(hidden_dim, dropout)

    def set_trainable_stage(self, stage: int) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        module = self.reference_trust if stage == 1 else self.evidence_router
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    def forward_raw(self, batch):
        stage1, weights, hidden, scalar = self.reference_trust(batch)
        if self.stage == 1:
            router = torch.zeros(len(stage1), 4, device=stage1.device, dtype=stage1.dtype)
            router[:, 0] = 1.0
            return stage1, stage1, weights, hidden, scalar, router
        score, router, routed_hidden = self.evidence_router(stage1, hidden, scalar)
        return score, stage1, weights, hidden, scalar, router

    def forward(self, batch):
        score, stage1, weights, _, _, router = self.forward_raw(batch)
        zero = torch.zeros_like(score)
        return ModelOutput(score, score, stage1, zero, zero, zero, zero, zero, zero,
                           zero, zero, zero, zero, zero, zero, weights, router)


class DropS0CDED(nn.Module):
    STRUCTURAL_DIM = 23  # 11 decomposition scalars + 12-D RCE scalar contract

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, rank_dim: int = 48) -> None:
        super().__init__()
        self.contrast_proj = nn.Sequential(nn.Linear(input_dim * 3, rank_dim), nn.LayerNorm(rank_dim), nn.GELU())
        self.scalar_proj = nn.Sequential(nn.Linear(self.STRUCTURAL_DIM, rank_dim), nn.LayerNorm(rank_dim), nn.GELU())
        self.adapter = nn.Sequential(
            nn.Linear(rank_dim * 2, rank_dim), nn.LayerNorm(rank_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(rank_dim, hidden_dim),
        )
        zero_linear(self.adapter[-1])

    def decompose(self, batch, reference_weights, stage1, scalar):
        candidate = F.normalize(batch["candidate"].float(), dim=-1)
        references = F.normalize(batch["references"].float(), dim=-1)
        mask = batch["reference_mask"].bool()
        trust = masked_normalize(reference_weights, mask)
        support = (0.5 * (torch.einsum("bd,bnd->bn", candidate, references) + 1.0)).clamp(0, 1)
        support = support.masked_fill(~mask, 0.0)
        support_mean = (trust * support).sum(1)
        support_disp = (trust * (support - support_mean[:, None]).square()).sum(1).sqrt()
        rr = (0.5 * (torch.einsum("bid,bjd->bij", references, references) + 1.0)).clamp(0, 1)
        eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
        peer_mask = (mask[:, :, None] & mask[:, None, :]) & ~eye
        peer_trust = trust[:, None, :] * peer_mask.float()
        peer_consensus = (rr * peer_trust).sum(2) / peer_trust.sum(2).clamp_min(1.0e-8)
        peer_consensus = peer_consensus.masked_fill(~mask, 0.0).clamp(0, 1)
        consensus_w = masked_normalize(trust * (0.15 + 0.85 * peer_consensus).square(), mask)
        dissent_raw = trust * (0.10 + 1.0 - peer_consensus + 0.75 * (support-support_mean[:, None]).abs())
        dissent_raw = dissent_raw.masked_fill(~mask, 0.0)
        dissent_w = torch.where(dissent_raw.sum(1, keepdim=True) > 1e-7,
                                dissent_raw / dissent_raw.sum(1, keepdim=True).clamp_min(1e-8), trust)
        consensus_proto = F.normalize((consensus_w[:, :, None] * references).sum(1), dim=-1)
        dissent_proto = F.normalize((dissent_w[:, :, None] * references).sum(1), dim=-1)
        consensus_support = (consensus_w * support).sum(1)
        dissent_support = (dissent_w * support).sum(1)
        gap = consensus_support - dissent_support
        disagreement = (trust * (1.0-peer_consensus)).sum(1)
        trust_entropy = normalized_entropy(trust, mask)
        consensus_entropy = normalized_entropy(consensus_w, mask)
        dissent_entropy = normalized_entropy(dissent_w, mask)
        consensus_mass = (trust * peer_consensus).sum(1)
        dissent_mass = (trust * (1.0-peer_consensus)).sum(1)
        strength = (0.20 + 0.80 * (0.55*disagreement + 0.45*support_disp).clamp(0, 1)).clamp(0.20, 1.0)
        decomposition = torch.stack([
            stage1, support_mean, consensus_support, dissent_support, gap, support_disp,
            disagreement, trust_entropy, consensus_entropy, dissent_entropy,
            consensus_mass-dissent_mass,
        ], -1)
        structural = torch.cat([decomposition, scalar.float()], -1)
        if structural.shape[1] != self.STRUCTURAL_DIM:
            raise RuntimeError(f"drop-s0 CDED contract is {self.STRUCTURAL_DIM}, got {structural.shape[1]}")
        contrast = torch.cat([candidate-consensus_proto, candidate-dissent_proto,
                              consensus_proto-dissent_proto], -1)
        return (contrast, structural, strength, consensus_support, dissent_support, gap,
                support_disp, disagreement, trust_entropy, consensus_entropy,
                dissent_entropy, consensus_mass, dissent_mass)

    def forward(self, batch, reference_weights, stage1, hidden, scalar):
        values = self.decompose(batch, reference_weights, stage1, scalar)
        contrast, structural, strength = values[:3]
        z = torch.cat([self.contrast_proj(contrast), self.scalar_proj(structural)], -1)
        raw = torch.tanh(self.adapter(z).float())
        rms = hidden.float().pow(2).mean(1, keepdim=True).sqrt().clamp_min(0.10)
        delta = 0.30 * rms * strength[:, None] * raw
        return hidden.float() + delta, (delta.pow(2).mean(1)+1e-12).sqrt(), values[2:]


class DropS0Full(nn.Module):
    def __init__(self, m1: DropS0M1, input_dim: int, hidden_dim: int, dropout: float, use_m2: bool = True) -> None:
        super().__init__()
        self.m1 = m1
        self.m2 = DropS0CDED(input_dim, hidden_dim, dropout)
        self.use_m2 = use_m2
        for parameter in self.m1.parameters():
            parameter.requires_grad_(False)

    def forward(self, batch):
        with torch.no_grad():
            module1, stage1, weights, hidden, scalar, original_router = self.m1.forward_raw(batch)
            diagnostics = self.m2.decompose(batch, weights, stage1, scalar)[2:]
        if self.use_m2:
            enhanced, hidden_rms, diagnostics = self.m2(batch, weights, stage1, hidden, scalar)
            score, router, _ = self.m1.evidence_router(stage1, enhanced, scalar)
        else:
            score, router, hidden_rms = module1, original_router, torch.zeros_like(module1)
        (strength, consensus, dissent, gap, dispersion, disagreement, trust_entropy,
         consensus_entropy, dissent_entropy, consensus_mass, dissent_mass) = diagnostics
        return ModelOutput(score.float(), module1.float(), stage1.float(), hidden_rms.float(),
                           strength.float(), consensus.float(), dissent.float(), gap.float(),
                           dispersion.float(), disagreement.float(), trust_entropy.float(),
                           consensus_entropy.float(), dissent_entropy.float(),
                           consensus_mass.float(), dissent_mass.float(), weights.float(), router.float())


@torch.no_grad()
def predict(model: nn.Module, split: dict, batch_size: int, device: torch.device, description: str, fields=("score",)):
    model.eval()
    output = {field: [] for field in fields}
    loader = DataLoader(TensorDataset(torch.arange(len(split["sample_ids"]))), batch_size=batch_size, shuffle=False)
    for (indices,) in tqdm(loader, desc=description, dynamic_ncols=True, leave=False):
        row = model(make_batch(split, indices, device))
        for field in fields:
            output[field].append(getattr(row, field).float().cpu())
    return {key: torch.cat(value) for key, value in output.items()}


def objective(metrics: dict) -> float:
    return float(metrics["tau_x100"] - 60.0*metrics["mae"] - 30.0*metrics["rmse"])


def train_m1_stage(args, cache: dict, run_root: Path, stage: int, device: torch.device) -> Path:
    output = run_root / "checkpoints" / f"drop_s0_m1_stage{stage}.best.pt"
    if output.is_file():
        payload = torch_load(output)
        if payload.get("format") == FORMAT and payload.get("stage") == stage:
            print(f"Reuse true drop-s0 M1 stage {stage}: {output}")
            return output
    model = DropS0M1(int(cache["embedding_dim"]), args.m1_hidden_dim, args.dropout, stage).to(device)
    if stage == 2:
        previous = torch_load(run_root / "checkpoints" / "drop_s0_m1_stage1.best.pt")
        model.load_state_dict(previous["model"], strict=True)
    model.set_trainable_stage(stage)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.m1_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    train_split, val_split = cache["train"], cache["val"]
    loader = DataLoader(TensorDataset(torch.arange(len(train_split["sample_ids"]))),
                        batch_size=args.m1_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed+stage))
    best_state, best_metrics, best_value, stale = None, None, -math.inf, 0
    for epoch in range(1, args.m1_epochs+1):
        model.train()
        if stage == 2:
            model.reference_trust.eval()
        for (indices,) in tqdm(loader, desc=f"drop-s0 RCE stage{stage} epoch {epoch:02d}", dynamic_ncols=True):
            batch = make_batch(train_split, indices, device)
            target = train_split["gold"].index_select(0, indices).to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                out = model(batch)
                point = 0.65*F.smooth_l1_loss(out.score, target, beta=0.10) + 0.35*F.mse_loss(out.score, target)
                loss = point
                if stage == 1 and len(indices) > 1:
                    corrupted = dict(batch)
                    refs = batch["references"].clone()
                    counts = batch["reference_mask"].sum(1).long()
                    slots = torch.floor(torch.rand(len(indices), device=device)*counts.clamp_min(1).float()).long()
                    rows = torch.arange(len(indices), device=device)
                    refs[rows, slots] = torch.roll(batch["references"][:, 0], 1, 0)
                    corrupted["references"] = refs
                    loss = loss + args.reference_corruption_weight * model(corrupted).reference_weights[rows, slots].mean()
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            scaler.step(optimizer); scaler.update()
        values = train_reference.metric_values(
            predict(model, val_split, args.eval_batch_size, device, f"drop-s0 RCE val {epoch:02d}")["score"].numpy(), val_split)
        train_reference.print_metrics(f"drop-s0 RCE stage{stage}", values)
        value = objective(values)
        if value > best_value:
            best_value, best_metrics, stale = value, values, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    payload = {"format": FORMAT, "stage": stage, "model": best_state,
               "embedding_dim": int(cache["embedding_dim"]), "hidden_dim": args.m1_hidden_dim,
               "dropout": args.dropout, "best_validation": best_metrics,
               "s0_present": False, "replacement_prior": None,
               "training": "Polaris train", "selection": "Polaris validation",
               "benchmark_labels_used": False}
    save_torch(payload, output)
    return output


def load_m1(path: Path, device: torch.device) -> tuple[DropS0M1, dict]:
    payload = torch_load(path)
    if payload.get("format") != FORMAT or payload.get("s0_present") is not False:
        raise RuntimeError("Not a true drop-s0 M1 checkpoint")
    model = DropS0M1(payload["embedding_dim"], payload["hidden_dim"], payload["dropout"], 2).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False); model.eval()
    return model, payload


def train_m2(args, cache: dict, run_root: Path, m1_path: Path, device: torch.device) -> Path:
    output = run_root / "checkpoints" / "drop_s0_m2.best.pt"
    if output.is_file() and torch_load(output).get("format") == FORMAT:
        print(f"Reuse true drop-s0 CDED: {output}")
        return output
    m1, p1 = load_m1(m1_path, device)
    model = DropS0Full(m1, p1["embedding_dim"], p1["hidden_dim"], p1["dropout"], True).to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for parameter in model.m2.parameters(): parameter.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.m2_lr, weight_decay=args.weight_decay)
    train_split, val_split = cache["train"], cache["val"]
    m1_reference = predict(model, train_split, args.eval_batch_size, device, "drop-s0 M1 train reference", ("module1_score",))["module1_score"]
    errors = (m1_reference-train_split["gold"].float()).abs()
    easy = (errors <= torch.quantile(errors, args.easy_anchor_quantile)).float()
    pairs = train_reference.make_ranking_pairs(
        train_split, args.m2_rank_minimum_gap,
        args.m2_rank_maximum_per_group, args.seed+7,
    )
    pair_rng = np.random.default_rng(args.seed+7707)
    loader = DataLoader(TensorDataset(torch.arange(len(train_split["sample_ids"]))),
                        batch_size=args.m2_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed+707))
    best_state, best_metrics, best_value, stale = None, None, -math.inf, 0
    for epoch in range(1, args.m2_epochs+1):
        model.train(); model.m1.eval()
        for (indices,) in tqdm(loader, desc=f"drop-s0 CDED epoch {epoch:02d}", dynamic_ncols=True):
            batch = make_batch(train_split, indices, device)
            target = train_split["gold"].index_select(0, indices).to(device).float()
            mask = easy.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            point = 0.55*F.smooth_l1_loss(out.score, target, beta=0.08) + 0.45*F.mse_loss(out.score, target)
            anchor = (((out.score-out.module1_score).square())*mask).sum()/mask.sum().clamp_min(1.0)
            hidden = out.hidden_change_rms.square().mean()
            selected = pair_rng.choice(len(pairs), size=min(args.pair_batch_size, len(pairs)), replace=False)
            pair = pairs[selected]
            better = model(make_batch(train_split, pair[:, 0], device)).score
            worse = model(make_batch(train_split, pair[:, 1], device)).score
            rank = F.softplus((worse-better)/args.m2_rank_temperature).mean()
            loss = point + args.rank_loss_weight*rank + args.easy_anchor_weight*anchor + args.hidden_regularization_weight*hidden
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip); optimizer.step()
        values = train_reference.metric_values(
            predict(model, val_split, args.eval_batch_size, device, f"drop-s0 CDED val {epoch:02d}")["score"].numpy(), val_split)
        train_reference.print_metrics("drop-s0 M1+CDED", values)
        value = objective(values)
        if value > best_value:
            best_value, best_metrics, stale = value, values, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.m2.state_dict().items()}
        else: stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience: break
    save_torch({"format": FORMAT, "module_state": best_state, "best_validation": best_metrics,
                "m1_sha256": sha256(m1_path), "s0_present": False,
                "replacement_prior": None, "benchmark_labels_used": False}, output)
    return output


def build_full(m1_path: Path, m2_path: Path, device: torch.device) -> DropS0Full:
    m1, p1 = load_m1(m1_path, device)
    p2 = torch_load(m2_path)
    if p2.get("s0_present") is not False or p2.get("replacement_prior") is not None:
        raise RuntimeError("CDED checkpoint is not true drop-s0")
    model = DropS0Full(m1, p1["embedding_dim"], p1["hidden_dim"], p1["dropout"], True).to(device)
    model.m2.load_state_dict(p2["module_state"], strict=True)
    model.requires_grad_(False); model.eval()
    return model


def source_split(cache: dict, split: str) -> dict:
    if cache.get("format") != train_reference.CACHE_FORMAT:
        raise RuntimeError(f"Wrong source cache: {cache.get('format')}")
    return cache[split] if cache.get("kind") == "polaris" else cache["split"]


def prepare_features(args, source_path: Path, split_name: str, feedback_path: Path,
                     output: Path, m1_path: Path, m2_path: Path, device: torch.device) -> None:
    if output.is_file() and torch_load(output).get("format") == FEATURE_FORMAT:
        print(f"Reuse true drop-s0 features: {output}")
        return
    source = torch_load(source_path)
    split = source_split(source, split_name)
    feedback_rows = verification_utils.read_jsonl_resume(feedback_path)
    verification_utils.validate_rows(feedback_rows, split["sample_ids"])
    feedback = verification_utils.matrix_from_rows(feedback_rows, split["sample_ids"])
    fields = tuple(name for name in ModelOutput.__dataclass_fields__ if name not in {"reference_weights"})
    model = build_full(m1_path, m2_path, device)
    values = predict(model, split, args.eval_batch_size, device, f"drop-s0 features {split_name}", fields)
    matrix = torch.stack([
        values["module1_score"], values["stage1_score"], values["hidden_change_rms"],
        values["structural_strength"], values["consensus_support"], values["dissent_support"],
        values["consensus_dissent_gap"], values["support_dispersion"],
        values["reference_disagreement"], values["trust_entropy"], values["consensus_entropy"],
        values["dissent_entropy"], values["consensus_mass"], values["dissent_mass"],
    ], 1)
    features = torch.cat([matrix, values["router_weights"], split["candidate_length"].float()[:, None]], 1)
    if features.shape != (len(split["sample_ids"]), BASE_DIM):
        raise RuntimeError(f"Expected true drop-s0 features [N,{BASE_DIM}], got {tuple(features.shape)}")
    payload = {"format": FEATURE_FORMAT, "split": split_name, "feature_names": FEATURE_NAMES,
               "feedback_fields": verification_utils.FIELDS+("VALID",),
               "sample_ids": list(split["sample_ids"]), "groups": list(split["groups"]),
               "records": list(split["records"]), "gold": split["gold"].float().cpu(),
               "anchor": values["score"].float(), "base_features": features.float(),
               "feedback": torch.from_numpy(feedback).float(), "m1_sha256": sha256(m1_path),
               "m2_sha256": sha256(m2_path), "s0_present": False, "replacement_prior": None,
               "base_dim": BASE_DIM, "benchmark_labels_used_for_training_or_selection": False}
    save_torch(payload, output)


class DropS0Corrector(DirectionMagnitudeCorrector):
    def __init__(self, hidden_dim=128, maximum_base_correction=0.05, maximum_feedback_correction=0.05):
        super().__init__(hidden_dim, maximum_base_correction, maximum_feedback_correction)
        self.base_encoder = nn.Sequential(nn.LayerNorm(BASE_DIM), nn.Linear(BASE_DIM, hidden_dim),
                                          nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())

    @staticmethod
    def validate_inputs(base_features, actual_feedback, expected_feedback, anchor):
        count = len(base_features)
        if tuple(base_features.shape) != (count, BASE_DIM):
            raise ValueError(f"Expected base features [N,{BASE_DIM}]")
        if tuple(actual_feedback.shape) != (count, 7) or tuple(expected_feedback.shape) != (count, 7):
            raise ValueError("Expected seven feedback fields")
        if tuple(anchor.shape) != (count,):
            raise ValueError("Expected anchor [N]")


def normalize(cache: dict, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (cache["base_features"].float()-mean)/std


@torch.no_grad()
def expectation_predict(model, values, batch_size, device, desc):
    model.eval(); rows=[]
    loader=DataLoader(TensorDataset(torch.arange(len(values))), batch_size=batch_size, shuffle=False)
    for (idx,) in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False): rows.append(model(values.index_select(0, idx).to(device)).cpu())
    return torch.cat(rows)


def train_expectation(args, train_cache, val_cache, device):
    mean=train_cache["base_features"].float().mean(0); std=train_cache["base_features"].float().std(0).clamp_min(1e-5)
    train_x=normalize(train_cache, mean, std); val_x=normalize(val_cache, mean, std)
    train_y=train_cache["feedback"][:, :7].float(); val_y=val_cache["feedback"][:, :7].float()
    model=FeedbackExpectationNet(BASE_DIM, args.expectation_hidden_dim).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=args.expectation_lr, weight_decay=args.weight_decay)
    loader=DataLoader(TensorDataset(torch.arange(len(train_x))), batch_size=args.feedback_batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(args.seed+3101))
    best_loss, best_state = math.inf, None
    for epoch in range(1, args.expectation_epochs+1):
        model.train()
        for (idx,) in tqdm(loader, desc=f"drop-s0 feedback expectation {epoch:02d}", dynamic_ncols=True):
            optimizer.zero_grad(set_to_none=True)
            loss=F.smooth_l1_loss(model(train_x.index_select(0,idx).to(device)), train_y.index_select(0,idx).to(device), beta=0.10)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip); optimizer.step()
        prediction=expectation_predict(model,val_x,args.eval_batch_size,device,"drop-s0 expectation val")
        value=float(F.mse_loss(prediction,val_y))
        if value < best_loss:
            best_loss=value; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_expected=expectation_predict(model,train_x,args.eval_batch_size,device,"drop-s0 expected train")
    val_expected=expectation_predict(model,val_x,args.eval_batch_size,device,"drop-s0 expected val")
    meta={"state":best_state,"hidden_dim":args.expectation_hidden_dim,"normalizer_mean":mean,"normalizer_std":std,
          "validation_mse":best_loss}
    return meta,(train_x,train_y,train_expected),(val_x,val_y,val_expected)


def validate_feature(cache: dict, split: str) -> None:
    if cache.get("format") != FEATURE_FORMAT or cache.get("split") != split:
        raise RuntimeError(f"Wrong true drop-s0 feature cache for {split}")
    if cache.get("s0_present") is not False or cache.get("replacement_prior") is not None:
        raise RuntimeError("Feature cache contains an s0/replacement prior")
    if tuple(cache["base_features"].shape) != (len(cache["sample_ids"]), BASE_DIM):
        raise RuntimeError("Wrong true drop-s0 feature dimensions")
    pipeline.require_all_feedback(cache, split)


def train_signed(args, run_root: Path, train_path: Path, val_path: Path, device: torch.device) -> Path:
    output=run_root/"checkpoints"/"drop_s0_signed_error.best.pt"
    if output.is_file() and torch_load(output).get("format") == CHECKPOINT_FORMAT:
        print(f"Reuse true drop-s0 SDMC: {output}"); return output
    train_cache=torch_load(train_path); val_cache=torch_load(val_path)
    validate_feature(train_cache,"train"); validate_feature(val_cache,"val")
    expectation,train_inputs,val_inputs=train_expectation(args,train_cache,val_cache,device)
    pairs,weights,audit=training_utils.make_hard_pairs(train_cache,args.rank_minimum_gap,
                                                       args.rank_maximum_per_group,args.seed)
    model=DropS0Corrector(args.feedback_hidden_dim,args.maximum_base_correction,
                          args.maximum_feedback_correction).to(device)
    base_snapshots=pipeline.train_base(args,model,train_cache,val_cache,train_inputs,val_inputs,pairs,weights,device)
    base_selected,base_candidates=pipeline.select_base(args,base_snapshots,model,val_cache,val_inputs,device)
    feedback_snapshots=pipeline.train_feedback(args,model,base_selected["alpha"],train_cache,val_cache,
                                               train_inputs,val_inputs,pairs,weights,device)
    feedback_selected,feedback_candidates=pipeline.select_feedback(args,feedback_snapshots,model,base_selected,
                                                                   val_cache,val_inputs,device)
    payload={"format":CHECKPOINT_FORMAT,"created_unix":time.time(),"s0_present":False,
             "replacement_prior":None,"base_dim":BASE_DIM,"expectation":expectation,
             "hidden_dim":args.feedback_hidden_dim,"maximum_base_correction":args.maximum_base_correction,
             "maximum_feedback_correction":args.maximum_feedback_correction,"base_selected":base_selected,
             "feedback_selected":feedback_selected,"pair_audit":audit,
             "benchmark_labels_used_for_training_or_selection":False}
    save_torch(payload,output)
    return output


def feedback_inputs(cache, checkpoint, args, device):
    meta=checkpoint["expectation"]
    normalized=normalize(cache,meta["normalizer_mean"].float(),meta["normalizer_std"].float())
    model=FeedbackExpectationNet(BASE_DIM,int(meta["hidden_dim"])).to(device); model.load_state_dict(meta["state"])
    expected=expectation_predict(model,normalized,args.eval_batch_size,device,"drop-s0 expected expert")
    return normalized,cache["feedback"][:,:7].float(),expected


def evaluate(args, run_root: Path, features: Path, checkpoint_path: Path, full_report: Path, device: torch.device) -> Path:
    cache=torch_load(features); validate_feature(cache,"expert")
    checkpoint=torch_load(checkpoint_path)
    if checkpoint.get("s0_present") is not False or checkpoint.get("replacement_prior") is not None:
        raise RuntimeError("Not a true drop-s0 checkpoint")
    model=DropS0Corrector(checkpoint["hidden_dim"],checkpoint["maximum_base_correction"],
                          checkpoint["maximum_feedback_correction"]).to(device)
    model.load_state_dict(checkpoint["feedback_selected"]["state"],strict=True)
    inputs=feedback_inputs(cache,checkpoint,args,device)
    pred=pipeline.predict_split(model,cache,*inputs,args.eval_batch_size,device,"true drop-s0 expert")
    base_alpha=float(checkpoint["base_selected"]["alpha"])
    feedback_alpha=float(checkpoint["feedback_selected"]["feedback_alpha"])
    control=pipeline.score_from(cache["anchor"],pred.base.correction,None,base_alpha)
    final=pipeline.score_from(cache["anchor"],pred.base.correction,pred.feedback.correction,base_alpha,feedback_alpha)
    metrics={
        "drop_s0_m1":training_utils.metric_values(cache["base_features"][:,0].numpy(),cache),
        "drop_s0_m12":training_utils.metric_values(cache["anchor"].numpy(),cache),
        "drop_s0_no_feedback":training_utils.metric_values(control.numpy(),cache),
        "drop_s0_full":training_utils.metric_values(final.numpy(),cache),
    }
    full=json.loads(full_report.read_text(encoding="utf-8"))
    full_metrics=full["metrics"]["all_feedback_second_stage"]
    report={"format":CHECKPOINT_FORMAT,"dataset":"expert","ablation":"structural deletion of s0",
            "s0_present":False,"replacement_prior":None,"benchmark_labels_used_for_training_or_selection":False,
            "metrics":metrics,"full_credicap":full_metrics,"checkpoint":str(checkpoint_path)}
    report_path=run_root/"expert"/"drop_s0_expert.metrics.json"; report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# CrediCap true structural drop-`s_0` ablation","",
           "| Method | Expert Tau-c ↑ | MAE ↓ | RMSE ↓ |","|---|---:|---:|---:|"]
    for label,value in (("w/o `s_0`: RCE",metrics["drop_s0_m1"]),
                        ("w/o `s_0`: RCE+CDED",metrics["drop_s0_m12"]),
                        ("w/o `s_0`: RCE+CDED+SDMC",metrics["drop_s0_full"]),
                        ("Full CrediCap (with RefFLEUR `s_0`)",full_metrics)):
        lines.append(f"| {label} | {value['tau_x100']/100:.4f} | {value['mae']:.4f} | {value['rmse']:.4f} |")
    lines += ["","- `s_0` is absent; no CLIP or constant replacement is used.",
              "- All changed-dimensional modules are retrained on Polaris train/validation.",
              "- Expert labels are used only for final metrics."]
    summary=run_root/"drop_s0_expert_summary.md"; summary.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines)); print(f"Summary: {summary}")
    return summary


def run(args) -> None:
    set_seed(args.seed); device=torch.device(args.device)
    # The frozen v9 training helpers call this generic name.  It is intentionally
    # the same 512-row batch used by the original signed-feedback stage.
    args.batch_size = args.feedback_batch_size
    source_root=args.fleur_root/"results"/"trijudge_formal_v2"
    feedback_root=args.fleur_root/"results"/"trijudge_feedback_calibrator_v6"/"feedback"
    run_root=args.fleur_root/"results"/"credicap_true_drop_s0_ablation"
    polaris_path=source_root/"polaris_train_val_cache.pt"; expert_path=source_root/"expert_cache.pt"
    full_report=args.fleur_root/"results"/"trijudge_feedback_signed_error_v9"/"expert"/"expert_feedback_signed_error.metrics.json"
    required=(polaris_path,expert_path,feedback_root/"polaris_train.jsonl",feedback_root/"polaris_val.jsonl",
              feedback_root/"expert.jsonl",full_report)
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    cache=train_reference.load_cache(polaris_path,expected_kind="polaris")
    m1_stage1=train_m1_stage(args,cache,run_root,1,device)
    m1_stage2=train_m1_stage(args,cache,run_root,2,device)
    m2=train_m2(args,cache,run_root,m1_stage2,device)
    feature_root=run_root/"features"; feature_root.mkdir(parents=True,exist_ok=True)
    for split,source,feedback in (("train",polaris_path,feedback_root/"polaris_train.jsonl"),
                                  ("val",polaris_path,feedback_root/"polaris_val.jsonl"),
                                  ("expert",expert_path,feedback_root/"expert.jsonl")):
        prepare_features(args,source,split,feedback,feature_root/f"{split}.pt",m1_stage2,m2,device)
    checkpoint=train_signed(args,run_root,feature_root/"train.pt",feature_root/"val.pt",device)
    evaluate(args,run_root,feature_root/"expert.pt",checkpoint,full_report,device)


def selfcheck(args) -> None:
    set_seed(args.seed); device=torch.device(args.device)
    batch={"image":torch.randn(5,32,device=device),"candidate":torch.randn(5,32,device=device),
           "references":torch.randn(5,4,32,device=device),
           "reference_mask":torch.ones(5,4,dtype=torch.bool,device=device),
           "candidate_length":torch.rand(5,device=device)}
    m1=DropS0M1(32,48,0.0,2).to(device); model=DropS0Full(m1,32,48,0.0,True).to(device)
    output=model(batch)
    if output.router_weights.shape != (5,4) or not torch.isfinite(output.score).all():
        raise RuntimeError("true drop-s0 model self-check failed")
    corrector=DropS0Corrector(64).to(device)
    result=corrector(torch.randn(5,BASE_DIM,device=device),torch.rand(5,7,device=device),
                     torch.rand(5,7,device=device),torch.rand(5,device=device))
    (result.base.correction.mean()+result.feedback.correction.mean()).backward()
    print("TRUE DROP-s0 ABLATION SELF-CHECK: PASS")
    print("RefFLEUR s0 input              : ABSENT")
    print("CLIP/constant replacement      : NONE")
    print("RCE scalar dimensions          : 12")
    print("RCE router experts             : 4")
    print("SDMC base feature dimensions   : 19")


def parser() -> argparse.ArgumentParser:
    root=argparse.ArgumentParser(description=__doc__); commands=root.add_subparsers(dest="command",required=True)
    check=commands.add_parser("selfcheck"); check.add_argument("--seed",type=int,default=2026); check.add_argument("--device",default="cpu"); check.set_defaults(function=selfcheck)
    command=commands.add_parser("run"); command.add_argument("--fleur-root",type=Path,required=True)
    command.add_argument("--device",default="cuda"); command.add_argument("--seed",type=int,default=2026)
    command.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True)
    command.add_argument("--m1-hidden-dim",type=int,default=192); command.add_argument("--dropout",type=float,default=0.10)
    command.add_argument("--m1-epochs",type=int,default=50); command.add_argument("--m2-epochs",type=int,default=60)
    command.add_argument("--minimum-epochs",type=int,default=8); command.add_argument("--patience",type=int,default=10)
    command.add_argument("--m1-batch-size",type=int,default=256); command.add_argument("--m2-batch-size",type=int,default=256)
    command.add_argument("--feedback-batch-size",type=int,default=512); command.add_argument("--eval-batch-size",type=int,default=4096)
    command.add_argument("--pair-batch-size",type=int,default=160); command.add_argument("--m1-lr",type=float,default=3e-4)
    command.add_argument("--m2-lr",type=float,default=1e-4); command.add_argument("--weight-decay",type=float,default=1e-4)
    command.add_argument("--gradient-clip",type=float,default=1.0); command.add_argument("--reference-corruption-weight",type=float,default=0.08)
    command.add_argument("--easy-anchor-quantile",type=float,default=0.35); command.add_argument("--easy-anchor-weight",type=float,default=0.12)
    command.add_argument("--hidden-regularization-weight",type=float,default=0.02); command.add_argument("--rank-loss-weight",type=float,default=0.20)
    command.add_argument("--m2-rank-minimum-gap",type=float,default=0.15)
    command.add_argument("--m2-rank-maximum-per-group",type=int,default=64)
    command.add_argument("--m2-rank-temperature",type=float,default=0.08)
    command.add_argument("--rank-minimum-gap",type=float,default=0.08); command.add_argument("--rank-maximum-per-group",type=int,default=96)
    command.add_argument("--rank-temperature",type=float,default=0.055)
    command.add_argument("--expectation-hidden-dim",type=int,default=128); command.add_argument("--expectation-epochs",type=int,default=8)
    command.add_argument("--expectation-lr",type=float,default=2e-4); command.add_argument("--feedback-hidden-dim",type=int,default=128)
    command.add_argument("--maximum-base-correction",type=float,default=0.05); command.add_argument("--maximum-feedback-correction",type=float,default=0.05)
    command.add_argument("--base-epochs",type=int,default=8); command.add_argument("--feedback-epochs",type=int,default=12)
    command.add_argument("--base-lr",type=float,default=1.5e-4); command.add_argument("--feedback-lr",type=float,default=1.5e-4)
    command.add_argument("--neutral-width",type=float,default=0.015); command.add_argument("--huber-beta",type=float,default=0.04)
    command.add_argument("--supervision-weight",type=float,default=0.75); command.add_argument("--mse-weight",type=float,default=8.0)
    command.add_argument("--mae-weight",type=float,default=0.30); command.add_argument("--rank-weight",type=float,default=0.22)
    command.add_argument("--tail-weight",type=float,default=3.0); command.add_argument("--mean-shift-weight",type=float,default=12.0)
    command.add_argument("--feedback-mean-shift-weight",type=float,default=8.0)
    command.add_argument("--base-alphas",type=float,nargs="+",default=(0.0,0.25,0.5,0.75,1.0))
    command.add_argument("--feedback-alphas",type=float,nargs="+",default=(0.1,0.25,0.5,0.75,1.0))
    command.add_argument("--maximum-feedback-mean-shift",type=float,default=0.005)
    command.add_argument("--minimum-feedback-tau-gain",type=float,default=0.01)
    command.set_defaults(function=run); return root


def main() -> None:
    args=parser().parse_args(); args.function(args)


if __name__ == "__main__": main()

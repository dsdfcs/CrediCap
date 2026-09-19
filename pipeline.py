#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from credicap import training_utils as common
from credicap import frozen_features as frozen
from credicap.score_correction import (
    FEEDBACK_DIM,
    ScoreCorrectionOutput,
    DirectionMagnitudeCorrector,
)


CHECKPOINT_FORMAT = "trijudge-feedback-signed-error-checkpoint-v9"


@dataclass
class SplitPrediction:
    base: ScoreCorrectionOutput
    feedback: ScoreCorrectionOutput


def require_all_feedback(cache: dict, split: str) -> int:
    valid = int((cache["feedback"][:, 7].float() > 0.5).sum())
    total = len(cache["sample_ids"])
    if valid != total:
        raise RuntimeError(
            f"v9 requires structured feedback for every {split} row: "
            f"{valid}/{total}"
        )
    return valid


def build_model(args: argparse.Namespace, device: torch.device):
    return DirectionMagnitudeCorrector(
        hidden_dim=args.hidden_dim,
        maximum_base_correction=args.maximum_base_correction,
        maximum_feedback_correction=args.maximum_feedback_correction,
    ).to(device)


@torch.inference_mode()
def prepare_inputs(
    cache: dict,
    v8_checkpoint: dict,
    batch_size: int,
    device: torch.device,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expectation, _ = frozen.build_frozen_v8(v8_checkpoint, device)
    normalized = common.normalized_base(
        cache,
        v8_checkpoint["normalizer_mean"].float(),
        v8_checkpoint["normalizer_std"].float(),
    )
    rows = []
    loader = DataLoader(
        TensorDataset(torch.arange(len(cache["sample_ids"]))),
        batch_size=batch_size,
        shuffle=False,
    )
    for (indices,) in tqdm(
        loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    ):
        rows.append(
            expectation(normalized.index_select(0, indices).to(device)).cpu()
        )
    expected = torch.cat(rows)
    actual = cache["feedback"][:, :FEEDBACK_DIM].float()
    if expected.shape != actual.shape:
        raise RuntimeError(
            f"Expected-feedback shape drift: {tuple(expected.shape)}"
        )
    for name, value in (
        ("normalized base", normalized),
        ("actual feedback", actual),
        ("expected feedback", expected),
    ):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite {name}")
    return normalized.float(), actual, expected.float()


def error_targets(
    error: torch.Tensor,
    maximum_correction: float,
    neutral_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    error = error.float()
    direction = torch.ones(len(error), dtype=torch.long)
    direction[error < -neutral_width] = 0
    direction[error > neutral_width] = 2
    clipped = error.clamp(-maximum_correction, maximum_correction)
    magnitude = (error.abs() / maximum_correction).clamp(0.0, 1.0)
    return direction, magnitude, clipped


def class_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=3).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = (weights / weights.mean()).clamp(0.25, 4.0)
    return weights.to(device)


def tail_flags(anchor: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    error = (anchor.float() - gold.float()).abs()
    threshold = torch.quantile(error, 0.90)
    return (error >= threshold).float()


def aggregate_outputs(rows: list[ScoreCorrectionOutput]) -> ScoreCorrectionOutput:
    return ScoreCorrectionOutput(
        correction=torch.cat([row.correction for row in rows]),
        direction_logits=torch.cat([row.direction_logits for row in rows]),
        direction_value=torch.cat([row.direction_value for row in rows]),
        magnitude=torch.cat([row.magnitude for row in rows]),
    )


@torch.no_grad()
def predict_split(
    model: DirectionMagnitudeCorrector,
    cache: dict,
    base_features: torch.Tensor,
    actual_feedback: torch.Tensor,
    expected_feedback: torch.Tensor,
    batch_size: int,
    device: torch.device,
    description: str,
) -> SplitPrediction:
    model.eval()
    base_rows = []
    feedback_rows = []
    loader = DataLoader(
        TensorDataset(torch.arange(len(cache["sample_ids"]))),
        batch_size=batch_size,
        shuffle=False,
    )
    for (indices,) in tqdm(
        loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    ):
        output = model(
            base_features.index_select(0, indices).to(device),
            actual_feedback.index_select(0, indices).to(device),
            expected_feedback.index_select(0, indices).to(device),
            cache["anchor"].index_select(0, indices).to(device),
        )
        base_rows.append(
            ScoreCorrectionOutput(
                correction=output.base.correction.cpu(),
                direction_logits=output.base.direction_logits.cpu(),
                direction_value=output.base.direction_value.cpu(),
                magnitude=output.base.magnitude.cpu(),
            )
        )
        feedback_rows.append(
            ScoreCorrectionOutput(
                correction=output.feedback.correction.cpu(),
                direction_logits=output.feedback.direction_logits.cpu(),
                direction_value=output.feedback.direction_value.cpu(),
                magnitude=output.feedback.magnitude.cpu(),
            )
        )
    return SplitPrediction(
        base=aggregate_outputs(base_rows),
        feedback=aggregate_outputs(feedback_rows),
    )


def score_from(
    anchor: torch.Tensor,
    base_correction: torch.Tensor,
    feedback_correction: torch.Tensor | None,
    base_alpha: float,
    feedback_alpha: float = 0.0,
) -> torch.Tensor:
    score = anchor.float() + float(base_alpha) * base_correction.float()
    if feedback_correction is not None:
        score = score + float(feedback_alpha) * feedback_correction.float()
    return score.clamp(0.0, 1.0)


def ranking_loss(
    high_score: torch.Tensor,
    low_score: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.softplus(-(high_score - low_score) / temperature).mean()


def correction_losses(
    output: ScoreCorrectionOutput,
    residual: torch.Tensor,
    maximum_correction: float,
    neutral_width: float,
    direction_weight: torch.Tensor,
    huber_beta: float,
) -> tuple[torch.Tensor, dict]:
    direction, magnitude, clipped = error_targets(
        residual.detach().cpu(), maximum_correction, neutral_width
    )
    direction = direction.to(output.correction.device)
    magnitude = magnitude.to(output.correction.device)
    clipped = clipped.to(output.correction.device)
    direction_loss = F.cross_entropy(
        output.direction_logits, direction, weight=direction_weight
    )
    magnitude_loss = F.smooth_l1_loss(
        output.magnitude, magnitude, beta=huber_beta
    )
    residual_loss = F.smooth_l1_loss(
        output.correction, clipped, beta=huber_beta
    )
    return (
        direction_loss + magnitude_loss + residual_loss,
        {
            "direction": direction_loss,
            "magnitude": magnitude_loss,
            "residual": residual_loss,
        },
    )


def train_base(
    args: argparse.Namespace,
    model: DirectionMagnitudeCorrector,
    train_cache: dict,
    val_cache: dict,
    train_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pairs: np.ndarray,
    pair_weights: np.ndarray,
    device: torch.device,
) -> list[dict]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base_parameters = list(model.base_parameters())
    for parameter in base_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        base_parameters, lr=args.base_lr, weight_decay=args.weight_decay
    )
    base_features = train_inputs[0]
    residual_all = train_cache["gold"].float() - train_cache["anchor"].float()
    direction_labels = error_targets(
        residual_all,
        args.maximum_base_correction,
        args.neutral_width,
    )[0]
    direction_weight = class_weights(direction_labels, device)
    tail = tail_flags(train_cache["anchor"], train_cache["gold"])
    loader = DataLoader(
        TensorDataset(torch.arange(len(base_features))),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    rng = np.random.default_rng(args.seed + 9001)
    cumulative = np.cumsum(pair_weights.astype(np.float64))
    snapshots = []
    print("=" * 120)
    print("PHASE A — TRAIN EQUAL-SOURCE NO-FEEDBACK ERROR PRIOR")
    print("Input: locked M1+CDED-M2 context only")
    print("Output: signed bounded correction around the locked M12 score")
    print("=" * 120, flush=True)
    for epoch in range(1, args.base_epochs + 1):
        model.train()
        progress = tqdm(
            loader,
            desc=f"v9 no-feedback prior epoch {epoch:02d}",
            dynamic_ncols=True,
        )
        running = 0.0
        seen = 0
        for (indices,) in progress:
            count = len(indices)
            positions = common.weighted_pair_positions(rng, cumulative, count)
            pair = torch.from_numpy(pairs[positions]).long()
            high = pair[:, 0]
            low = pair[:, 1]
            optimizer.zero_grad(set_to_none=True)

            _, output = model.forward_base(
                base_features.index_select(0, indices).to(device)
            )
            anchor = train_cache["anchor"].index_select(0, indices).to(device)
            gold = train_cache["gold"].index_select(0, indices).to(device)
            residual = gold - anchor
            supervised, _ = correction_losses(
                output,
                residual,
                args.maximum_base_correction,
                args.neutral_width,
                direction_weight,
                args.huber_beta,
            )
            score = (anchor + output.correction).clamp(0.0, 1.0)
            weight = 1.0 + args.tail_weight * tail.index_select(
                0, indices
            ).to(device)
            mse = (weight * (score - gold).square()).mean()
            mae = F.l1_loss(score, gold)

            _, high_output = model.forward_base(
                base_features.index_select(0, high).to(device)
            )
            _, low_output = model.forward_base(
                base_features.index_select(0, low).to(device)
            )
            high_score = (
                train_cache["anchor"].index_select(0, high).to(device)
                + high_output.correction
            ).clamp(0.0, 1.0)
            low_score = (
                train_cache["anchor"].index_select(0, low).to(device)
                + low_output.correction
            ).clamp(0.0, 1.0)
            rank = ranking_loss(high_score, low_score, args.rank_temperature)
            correction_mean = output.correction.mean()
            loss = (
                args.supervision_weight * supervised
                + args.mse_weight * mse
                + args.mae_weight * mae
                + args.rank_weight * rank
                + args.mean_shift_weight * correction_mean.square()
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite v9 no-feedback loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                base_parameters,
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            running += float(loss.detach()) * count
            seen += count
            progress.set_postfix(loss=f"{running / max(seen, 1):.6f}")

        prediction = predict_split(
            model,
            val_cache,
            *val_inputs,
            args.eval_batch_size,
            device,
            f"v9 no-feedback validation epoch {epoch:02d}",
        )
        val_residual = val_cache["gold"].float() - val_cache["anchor"].float()
        _, _, clipped = error_targets(
            val_residual,
            args.maximum_base_correction,
            args.neutral_width,
        )
        risk_loss = float(
            F.smooth_l1_loss(
                prediction.base.correction, clipped, beta=args.huber_beta
            )
        )
        print(
            f"base epoch={epoch:02d} validation_residual_loss={risk_loss:.7f}",
            flush=True,
        )
        snapshots.append(
            {
                "epoch": epoch,
                "validation_residual_loss": risk_loss,
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            }
        )
    return snapshots


def select_base(
    args: argparse.Namespace,
    snapshots: list[dict],
    model: DirectionMagnitudeCorrector,
    val_cache: dict,
    val_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[dict, list[dict]]:
    anchor = val_cache["anchor"].float()
    anchor_numpy = anchor.numpy()
    anchor_metrics = common.metric_values(anchor_numpy, val_cache)
    candidates = []
    for snapshot in snapshots:
        model.load_state_dict(snapshot["state"], strict=True)
        prediction = predict_split(
            model,
            val_cache,
            *val_inputs,
            args.eval_batch_size,
            device,
            f"v9 base grid epoch {snapshot['epoch']}",
        )
        for alpha in args.base_alphas:
            score = score_from(
                anchor,
                prediction.base.correction,
                None,
                alpha,
            ).numpy()
            metrics = common.metric_values(score, val_cache)
            effect = common.improvement(metrics, anchor_metrics)
            robust = frozen.robustness(score, anchor_numpy, val_cache)
            accepted = bool(effect["all_three_better"] and robust["stable"])
            objective = float(
                metrics["tau_x100"]
                - 28.0 * metrics["mae"]
                - 65.0 * metrics["rmse"]
                - 8.0 * metrics["tail10_rmse"]
                + 0.20 * robust["minimum_tau_gain"]
                + 20.0 * robust["minimum_mae_reduction"]
                + 40.0 * robust["minimum_rmse_reduction"]
            )
            candidates.append(
                {
                    "epoch": snapshot["epoch"],
                    "alpha": float(alpha),
                    "metrics": metrics,
                    "effect": effect,
                    "robustness": robust,
                    "accepted": accepted,
                    "objective": objective,
                    "state": snapshot["state"],
                }
            )
    eligible = [row for row in candidates if row["accepted"]]
    selected = max(eligible or candidates, key=lambda row: row["objective"])
    selected["gate_passed"] = bool(eligible)
    model.load_state_dict(selected["state"], strict=True)
    return selected, candidates


def train_feedback(
    args: argparse.Namespace,
    model: DirectionMagnitudeCorrector,
    base_alpha: float,
    train_cache: dict,
    val_cache: dict,
    train_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pairs: np.ndarray,
    pair_weights: np.ndarray,
    device: torch.device,
) -> list[dict]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    feedback_parameters = list(model.feedback_parameters())
    for parameter in feedback_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        feedback_parameters,
        lr=args.feedback_lr,
        weight_decay=args.weight_decay,
    )
    base_features, actual, expected = train_inputs
    with torch.no_grad():
        base_prediction = predict_split(
            model,
            train_cache,
            *train_inputs,
            args.eval_batch_size,
            device,
            "locked no-feedback prior Polaris train",
        ).base
        control_score = score_from(
            train_cache["anchor"],
            base_prediction.correction,
            None,
            base_alpha,
        )
    residual_all = train_cache["gold"].float() - control_score
    direction_labels = error_targets(
        residual_all,
        args.maximum_feedback_correction,
        args.neutral_width,
    )[0]
    direction_weight = class_weights(direction_labels, device)
    tail = tail_flags(control_score, train_cache["gold"])
    loader = DataLoader(
        TensorDataset(torch.arange(len(base_features))),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    rng = np.random.default_rng(args.seed + 9002)
    cumulative = np.cumsum(pair_weights.astype(np.float64))
    snapshots = []
    print("=" * 120)
    print("PHASE B — TRAIN SIGNED FEEDBACK ERROR CORRECTOR")
    print("Input: M1/M2 context + all 7 actual/expected feedback fields")
    print("Output: LOWER/KEEP/HIGHER plus a bounded correction magnitude")
    print("M1, CDED-M2 and the selected no-feedback prior are frozen")
    print("=" * 120, flush=True)
    for epoch in range(1, args.feedback_epochs + 1):
        model.train()
        progress = tqdm(
            loader,
            desc=f"v9 signed feedback epoch {epoch:02d}",
            dynamic_ncols=True,
        )
        running = 0.0
        seen = 0
        for (indices,) in progress:
            count = len(indices)
            positions = common.weighted_pair_positions(rng, cumulative, count)
            pair = torch.from_numpy(pairs[positions]).long()
            high = pair[:, 0]
            low = pair[:, 1]
            optimizer.zero_grad(set_to_none=True)

            output = model(
                base_features.index_select(0, indices).to(device),
                actual.index_select(0, indices).to(device),
                expected.index_select(0, indices).to(device),
                train_cache["anchor"].index_select(0, indices).to(device),
            )
            anchor = train_cache["anchor"].index_select(0, indices).to(device)
            gold = train_cache["gold"].index_select(0, indices).to(device)
            control = (
                anchor + base_alpha * output.base.correction.detach()
            ).clamp(0.0, 1.0)
            residual = gold - control
            supervised, _ = correction_losses(
                output.feedback,
                residual,
                args.maximum_feedback_correction,
                args.neutral_width,
                direction_weight,
                args.huber_beta,
            )
            final_score = (control + output.feedback.correction).clamp(0.0, 1.0)
            weight = 1.0 + args.tail_weight * tail.index_select(
                0, indices
            ).to(device)
            mse = (weight * (final_score - gold).square()).mean()
            mae = F.l1_loss(final_score, gold)

            high_output = model(
                base_features.index_select(0, high).to(device),
                actual.index_select(0, high).to(device),
                expected.index_select(0, high).to(device),
                train_cache["anchor"].index_select(0, high).to(device),
            )
            low_output = model(
                base_features.index_select(0, low).to(device),
                actual.index_select(0, low).to(device),
                expected.index_select(0, low).to(device),
                train_cache["anchor"].index_select(0, low).to(device),
            )
            high_score = (
                train_cache["anchor"].index_select(0, high).to(device)
                + base_alpha * high_output.base.correction.detach()
                + high_output.feedback.correction
            ).clamp(0.0, 1.0)
            low_score = (
                train_cache["anchor"].index_select(0, low).to(device)
                + base_alpha * low_output.base.correction.detach()
                + low_output.feedback.correction
            ).clamp(0.0, 1.0)
            rank = ranking_loss(high_score, low_score, args.rank_temperature)
            signed_mean = output.feedback.correction.mean()
            loss = (
                args.supervision_weight * supervised
                + args.mse_weight * mse
                + args.mae_weight * mae
                + args.rank_weight * rank
                + args.feedback_mean_shift_weight * signed_mean.square()
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite v9 feedback loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                feedback_parameters,
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            running += float(loss.detach()) * count
            seen += count
            progress.set_postfix(loss=f"{running / max(seen, 1):.6f}")

        prediction = predict_split(
            model,
            val_cache,
            *val_inputs,
            args.eval_batch_size,
            device,
            f"v9 feedback validation epoch {epoch:02d}",
        )
        control = score_from(
            val_cache["anchor"],
            prediction.base.correction,
            None,
            base_alpha,
        )
        residual = val_cache["gold"].float() - control
        _, _, clipped = error_targets(
            residual,
            args.maximum_feedback_correction,
            args.neutral_width,
        )
        residual_loss = float(
            F.smooth_l1_loss(
                prediction.feedback.correction,
                clipped,
                beta=args.huber_beta,
            )
        )
        print(
            f"feedback epoch={epoch:02d} "
            f"validation_residual_loss={residual_loss:.7f} "
            f"mean|delta|={float(prediction.feedback.correction.abs().mean()):.7f}",
            flush=True,
        )
        snapshots.append(
            {
                "epoch": epoch,
                "validation_residual_loss": residual_loss,
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            }
        )
    return snapshots


def select_feedback(
    args: argparse.Namespace,
    snapshots: list[dict],
    model: DirectionMagnitudeCorrector,
    base_selected: dict,
    val_cache: dict,
    val_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[dict, list[dict]]:
    anchor = val_cache["anchor"].float()
    anchor_numpy = anchor.numpy()
    anchor_metrics = common.metric_values(anchor_numpy, val_cache)
    candidates = []
    for snapshot in snapshots:
        model.load_state_dict(snapshot["state"], strict=True)
        prediction = predict_split(
            model,
            val_cache,
            *val_inputs,
            args.eval_batch_size,
            device,
            f"v9 feedback grid epoch {snapshot['epoch']}",
        )
        control = score_from(
            anchor,
            prediction.base.correction,
            None,
            base_selected["alpha"],
        )
        control_numpy = control.numpy()
        control_metrics = common.metric_values(control_numpy, val_cache)
        for alpha in args.feedback_alphas:
            score = score_from(
                anchor,
                prediction.base.correction,
                prediction.feedback.correction,
                base_selected["alpha"],
                alpha,
            )
            score_numpy = score.numpy()
            metrics = common.metric_values(score_numpy, val_cache)
            versus_anchor = common.improvement(metrics, anchor_metrics)
            versus_control = common.improvement(metrics, control_metrics)
            robust = frozen.robustness(score_numpy, control_numpy, val_cache)
            mean_correction = float(
                (alpha * prediction.feedback.correction).mean()
            )
            accepted = bool(
                versus_anchor["all_three_better"]
                and versus_control["all_three_better"]
                and robust["stable"]
                and abs(mean_correction) <= args.maximum_feedback_mean_shift
                and versus_control["tau_gain"] >= args.minimum_feedback_tau_gain
            )
            objective = float(
                metrics["tau_x100"]
                - 30.0 * metrics["mae"]
                - 70.0 * metrics["rmse"]
                - 10.0 * metrics["tail10_rmse"]
                + 0.35 * versus_control["tau_gain"]
                + 25.0 * versus_control["mae_reduction"]
                + 55.0 * versus_control["rmse_reduction"]
                + 0.20 * robust["minimum_tau_gain"]
                - 20.0 * abs(mean_correction)
            )
            candidates.append(
                {
                    "epoch": snapshot["epoch"],
                    "feedback_alpha": float(alpha),
                    "metrics": metrics,
                    "control_metrics": control_metrics,
                    "versus_anchor": versus_anchor,
                    "versus_control": versus_control,
                    "robustness": robust,
                    "mean_feedback_correction": mean_correction,
                    "mean_abs_feedback_correction": float(
                        (alpha * prediction.feedback.correction).abs().mean()
                    ),
                    "accepted": accepted,
                    "objective": objective,
                    "state": snapshot["state"],
                }
            )
    eligible = [row for row in candidates if row["accepted"]]
    selected = max(eligible or candidates, key=lambda row: row["objective"])
    selected["gate_passed"] = bool(eligible)
    model.load_state_dict(selected["state"], strict=True)
    return selected, candidates


def clean_candidate(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "state"}


def train(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        checkpoint = load_checkpoint(args.output)
        print("Valid completed v9 checkpoint already exists")
        common.print_metrics(
            "Locked M1+CDED-M2 validation", checkpoint["anchor_validation"]
        )
        common.print_metrics(
            "Selected no-feedback control",
            checkpoint["base_selected"]["metrics"],
        )
        common.print_metrics(
            "Selected signed-feedback output",
            checkpoint["feedback_selected"]["metrics"],
        )
        return

    common.set_seed(args.seed)
    train_cache = common.torch_load(args.train_features)
    val_cache = common.torch_load(args.val_features)
    common.validate_feature_cache(train_cache, "train")
    common.validate_feature_cache(val_cache, "val")
    train_valid = require_all_feedback(train_cache, "Polaris train")
    val_valid = require_all_feedback(val_cache, "Polaris validation")
    v8_checkpoint = frozen.load_v8_checkpoint(args.v8_checkpoint)
    device = torch.device(args.device)
    train_inputs = prepare_inputs(
        train_cache,
        v8_checkpoint,
        args.eval_batch_size,
        device,
        "expected feedback Polaris train",
    )
    val_inputs = prepare_inputs(
        val_cache,
        v8_checkpoint,
        args.eval_batch_size,
        device,
        "expected feedback Polaris validation",
    )
    pairs, pair_weights, pair_audit = common.make_hard_pairs(
        train_cache,
        args.rank_minimum_gap,
        args.rank_maximum_per_group,
        args.seed,
    )
    anchor_metrics = common.metric_values(
        val_cache["anchor"].numpy(), val_cache
    )
    common.print_metrics("Locked M1+CDED-M2 validation", anchor_metrics)
    print(f"Polaris same-image ranking pairs: {len(pairs):,}")
    model = build_model(args, device)

    base_snapshots = train_base(
        args,
        model,
        train_cache,
        val_cache,
        train_inputs,
        val_inputs,
        pairs,
        pair_weights,
        device,
    )
    base_selected, base_candidates = select_base(
        args,
        base_snapshots,
        model,
        val_cache,
        val_inputs,
        device,
    )
    print("=" * 120)
    print("LOCK NO-FEEDBACK CONTROL ON POLARIS VALIDATION")
    common.print_metrics("M1+CDED-M2", anchor_metrics)
    common.print_metrics("Selected no-feedback prior", base_selected["metrics"])
    print(f"No-feedback gate: {base_selected['gate_passed']}")
    print(f"Selected base alpha: {base_selected['alpha']:.3f}")
    print("=" * 120, flush=True)

    feedback_snapshots = train_feedback(
        args,
        model,
        base_selected["alpha"],
        train_cache,
        val_cache,
        train_inputs,
        val_inputs,
        pairs,
        pair_weights,
        device,
    )
    feedback_selected, feedback_candidates = select_feedback(
        args,
        feedback_snapshots,
        model,
        base_selected,
        val_cache,
        val_inputs,
        device,
    )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "created_unix": time.time(),
        "seed": args.seed,
        "training": "official Polaris train only",
        "selection": "official Polaris validation only",
        "benchmark_labels_used_for_training_or_selection": False,
        "module1_sha256": common.LOCKED_M1_SHA256,
        "module2_sha256": common.LOCKED_M2_SHA256,
        "v8_checkpoint_sha256": common.sha256(args.v8_checkpoint),
        "train_features_sha256": common.sha256(args.train_features),
        "val_features_sha256": common.sha256(args.val_features),
        "all_feedback_required": True,
        "all_feedback_rows_used": True,
        "all_seven_feedback_fields_used": True,
        "hard_feedback_selection": False,
        "uses_v8_correction_direction": False,
        "allows_signed_feedback_correction": True,
        "hidden_dim": args.hidden_dim,
        "maximum_base_correction": args.maximum_base_correction,
        "maximum_feedback_correction": args.maximum_feedback_correction,
        "anchor_validation": anchor_metrics,
        "pair_audit": pair_audit,
        "feedback_rows": {"train": train_valid, "val": val_valid},
        "base_selected": base_selected,
        "feedback_selected": feedback_selected,
        "base_gate_passed": bool(base_selected["gate_passed"]),
        "feedback_gate_passed": bool(feedback_selected["gate_passed"]),
        "base_candidates": [clean_candidate(row) for row in base_candidates],
        "feedback_candidates": [
            clean_candidate(row) for row in feedback_candidates
        ],
    }
    common.save_torch(payload, args.output)
    validation_json = {
        key: value
        for key, value in payload.items()
        if key not in {"base_selected", "feedback_selected"}
    }
    validation_json["base_selected"] = clean_candidate(base_selected)
    validation_json["feedback_selected"] = clean_candidate(feedback_selected)
    common.write_json(args.output.with_suffix(".validation.json"), validation_json)
    print("=" * 120)
    print("SIGNED FEEDBACK ERROR CORRECTOR v9 — POLARIS VALIDATION")
    print("=" * 120)
    common.print_metrics("Locked M1+CDED-M2", anchor_metrics)
    common.print_metrics("No-feedback control", base_selected["metrics"])
    common.print_metrics("All-feedback second stage", feedback_selected["metrics"])
    print(f"FEEDBACK-OVER-CONTROL ALL-THREE GATE: {payload['feedback_gate_passed']}")
    print(f"Base alpha    : {base_selected['alpha']:.3f}")
    print(f"Feedback alpha: {feedback_selected['feedback_alpha']:.3f}")
    print(
        "Mean signed/absolute feedback correction: "
        f"{feedback_selected['mean_feedback_correction']:+.8f} / "
        f"{feedback_selected['mean_abs_feedback_correction']:.8f}"
    )
    print(f"Checkpoint: {args.output}")
    print("=" * 120, flush=True)


def load_checkpoint(path: Path) -> dict:
    checkpoint = common.torch_load(path)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Wrong v9 checkpoint: {checkpoint.get('format')}")
    if checkpoint.get("module1_sha256") != common.LOCKED_M1_SHA256:
        raise RuntimeError("v9 checkpoint is not tied to locked M1")
    if checkpoint.get("module2_sha256") != common.LOCKED_M2_SHA256:
        raise RuntimeError("v9 checkpoint is not tied to locked CDED-M2")
    if checkpoint.get("benchmark_labels_used_for_training_or_selection") is not False:
        raise RuntimeError("Benchmark-selected v9 checkpoint is forbidden")
    for key in (
        "all_feedback_required",
        "all_feedback_rows_used",
        "all_seven_feedback_fields_used",
        "allows_signed_feedback_correction",
    ):
        if checkpoint.get(key) is not True:
            raise RuntimeError(f"Missing v9 contract: {key}")
    if checkpoint.get("hard_feedback_selection") is not False:
        raise RuntimeError("v9 unexpectedly contains feedback selection")
    if checkpoint.get("uses_v8_correction_direction") is not False:
        raise RuntimeError("v9 unexpectedly follows the v8 direction")
    return checkpoint


def evaluate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    if common.sha256(args.v8_checkpoint) != checkpoint["v8_checkpoint_sha256"]:
        raise RuntimeError("Frozen v8 expectation checkpoint changed")
    cache = common.torch_load(args.features)
    common.validate_feature_cache(cache, args.split)
    valid = require_all_feedback(cache, args.split)
    v8_checkpoint = frozen.load_v8_checkpoint(args.v8_checkpoint)
    device = torch.device(args.device)
    inputs = prepare_inputs(
        cache,
        v8_checkpoint,
        args.batch_size,
        device,
        f"expected feedback {args.split}",
    )
    local_args = argparse.Namespace(
        hidden_dim=int(checkpoint["hidden_dim"]),
        maximum_base_correction=float(
            checkpoint["maximum_base_correction"]
        ),
        maximum_feedback_correction=float(
            checkpoint["maximum_feedback_correction"]
        ),
    )
    model = build_model(local_args, device)
    model.load_state_dict(checkpoint["feedback_selected"]["state"], strict=True)
    prediction = predict_split(
        model,
        cache,
        *inputs,
        args.batch_size,
        device,
        f"v9 signed feedback {args.split}",
    )
    base_alpha = float(checkpoint["base_selected"]["alpha"])
    feedback_alpha = float(
        checkpoint["feedback_selected"]["feedback_alpha"]
    )
    control = score_from(
        cache["anchor"],
        prediction.base.correction,
        None,
        base_alpha,
    )
    final = score_from(
        cache["anchor"],
        prediction.base.correction,
        prediction.feedback.correction,
        base_alpha,
        feedback_alpha,
    )
    metrics = {
        "locked_m12_no_feedback": common.metric_values(
            cache["anchor"].numpy(), cache
        ),
        "equal_source_no_feedback_control": common.metric_values(
            control.numpy(), cache
        ),
        "all_feedback_second_stage": common.metric_values(final.numpy(), cache),
    }
    if args.split == "expert":
        expected_m12 = (54.128004, 0.102607, 0.148305)
        for key, target, tolerance in zip(
            ("tau_x100", "mae", "rmse"),
            expected_m12,
            (0.002, 0.00002, 0.00002),
        ):
            if abs(metrics["locked_m12_no_feedback"][key] - target) > tolerance:
                raise RuntimeError(f"Locked M12 Expert drift in {key}")
    versus_m12 = common.improvement(
        metrics["all_feedback_second_stage"],
        metrics["locked_m12_no_feedback"],
    )
    versus_control = common.improvement(
        metrics["all_feedback_second_stage"],
        metrics["equal_source_no_feedback_control"],
    )
    applied_feedback = feedback_alpha * prediction.feedback.correction
    lower = int((applied_feedback < -1.0e-8).sum())
    keep = int((applied_feedback.abs() <= 1.0e-8).sum())
    higher = int((applied_feedback > 1.0e-8).sum())
    diagnostics = {
        "valid_feedback_rows": valid,
        "feedback_rows_processed": len(cache["sample_ids"]),
        "all_feedback_rows_processed": valid == len(cache["sample_ids"]),
        "all_seven_feedback_fields_used": True,
        "direction_counts": {
            "LOWER": lower,
            "KEEP": keep,
            "HIGHER": higher,
        },
        "mean_signed_feedback_correction": float(applied_feedback.mean()),
        "mean_absolute_feedback_correction": float(
            applied_feedback.abs().mean()
        ),
        "maximum_absolute_feedback_correction": float(
            applied_feedback.abs().max()
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / f"{args.split}_feedback_signed_error_results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(cache["records"]):
            row = dict(record)
            row.update(
                {
                    "format": CHECKPOINT_FORMAT,
                    "m12_score": float(cache["anchor"][index]),
                    "no_feedback_control_score": float(control[index]),
                    "all_feedback_second_stage_score": float(final[index]),
                    "base_correction": float(
                        base_alpha * prediction.base.correction[index]
                    ),
                    "feedback_correction": float(applied_feedback[index]),
                    "feedback_direction_value": float(
                        prediction.feedback.direction_value[index]
                    ),
                    "feedback_magnitude": float(
                        prediction.feedback.magnitude[index]
                    ),
                    "all_seven_feedback_fields_processed": True,
                }
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "experiment": CHECKPOINT_FORMAT,
        "dataset": args.split,
        "training": "official Polaris train only",
        "selection": "official Polaris validation only",
        "benchmark_labels_used_for_training_or_selection": False,
        "polaris_feedback_gate": bool(checkpoint["feedback_gate_passed"]),
        "base_alpha": base_alpha,
        "feedback_alpha": feedback_alpha,
        "metrics": metrics,
        "effects": {
            "feedback_vs_locked_m12": versus_m12,
            "feedback_vs_equal_source_no_feedback": versus_control,
        },
        "diagnostics": diagnostics,
        "checkpoint_sha256": common.sha256(args.checkpoint),
        "result": str(result_path),
    }
    report_path = args.output_root / f"{args.split}_feedback_signed_error.metrics.json"
    common.write_json(report_path, report)
    lines = [
        f"# TriJudge Feedback-SignedError v9 — {args.split}",
        "",
        "| Condition | Tau ↑ | MAE ↓ | RMSE ↓ | Bias |",
        "|---|---:|---:|---:|---:|",
        common.metric_row(
            "Locked M1+CDED-M2 / no feedback",
            metrics["locked_m12_no_feedback"],
        ),
        common.metric_row(
            "Equal-source no-feedback second stage",
            metrics["equal_source_no_feedback_control"],
        ),
        common.metric_row(
            "All-feedback signed second stage",
            metrics["all_feedback_second_stage"],
        ),
        "",
        f"- Feedback vs M12: `{json.dumps(versus_m12)}`",
        f"- Feedback vs equal-source no-feedback: `{json.dumps(versus_control)}`",
        f"- Polaris feedback-over-control gate: **{checkpoint['feedback_gate_passed']}**",
        f"- Valid feedback processed: **{valid}/{len(cache['sample_ids'])}**",
        "- All seven structured feedback fields are supplied for every row.",
        f"- Signed actions: `{json.dumps(diagnostics['direction_counts'])}`",
        f"- Mean signed/absolute feedback correction: **{diagnostics['mean_signed_feedback_correction']:+.8f} / {diagnostics['mean_absolute_feedback_correction']:.8f}**",
        "- The second stage may raise or lower a score; it does not follow the old v8 direction.",
        "- No feedback threshold, Top-K selector, abstention, or benchmark-label tuning is used.",
    ]
    summary_path = args.output_root / f"{args.split}_feedback_signed_error_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Result : {result_path}")
    print(f"Report : {report_path}")
    print(f"Summary: {summary_path}")


def audit(args: argparse.Namespace) -> None:
    checkpoint = frozen.load_v8_checkpoint(args.v8_checkpoint)
    for path, split in (
        (args.train_features, "train"),
        (args.val_features, "val"),
    ):
        cache = common.torch_load(path)
        common.validate_feature_cache(cache, split)
        valid = require_all_feedback(cache, f"Polaris {split}")
        print(
            f"{split:<5}: rows={len(cache['sample_ids']):,} "
            f"feedback={valid:,}/{len(cache['sample_ids']):,} "
            f"sha256={common.sha256(path)}"
        )
    print(f"Frozen expectation checkpoint: {common.sha256(args.v8_checkpoint)}")
    print(f"Frozen v8 passed Polaris gate : {checkpoint['feedback_active']}")
    print("V9 SIGNED-FEEDBACK CACHE/LINEAGE AUDIT: PASS")
    print("Expert cache remains unopened until the v9 checkpoint is locked")


def selfcheck(args: argparse.Namespace) -> None:
    common.set_seed(args.seed)
    model = DirectionMagnitudeCorrector(
        hidden_dim=64,
        maximum_base_correction=0.05,
        maximum_feedback_correction=0.05,
    )
    base = torch.randn(8, 21)
    actual = torch.rand(8, 7, requires_grad=True)
    expected = torch.rand(8, 7)
    anchor = torch.linspace(0.1, 0.8, 8)
    output = model(base, actual, expected, anchor)
    probe = (
        output.feedback.correction.sum()
        + output.feedback.direction_logits.square().sum()
        + output.feedback.magnitude.sum()
    )
    probe.backward()
    field_gradient = actual.grad.abs().sum(dim=0)
    if not torch.isfinite(field_gradient).all():
        raise RuntimeError("Non-finite feedback gradient")
    if not torch.all(field_gradient > 0.0):
        raise RuntimeError("At least one structured feedback field is disconnected")
    base_ids = {id(parameter) for parameter in model.base_parameters()}
    feedback_ids = {id(parameter) for parameter in model.feedback_parameters()}
    if base_ids & feedback_ids:
        raise RuntimeError("Base and feedback parameter sets overlap")
    with torch.no_grad():
        model.feedback_head.direction.bias.copy_(torch.tensor([2.0, 0.0, -2.0]))
        lower = model(base, actual, expected, anchor).feedback.correction
        model.feedback_head.direction.bias.copy_(torch.tensor([-2.0, 0.0, 2.0]))
        higher = model(base, actual, expected, anchor).feedback.correction
    if not torch.all(lower < 0.0) or not torch.all(higher > 0.0):
        raise RuntimeError("Signed correction cannot express both directions")
    print("FEEDBACK SIGNED-ERROR v9 SELF-CHECK: PASS")
    print("All seven feedback fields connected : YES")
    print("Signed LOWER and HIGHER actions       : YES")
    print("Base/feedback parameter separation    : YES")
    print("Feedback selector / row rejection     : NONE")
    print(f"Feedback-input gradient sum           : {float(field_gradient.sum()):.9e}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(required=True)

    check = sub.add_parser("selfcheck")
    check.add_argument("--seed", type=int, default=2026)
    check.set_defaults(function=selfcheck)

    cache_audit = sub.add_parser("audit")
    cache_audit.add_argument("--v8-checkpoint", type=Path, required=True)
    cache_audit.add_argument("--train-features", type=Path, required=True)
    cache_audit.add_argument("--val-features", type=Path, required=True)
    cache_audit.set_defaults(function=audit)

    training = sub.add_parser("train")
    training.add_argument("--v8-checkpoint", type=Path, required=True)
    training.add_argument("--train-features", type=Path, required=True)
    training.add_argument("--val-features", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--hidden-dim", type=int, default=128)
    training.add_argument("--maximum-base-correction", type=float, default=0.05)
    training.add_argument("--maximum-feedback-correction", type=float, default=0.05)
    training.add_argument("--base-epochs", type=int, default=8)
    training.add_argument("--feedback-epochs", type=int, default=12)
    training.add_argument("--batch-size", type=int, default=512)
    training.add_argument("--eval-batch-size", type=int, default=4096)
    training.add_argument("--base-lr", type=float, default=1.5e-4)
    training.add_argument("--feedback-lr", type=float, default=1.5e-4)
    training.add_argument("--weight-decay", type=float, default=2.0e-4)
    training.add_argument("--neutral-width", type=float, default=0.015)
    training.add_argument("--huber-beta", type=float, default=0.04)
    training.add_argument("--supervision-weight", type=float, default=0.75)
    training.add_argument("--mse-weight", type=float, default=8.0)
    training.add_argument("--mae-weight", type=float, default=0.30)
    training.add_argument("--rank-weight", type=float, default=0.22)
    training.add_argument("--tail-weight", type=float, default=3.0)
    training.add_argument("--mean-shift-weight", type=float, default=12.0)
    training.add_argument(
        "--feedback-mean-shift-weight", type=float, default=8.0
    )
    training.add_argument("--rank-temperature", type=float, default=0.055)
    training.add_argument("--rank-minimum-gap", type=float, default=0.08)
    training.add_argument("--rank-maximum-per-group", type=int, default=96)
    training.add_argument("--gradient-clip", type=float, default=2.0)
    training.add_argument(
        "--base-alphas",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    training.add_argument(
        "--feedback-alphas",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50, 0.75, 1.0),
    )
    training.add_argument(
        "--maximum-feedback-mean-shift", type=float, default=0.005
    )
    training.add_argument(
        "--minimum-feedback-tau-gain", type=float, default=0.01
    )
    training.add_argument("--seed", type=int, default=2026)
    training.add_argument("--device", default="cuda")
    training.add_argument("--overwrite", action="store_true")
    training.set_defaults(function=train)

    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--v8-checkpoint", type=Path, required=True)
    evaluation.add_argument("--features", type=Path, required=True)
    evaluation.add_argument(
        "--split", choices=("expert", "cf", "composite"), required=True
    )
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output-root", type=Path, required=True)
    evaluation.add_argument("--batch-size", type=int, default=4096)
    evaluation.add_argument("--device", default="cuda")
    evaluation.set_defaults(function=evaluate)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)

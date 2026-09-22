#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from credicap import verification_utils as feedback_common
from credicap import reference_pipeline as m12_runner
from credicap import data_utils as base
from credicap.feature_model import FeatureFusionCalibrator


FEATURE_FORMAT = "trijudge-feedback-calibrator-features-v6"
CHECKPOINT_FORMAT = "trijudge-feedback-calibrator-checkpoint-v6"
FEATURE_NAMES = (
    "reffleur_score",
    "module1_score",
    "m1_stage1_score",
    "m2_hidden_change_rms",
    "structural_strength",
    "consensus_support",
    "dissent_support",
    "consensus_dissent_gap",
    "support_dispersion",
    "reference_disagreement",
    "trust_entropy",
    "consensus_entropy",
    "dissent_entropy",
    "consensus_mass",
    "dissent_mass",
    "router_reffleur",
    "router_stage1",
    "router_image",
    "router_reference",
    "router_interaction",
    "candidate_length",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def source_split(cache_path: Path, split_name: str) -> tuple[dict, dict]:
    cache = torch_load(cache_path)
    if cache.get("format") != base.CACHE_FORMAT:
        raise RuntimeError(f"Wrong source cache format: {cache.get('format')}")
    if cache.get("kind") == "polaris":
        if split_name not in {"train", "val"}:
            raise RuntimeError("Polaris source requires train or val")
        split = cache[split_name]
    elif cache.get("kind") == "benchmark":
        if split_name != cache.get("dataset"):
            raise RuntimeError(
                f"Benchmark contains {cache.get('dataset')}, not {split_name}"
            )
        split = cache["split"]
    else:
        raise RuntimeError(f"Unknown cache kind: {cache.get('kind')}")
    return cache, split


def feature_vector(split: dict, output) -> torch.Tensor:
    if output.router_weights.shape[1] != 5:
        raise RuntimeError("Expected the locked five-expert M1 router")
    values = [
        split["baseline"].float(),
        output.module1_score.float().cpu(),
        output.stage1_score.float().cpu(),
        output.hidden_change_rms.float().cpu(),
        output.structural_strength.float().cpu(),
        output.consensus_support.float().cpu(),
        output.dissent_support.float().cpu(),
        output.consensus_dissent_gap.float().cpu(),
        output.support_dispersion.float().cpu(),
        output.reference_disagreement.float().cpu(),
        output.trust_entropy.float().cpu(),
        output.consensus_entropy.float().cpu(),
        output.dissent_entropy.float().cpu(),
        output.consensus_mass.float().cpu(),
        output.dissent_mass.float().cpu(),
    ]
    matrix = torch.stack(values, dim=1)
    return torch.cat(
        [
            matrix,
            output.router_weights.float().cpu(),
            split["candidate_length"].float()[:, None],
        ],
        dim=1,
    )


@torch.inference_mode()
def extract_locked_features(
    model,
    split: dict,
    batch_size: int,
    device: torch.device,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchors: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    loader = DataLoader(
        TensorDataset(torch.arange(len(split["sample_ids"]))),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    for (indices,) in tqdm(loader, desc=description, dynamic_ncols=True):
        local_split = {
            key: value.index_select(0, indices)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in split.items()
        }
        batch = base.make_batch(split, indices, device)
        output = model(batch)
        anchors.append(output.score.float().cpu())
        features.append(feature_vector(local_split, output))
    anchor = torch.cat(anchors)
    feature = torch.cat(features)
    if feature.shape != (len(split["sample_ids"]), len(FEATURE_NAMES)):
        raise RuntimeError(
            f"Feature contract mismatch: {tuple(feature.shape)}"
        )
    return anchor, feature


def validate_feature_cache(
    cache: dict,
    required_split: str | None = None,
) -> None:
    if cache.get("format") != FEATURE_FORMAT:
        raise RuntimeError(f"Wrong feature cache format: {cache.get('format')}")
    if required_split is not None and cache.get("split") != required_split:
        raise RuntimeError(
            f"Feature cache split mismatch: {cache.get('split')} != "
            f"{required_split}"
        )
    if tuple(cache.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError("Feature-name contract mismatch")
    if tuple(cache.get("feedback_fields", ())) != (
        feedback_common.FIELDS + ("VALID",)
    ):
        raise RuntimeError("Feedback-field contract mismatch")
    if cache.get("module1_sha256") != m12_runner.LOCKED_M1_SHA256:
        raise RuntimeError("Feature cache is not tied to locked M1")
    if cache.get("module2_sha256") != m12_runner.LOCKED_M2_SHA256:
        raise RuntimeError("Feature cache is not tied to locked CDED-M2")
    count = len(cache["sample_ids"])
    for key in ("records", "groups", "gold", "anchor", "base_features", "feedback"):
        if len(cache[key]) != count:
            raise RuntimeError(f"Feature cache length mismatch: {key}")
    if cache["base_features"].shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("Base feature dimension mismatch")
    if cache["feedback"].shape[1] != len(feedback_common.FIELDS) + 1:
        raise RuntimeError("Feedback feature dimension mismatch")
    if len(set(cache["sample_ids"])) != count:
        raise RuntimeError("Duplicate sample IDs in feature cache")
    for key in ("gold", "anchor", "base_features", "feedback"):
        value = cache[key]
        if not torch.isfinite(value.float()).all():
            bad = int((~torch.isfinite(value.float())).sum())
            raise RuntimeError(
                f"Feature cache contains {bad} non-finite values in {key}"
            )


def prepare(args: argparse.Namespace) -> None:
    m12_runner.validate_hash(
        args.module1_checkpoint,
        m12_runner.LOCKED_M1_SHA256,
        "LOCKED M1",
    )
    m12_runner.validate_hash(
        args.module2_checkpoint,
        m12_runner.LOCKED_M2_SHA256,
        "LOCKED CDED-M2",
    )
    if args.output.exists() and not args.overwrite:
        existing = torch_load(args.output)
        invalid_reason: str | None = None
        try:
            validate_feature_cache(existing, args.split)
        except RuntimeError as exc:
            invalid_reason = str(exc)
        if invalid_reason is None:
            expected = {
                "source_cache_sha256": sha256(args.source_cache),
                "feedback_sha256": sha256(args.feedback),
            }
            stale = [
                key
                for key, value in expected.items()
                if existing.get(key) != value
            ]
            if not stale:
                print(f"Exact valid feature cache already exists: {args.output}")
                return
            invalid_reason = f"stale fields: {', '.join(stale)}"
        if not args.refresh_stale:
            raise RuntimeError(
                f"Existing feature cache cannot be reused ({invalid_reason}); "
                "rerun prepare with --refresh-stale"
            )
        print(
            f"Refreshing feature cache ({invalid_reason}): {args.output}",
            flush=True,
        )
    source, split = source_split(args.source_cache, args.split)
    feedback_rows = feedback_common.read_jsonl_resume(args.feedback)
    feedback_common.validate_rows(feedback_rows, split["sample_ids"])
    feedback_matrix = feedback_common.matrix_from_rows(
        feedback_rows, split["sample_ids"]
    )
    device = torch.device(args.device)
    model = m12_runner.build_model(
        args.module1_checkpoint,
        args.module2_checkpoint,
        "m12",
        device,
    )
    anchor, frozen_features = extract_locked_features(
        model,
        split,
        args.batch_size,
        device,
        f"Locked M1+CDED-M2 features {args.split}",
    )
    payload = {
        "format": FEATURE_FORMAT,
        "created_unix": time.time(),
        "split": args.split,
        "source_cache": str(args.source_cache),
        "source_signature": source.get("signature"),
        "source_cache_sha256": sha256(args.source_cache),
        "feedback_file": str(args.feedback),
        "feedback_sha256": sha256(args.feedback),
        "module1_sha256": m12_runner.LOCKED_M1_SHA256,
        "module2_sha256": m12_runner.LOCKED_M2_SHA256,
        "feature_names": FEATURE_NAMES,
        "feedback_fields": feedback_common.FIELDS + ("VALID",),
        "sample_ids": list(split["sample_ids"]),
        "groups": list(split["groups"]),
        "records": list(split["records"]),
        "gold": split["gold"].float().cpu(),
        "anchor": anchor.float().cpu(),
        "base_features": frozen_features.float().cpu(),
        "feedback": torch.from_numpy(feedback_matrix).float(),
        "strict_feedback": int(feedback_matrix[:, -1].sum()),
    }
    validate_feature_cache(payload, args.split)
    save_torch(payload, args.output)
    print(f"Feedback feature cache saved: {args.output}")
    print(f"Rows            : {len(payload['sample_ids'])}")
    print(f"Strict feedback : {payload['strict_feedback']}")
    print(f"Feature shape   : {tuple(payload['base_features'].shape)}")


def metric_values(prediction: np.ndarray, cache: dict) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    gold = cache["gold"].numpy().astype(np.float64)
    if prediction.shape != gold.shape:
        raise RuntimeError("Prediction/gold shape mismatch")
    tau_prediction: list[float] = []
    tau_gold: list[float] = []
    for score, record in zip(prediction.tolist(), cache["records"]):
        ratings = [float(value) for value in record.get("human_ratings", [])]
        if not ratings:
            ratings = [float(record.get("human_score_normalized", 0.0))]
        tau_prediction.extend([float(score)] * len(ratings))
        tau_gold.extend(ratings)
    variant = "b" if cache["split"] in {"train", "val", "cf"} else "c"
    tau = scipy.stats.kendalltau(
        tau_prediction,
        tau_gold,
        variant=variant,
    ).statistic
    difference = prediction - gold
    return {
        "split": cache["split"],
        "n": len(prediction),
        "tau_variant": variant,
        "tau_x100": float(100.0 * tau),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "bias": float(np.mean(difference)),
    }


def metric_objective(values: dict) -> float:
    return float(values["tau_x100"] - 60.0 * values["mae"] - 30.0 * values["rmse"])


def print_metrics(label: str, values: dict) -> None:
    print(
        f"{label:<34} Tau-{values['tau_variant']}={values['tau_x100']:.6f}  "
        f"MAE={values['mae']:.6f}  RMSE={values['rmse']:.6f}  "
        f"bias={values['bias']:+.6f}",
        flush=True,
    )


def all_three_better(candidate: dict, control: dict) -> bool:
    return bool(
        candidate["tau_x100"] > control["tau_x100"]
        and candidate["mae"] < control["mae"]
        and candidate["rmse"] < control["rmse"]
    )


def make_pairs(
    cache: dict,
    minimum_gap: float,
    maximum_per_group: int,
    seed: int,
) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(cache["groups"]):
        groups[str(group)].append(index)
    gold = cache["gold"].numpy()
    rng = np.random.default_rng(seed)
    result: list[tuple[int, int]] = []
    for indices in groups.values():
        local: list[tuple[int, int]] = []
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                difference = float(gold[left] - gold[right])
                if abs(difference) < minimum_gap:
                    continue
                local.append((left, right) if difference > 0 else (right, left))
        if len(local) > maximum_per_group:
            chosen = rng.choice(
                len(local), size=maximum_per_group, replace=False
            )
            local = [local[int(index)] for index in chosen]
        result.extend(local)
    if not result:
        raise RuntimeError("No same-image Polaris ranking pairs were created")
    return np.asarray(result, dtype=np.int64)


def normalized_base(
    cache: dict,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (cache["base_features"].float() - mean) / std


@torch.no_grad()
def predict_calibrator(
    model: FeatureFusionCalibrator,
    cache: dict,
    base_values: torch.Tensor,
    feedback_values: torch.Tensor,
    batch_size: int,
    device: torch.device,
    description: str,
) -> tuple[np.ndarray, dict]:
    model.eval()
    scores: list[torch.Tensor] = []
    corrections: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    loader = DataLoader(
        TensorDataset(torch.arange(len(cache["sample_ids"]))),
        batch_size=batch_size,
        shuffle=False,
    )
    for (indices,) in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
        output = model(
            cache["anchor"].index_select(0, indices).to(device),
            base_values.index_select(0, indices).to(device),
            feedback_values.index_select(0, indices).to(device),
        )
        scores.append(output.score.float().cpu())
        corrections.append(output.correction.float().cpu())
        gates.append(output.gate.float().cpu())
    score = torch.cat(scores).numpy()
    correction = torch.cat(corrections).numpy()
    gate = torch.cat(gates).numpy()
    return score, {
        "mean_abs_correction": float(np.abs(correction).mean()),
        "maximum_abs_correction": float(np.abs(correction).max()),
        "mean_gate": float(gate.mean()),
        "changed_rows": int(np.count_nonzero(np.abs(correction) > 1.0e-7)),
    }


def train_candidate(
    args: argparse.Namespace,
    train_cache: dict,
    val_cache: dict,
    train_base: torch.Tensor,
    val_base: torch.Tensor,
    train_feedback: torch.Tensor,
    val_feedback: torch.Tensor,
    pairs: np.ndarray,
    branch: str,
    maximum_correction: float,
    rank_weight: float,
    candidate_index: int,
    device: torch.device,
) -> dict:
    # One published seed, no seed ensemble.  Re-seeding each configuration
    # makes the Polaris validation comparison exactly reproducible.
    set_seed(args.seed)
    model = FeatureFusionCalibrator(
        base_dim=train_base.shape[1],
        feedback_dim=train_feedback.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        maximum_correction=maximum_correction,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # This calibration MLP is tiny. FP16 loss scaling gives no meaningful
    # memory benefit and can overflow its zero-initialized correction head on
    # the first backward pass, so all calibration optimization is FP32.
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    point_loader = DataLoader(
        TensorDataset(torch.arange(len(train_cache["sample_ids"]))),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    pair_rng = np.random.default_rng(args.seed + 1709)
    best: dict | None = None
    print("=" * 116)
    print(
        f"TRAIN {branch.upper()} candidate={candidate_index:02d} "
        f"max_correction={maximum_correction:.3f} rank_weight={rank_weight:.3f}"
    )
    print("=" * 116, flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        progress = tqdm(
            point_loader,
            desc=f"{branch} c{candidate_index:02d} epoch {epoch:02d}",
            dynamic_ncols=True,
        )
        for (indices,) in progress:
            count = len(indices)
            pair_positions = pair_rng.integers(0, len(pairs), size=count)
            chosen_pairs = torch.from_numpy(pairs[pair_positions]).long()
            high = chosen_pairs[:, 0]
            low = chosen_pairs[:, 1]
            point_feedback = train_feedback.index_select(0, indices)
            high_feedback = train_feedback.index_select(0, high)
            low_feedback = train_feedback.index_select(0, low)
            if branch == "feedback" and args.feedback_dropout > 0:
                keep = (
                    torch.rand(count, 1) >= args.feedback_dropout
                ).float()
                point_feedback = point_feedback * keep
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=False):
                point = model(
                    train_cache["anchor"].index_select(0, indices).to(device),
                    train_base.index_select(0, indices).to(device),
                    point_feedback.to(device),
                )
                high_output = model(
                    train_cache["anchor"].index_select(0, high).to(device),
                    train_base.index_select(0, high).to(device),
                    high_feedback.to(device),
                )
                low_output = model(
                    train_cache["anchor"].index_select(0, low).to(device),
                    train_base.index_select(0, low).to(device),
                    low_feedback.to(device),
                )
                gold = train_cache["gold"].index_select(0, indices).to(device)
                regression = F.smooth_l1_loss(
                    point.score, gold, beta=args.huber_beta
                )
                absolute = F.l1_loss(point.score, gold)
                ranking = F.softplus(
                    -(high_output.score - low_output.score)
                    / args.rank_temperature
                ).mean()
                gold_residual = gold - train_cache["anchor"].index_select(
                    0, indices
                ).to(device)
                active = gold_residual.abs() >= args.direction_minimum
                if active.any():
                    direction = F.softplus(
                        -torch.sign(gold_residual[active])
                        * point.correction[active]
                        / args.direction_temperature
                    ).mean()
                else:
                    direction = point.correction.sum() * 0.0
                regularization = point.correction.abs().mean()
                loss = (
                    regression
                    + args.mae_weight * absolute
                    + rank_weight * ranking
                    + args.direction_weight * direction
                    + args.correction_weight * regularization
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * count
            seen += count
            progress.set_postfix(loss=f"{running / max(seen, 1):.5f}")
        prediction, diagnostics = predict_calibrator(
            model,
            val_cache,
            val_base,
            val_feedback,
            args.eval_batch_size,
            device,
            f"{branch} c{candidate_index:02d} validation",
        )
        anchor_values = val_cache["anchor"].numpy()
        alpha_rows = []
        for alpha in args.blend_alphas:
            blended = anchor_values + float(alpha) * (prediction - anchor_values)
            values = metric_values(blended, val_cache)
            alpha_rows.append(
                {
                    "alpha": float(alpha),
                    "metrics": values,
                    "objective": metric_objective(values),
                }
            )
        selected_alpha = max(alpha_rows, key=lambda row: row["objective"])
        metrics = selected_alpha["metrics"]
        objective = selected_alpha["objective"]
        diagnostics = dict(diagnostics)
        diagnostics["blend_alpha"] = selected_alpha["alpha"]
        diagnostics["mean_abs_deployed_correction"] = (
            diagnostics["mean_abs_correction"] * selected_alpha["alpha"]
        )
        print(
            f"epoch={epoch:02d} objective={objective:.6f} "
            f"alpha={selected_alpha['alpha']:.2f} "
            f"mean|correction|={diagnostics['mean_abs_deployed_correction']:.6f}"
        )
        print_metrics("Polaris validation", metrics)
        if best is None or objective > best["objective"]:
            best = {
                "epoch": epoch,
                "objective": objective,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "configuration": {
                    "branch": branch,
                    "maximum_correction": maximum_correction,
                    "rank_weight": rank_weight,
                    "candidate_index": candidate_index,
                    "blend_alpha": selected_alpha["alpha"],
                },
            }
    if best is None:
        raise RuntimeError("Candidate training produced no checkpoint")
    return best


def branch_model(checkpoint: dict, branch: str, device: torch.device):
    selected = checkpoint[branch]
    configuration = selected["configuration"]
    model = FeatureFusionCalibrator(
        base_dim=int(checkpoint["base_dim"]),
        feedback_dim=int(checkpoint["feedback_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        maximum_correction=float(configuration["maximum_correction"]),
    ).to(device)
    model.load_state_dict(selected["state"], strict=True)
    model.eval()
    return model


def train(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        existing = load_checkpoint(args.output)
        print("Valid completed calibrator checkpoint already exists")
        print_metrics("Locked M1+CDED-M2", existing["anchor_validation"])
        print_metrics("Equal-capacity no-feedback", existing["nofeedback"]["metrics"])
        print_metrics("Structured-feedback calibrator", existing["feedback"]["metrics"])
        print(f"FEEDBACK ALL-THREE GATE: {existing['feedback_active']}")
        return
    train_cache = torch_load(args.train_features)
    val_cache = torch_load(args.val_features)
    validate_feature_cache(train_cache)
    validate_feature_cache(val_cache)
    if train_cache["split"] != "train" or val_cache["split"] != "val":
        raise RuntimeError("Training requires Polaris train/val feature caches")
    if train_cache["feature_names"] != val_cache["feature_names"]:
        raise RuntimeError("Train/val base feature contracts differ")
    if args.amp:
        print(
            "Calibration AMP request ignored: training is forced to FP32 "
            "to prevent non-finite scaled gradients.",
            flush=True,
        )
    mean = train_cache["base_features"].mean(dim=0)
    std = train_cache["base_features"].std(dim=0).clamp_min(1.0e-4)
    train_base = normalized_base(train_cache, mean, std)
    val_base = normalized_base(val_cache, mean, std)
    real_train_feedback = train_cache["feedback"].float()
    real_val_feedback = val_cache["feedback"].float()
    for label, values in (
        ("Polaris train", real_train_feedback),
        ("Polaris val", real_val_feedback),
    ):
        unique_rows = int(torch.unique(values, dim=0).shape[0])
        strict_rate = float(values[:, -1].mean())
        content_std = float(values[:, :-1].std(dim=0).mean())
        print(
            f"{label} feedback audit: unique={unique_rows} "
            f"strict={strict_rate:.2%} mean_field_std={content_std:.6f}",
            flush=True,
        )
        if unique_rows < 8 or content_std < 0.02:
            raise RuntimeError(
                f"{label} feedback collapsed to nearly constant output; "
                "do not train a fake feedback module"
            )
    zero_train_feedback = torch.zeros_like(real_train_feedback)
    zero_val_feedback = torch.zeros_like(real_val_feedback)
    pairs = make_pairs(
        train_cache,
        args.rank_minimum_gap,
        args.rank_maximum_per_group,
        args.seed,
    )
    print(f"Polaris same-image ranking pairs: {len(pairs):,}")
    anchor_metrics = metric_values(val_cache["anchor"].numpy(), val_cache)
    print_metrics("Locked M1+CDED-M2 validation", anchor_metrics)
    device = torch.device(args.device)
    candidates: dict[str, list[dict]] = {"nofeedback": [], "feedback": []}
    grid = [
        (float(maximum), float(rank))
        for maximum in args.maximum_corrections
        for rank in args.rank_weights
    ]
    for branch in ("nofeedback", "feedback"):
        train_feedback = (
            zero_train_feedback if branch == "nofeedback" else real_train_feedback
        )
        val_feedback = (
            zero_val_feedback if branch == "nofeedback" else real_val_feedback
        )
        for index, (maximum, rank) in enumerate(grid, start=1):
            candidates[branch].append(
                train_candidate(
                    args,
                    train_cache,
                    val_cache,
                    train_base,
                    val_base,
                    train_feedback,
                    val_feedback,
                    pairs,
                    branch,
                    maximum,
                    rank,
                    index,
                    device,
                )
            )
    nofeedback = max(candidates["nofeedback"], key=lambda row: row["objective"])
    eligible_feedback = [
        row
        for row in candidates["feedback"]
        if all_three_better(row["metrics"], anchor_metrics)
        and all_three_better(row["metrics"], nofeedback["metrics"])
    ]
    feedback = max(
        eligible_feedback or candidates["feedback"],
        key=lambda row: row["objective"],
    )
    feedback_active = bool(eligible_feedback)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "created_unix": time.time(),
        "seed": args.seed,
        "seed_ensemble": False,
        "five_fold_oof": False,
        "training": "official Polaris train only",
        "selection": "official Polaris validation only",
        "benchmark_labels_used_for_training_or_selection": False,
        "module1_sha256": m12_runner.LOCKED_M1_SHA256,
        "module2_sha256": m12_runner.LOCKED_M2_SHA256,
        "train_features_sha256": sha256(args.train_features),
        "val_features_sha256": sha256(args.val_features),
        "feature_names": FEATURE_NAMES,
        "feedback_fields": feedback_common.FIELDS + ("VALID",),
        "base_dim": len(FEATURE_NAMES),
        "feedback_dim": len(feedback_common.FIELDS) + 1,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "normalizer_mean": mean,
        "normalizer_std": std,
        "anchor_validation": anchor_metrics,
        "nofeedback": nofeedback,
        "feedback": feedback,
        "feedback_active": feedback_active,
        "feedback_gate": (
            "PASS: feedback beats M12 and equal-capacity no-feedback on Tau/MAE/RMSE"
            if feedback_active
            else "FAIL: raw feedback model retained for diagnosis; deployment falls back to exact M12"
        ),
        "candidate_metrics": {
            branch: [
                {
                    "epoch": row["epoch"],
                    "objective": row["objective"],
                    "metrics": row["metrics"],
                    "diagnostics": row["diagnostics"],
                    "configuration": row["configuration"],
                }
                for row in rows
            ]
            for branch, rows in candidates.items()
        },
    }
    save_torch(payload, args.output)
    report = {
        key: value
        for key, value in payload.items()
        if key not in {"normalizer_mean", "normalizer_std", "nofeedback", "feedback"}
    }
    report["nofeedback"] = {
        key: value for key, value in nofeedback.items() if key != "state"
    }
    report["feedback"] = {
        key: value for key, value in feedback.items() if key != "state"
    }
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("=" * 116)
    print("POLARIS VALIDATION — SELECTED SINGLE CHECKPOINT")
    print("=" * 116)
    print_metrics("Locked M1+CDED-M2", anchor_metrics)
    print_metrics("Equal-capacity no-feedback", nofeedback["metrics"])
    print_metrics("Structured-feedback calibrator", feedback["metrics"])
    print(f"FEEDBACK ALL-THREE GATE: {feedback_active}")
    print(f"Checkpoint: {args.output}")
    print("=" * 116)


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch_load(path)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Wrong checkpoint format: {checkpoint.get('format')}")
    if checkpoint.get("module1_sha256") != m12_runner.LOCKED_M1_SHA256:
        raise RuntimeError("Calibrator is not tied to locked M1")
    if checkpoint.get("module2_sha256") != m12_runner.LOCKED_M2_SHA256:
        raise RuntimeError("Calibrator is not tied to locked M2")
    if checkpoint.get("benchmark_labels_used_for_training_or_selection") is not False:
        raise RuntimeError("Benchmark-label-selected checkpoint is forbidden")
    return checkpoint


def improvement(candidate: dict, control: dict) -> dict:
    return {
        "tau_gain": candidate["tau_x100"] - control["tau_x100"],
        "mae_reduction": control["mae"] - candidate["mae"],
        "rmse_reduction": control["rmse"] - candidate["rmse"],
        "all_three_better": all_three_better(candidate, control),
    }


def metric_row(label: str, values: dict) -> str:
    return (
        f"| {label} | {values['tau_x100']:.6f} | {values['mae']:.6f} | "
        f"{values['rmse']:.6f} | {values['bias']:+.6f} |"
    )


def evaluate(args: argparse.Namespace) -> None:
    cache = torch_load(args.features)
    validate_feature_cache(cache)
    checkpoint = load_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    mean = checkpoint["normalizer_mean"].float()
    std = checkpoint["normalizer_std"].float()
    normalized = normalized_base(cache, mean, std)
    zero_feedback = torch.zeros_like(cache["feedback"])
    nofeedback_model = branch_model(checkpoint, "nofeedback", device)
    feedback_model = branch_model(checkpoint, "feedback", device)
    nofeedback_score, nofeedback_diagnostics = predict_calibrator(
        nofeedback_model,
        cache,
        normalized,
        zero_feedback,
        args.batch_size,
        device,
        "Equal-capacity no-feedback Expert",
    )
    feedback_score, feedback_diagnostics = predict_calibrator(
        feedback_model,
        cache,
        normalized,
        cache["feedback"].float(),
        args.batch_size,
        device,
        "Structured-feedback Expert",
    )
    anchor = cache["anchor"].numpy()
    nofeedback_alpha = float(
        checkpoint["nofeedback"]["configuration"]["blend_alpha"]
    )
    feedback_alpha = float(
        checkpoint["feedback"]["configuration"]["blend_alpha"]
    )
    nofeedback_score = anchor + nofeedback_alpha * (nofeedback_score - anchor)
    feedback_score = anchor + feedback_alpha * (feedback_score - anchor)
    nofeedback_diagnostics["blend_alpha"] = nofeedback_alpha
    feedback_diagnostics["blend_alpha"] = feedback_alpha
    active = bool(checkpoint["feedback_active"])
    deployed_score = feedback_score if active else anchor.copy()
    metrics = {
        "locked_m12": metric_values(anchor, cache),
        "nofeedback_control": metric_values(nofeedback_score, cache),
        "raw_feedback": metric_values(feedback_score, cache),
        "deployed": metric_values(deployed_score, cache),
    }
    if cache["split"] == "expert":
        m12_runner.check_expected("expert", "m12", metrics["locked_m12"])
    effects = {
        "feedback_vs_m12": improvement(metrics["raw_feedback"], metrics["locked_m12"]),
        "feedback_vs_nofeedback": improvement(
            metrics["raw_feedback"], metrics["nofeedback_control"]
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / f"{cache['split']}_feedback_calibrator_results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(cache["records"]):
            row = dict(record)
            row.update(
                {
                    "format": CHECKPOINT_FORMAT,
                    "m12_score": float(anchor[index]),
                    "nofeedback_score": float(nofeedback_score[index]),
                    "raw_feedback_score": float(feedback_score[index]),
                    "deployed_score": float(deployed_score[index]),
                    "feedback_gate_passed_on_polaris": active,
                    "feedback_vector": cache["feedback"][index].tolist(),
                }
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "experiment": CHECKPOINT_FORMAT,
        "dataset": cache["split"],
        "training": "official Polaris train only",
        "selection": "official Polaris validation only",
        "benchmark_labels_used_for_training_or_selection": False,
        "five_fold_oof": False,
        "seed_ensemble": False,
        "seed": checkpoint["seed"],
        "feedback_active_from_polaris_validation": active,
        "metrics": metrics,
        "effects": effects,
        "diagnostics": {
            "nofeedback": nofeedback_diagnostics,
            "feedback": feedback_diagnostics,
            "strict_feedback": cache["strict_feedback"],
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "result": str(result_path),
    }
    report_path = args.output_root / f"{cache['split']}_feedback_calibrator.metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        f"# TriJudge Feedback-Calibrator v6 — {cache['split']}",
        "",
        "| Condition | Tau ↑ | MAE ↓ | RMSE ↓ | Bias |",
        "|---|---:|---:|---:|---:|",
        metric_row("Locked M1+CDED-M2", metrics["locked_m12"]),
        metric_row("Equal-capacity no-feedback control", metrics["nofeedback_control"]),
        metric_row("Raw structured-feedback model", metrics["raw_feedback"]),
        metric_row("Deployed output", metrics["deployed"]),
        "",
        f"- Feedback vs M12: `{json.dumps(effects['feedback_vs_m12'])}`",
        f"- Feedback vs no-feedback control: `{json.dumps(effects['feedback_vs_nofeedback'])}`",
        f"- Polaris all-three gate: **{active}**",
        f"- Strict structured feedback: **{cache['strict_feedback']}/{len(cache['sample_ids'])}**",
        "- Expert labels were used only here for final metrics.",
    ]
    summary_path = args.output_root / f"{cache['split']}_feedback_calibrator_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Result : {result_path}")
    print(f"Report : {report_path}")
    print(f"Summary: {summary_path}")


def selfcheck(args: argparse.Namespace) -> None:
    feedback_common.selfcheck()
    set_seed(args.seed)
    model = FeatureFusionCalibrator(
        base_dim=len(FEATURE_NAMES),
        feedback_dim=len(feedback_common.FIELDS) + 1,
        hidden_dim=32,
        dropout=0.0,
        maximum_correction=0.05,
    )
    anchor = torch.tensor([0.2, 0.7])
    base_features = torch.randn(2, len(FEATURE_NAMES))
    feedback = torch.rand(2, len(feedback_common.FIELDS) + 1)
    output = model(anchor, base_features, feedback)
    if not torch.allclose(output.score, anchor, atol=1.0e-7):
        raise RuntimeError("Calibrator must initialize at exact M12 identity")
    loss = output.score.sum() + output.gate.sum()
    loss.backward()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not math.isfinite(gradient) or gradient <= 0:
        raise RuntimeError("Calibrator gradient self-check failed")
    print("FEEDBACK CALIBRATOR v6 SELF-CHECK: PASS")
    print(f"Identity max error: {(output.score-anchor).abs().max().item():.3e}")
    print(f"Gradient sum      : {gradient:.6e}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(required=True)

    check = sub.add_parser("selfcheck")
    check.add_argument("--seed", type=int, default=2026)
    check.set_defaults(function=selfcheck)

    preparation = sub.add_parser("prepare")
    preparation.add_argument("--source-cache", type=Path, required=True)
    preparation.add_argument(
        "--split",
        choices=("train", "val", "expert", "cf", "composite"),
        required=True,
    )
    preparation.add_argument("--feedback", type=Path, required=True)
    preparation.add_argument("--module1-checkpoint", type=Path, required=True)
    preparation.add_argument("--module2-checkpoint", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--batch-size", type=int, default=1024)
    preparation.add_argument("--device", default="cuda")
    preparation.add_argument("--overwrite", action="store_true")
    preparation.add_argument("--refresh-stale", action="store_true")
    preparation.set_defaults(function=prepare)

    training = sub.add_parser("train")
    training.add_argument("--train-features", type=Path, required=True)
    training.add_argument("--val-features", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--maximum-corrections", type=float, nargs="+", default=(0.02, 0.04, 0.06))
    training.add_argument("--rank-weights", type=float, nargs="+", default=(0.20, 0.45, 0.70))
    training.add_argument(
        "--blend-alphas",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50, 0.75, 1.00),
    )
    training.add_argument("--epochs", type=int, default=18)
    training.add_argument("--batch-size", type=int, default=512)
    training.add_argument("--eval-batch-size", type=int, default=2048)
    training.add_argument("--hidden-dim", type=int, default=128)
    training.add_argument("--dropout", type=float, default=0.10)
    training.add_argument("--feedback-dropout", type=float, default=0.10)
    training.add_argument("--lr", type=float, default=2.0e-4)
    training.add_argument("--weight-decay", type=float, default=1.0e-4)
    training.add_argument("--huber-beta", type=float, default=0.05)
    training.add_argument("--mae-weight", type=float, default=0.25)
    training.add_argument("--direction-weight", type=float, default=0.08)
    training.add_argument("--correction-weight", type=float, default=0.015)
    training.add_argument("--direction-minimum", type=float, default=0.03)
    training.add_argument("--direction-temperature", type=float, default=0.02)
    training.add_argument("--rank-minimum-gap", type=float, default=0.10)
    training.add_argument("--rank-maximum-per-group", type=int, default=64)
    training.add_argument("--rank-temperature", type=float, default=0.08)
    training.add_argument("--gradient-clip", type=float, default=2.0)
    training.add_argument("--seed", type=int, default=2026)
    training.add_argument("--amp", action="store_true")
    training.add_argument("--device", default="cuda")
    training.add_argument("--overwrite", action="store_true")
    training.set_defaults(function=train)

    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--features", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output-root", type=Path, required=True)
    evaluation.add_argument("--batch-size", type=int, default=2048)
    evaluation.add_argument("--device", default="cuda")
    evaluation.set_defaults(function=evaluate)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)

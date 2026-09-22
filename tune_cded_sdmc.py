#!/usr/bin/env python3
"""Polaris-only hyperparameter selection for CDED and SDMC.

The locked RCE checkpoint is never changed.  Fifteen CDED configurations are
trained on Polaris train and compared on Polaris validation.  After selecting
and freezing one CDED checkpoint, five SDMC correction budgets are trained on
the same train/validation split.  Flickr8k-Expert is loaded only after both
choices have been fixed, and is evaluated once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from credicap import decomposition_training
from credicap import pipeline
from credicap import prepare_features as feature_tools
from credicap import train_decomposition
from credicap import train_reference
from credicap import training_utils
from credicap import verification_utils
from credicap.expected_feedback import FeedbackExpectationNet
from credicap.score_correction import DirectionMagnitudeCorrector


FORMAT = "credicap-polaris-tuned-cded-sdmc-v2"
SDMC_FORMAT = "credicap-polaris-tuned-sdmc-v2"
FEATURE_FORMAT = feature_tools.FEATURE_FORMAT
TAU_TIE_X100 = 0.05  # 0.0005 on the paper's 0--1 Tau scale.


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


def slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


class TunableCDED(
    decomposition_training.ConsensusDissentEvidenceDecomposition
):
    """Original CDED with only injection strength and rho floor exposed."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        injection_strength: float,
        rho_minimum: float,
        rank_dim: int = 48,
    ) -> None:
        super().__init__(input_dim, hidden_dim, dropout, rank_dim)
        if not 0.0 < injection_strength <= 1.0:
            raise ValueError("CDED injection strength must be in (0,1]")
        if not 0.0 <= rho_minimum < 1.0:
            raise ValueError("CDED rho minimum must be in [0,1)")
        self.injection_strength = float(injection_strength)
        self.rho_minimum = float(rho_minimum)

    def decompose(self, batch, reference_weights, stage1_score, scalar):
        values = list(
            super().decompose(
                batch, reference_weights, stage1_score, scalar
            )
        )
        support_dispersion = values[6]
        reference_disagreement = values[7]
        raw_strength = (
            0.55 * reference_disagreement
            + 0.45 * support_dispersion
        ).clamp(0.0, 1.0)
        values[2] = (
            self.rho_minimum
            + (1.0 - self.rho_minimum) * raw_strength
        ).clamp(self.rho_minimum, 1.0)
        return tuple(values)

    def forward(
        self,
        batch,
        stage1_score,
        stage1_hidden,
        scalar,
        reference_weights,
    ):
        values = self.decompose(
            batch, reference_weights, stage1_score, scalar
        )
        contrast, structural, structural_strength = values[:3]
        z = torch.cat(
            [self.contrast_proj(contrast), self.scalar_proj(structural)],
            dim=-1,
        )
        raw_delta = torch.tanh(self.adapter(z).float())
        base_rms = (
            stage1_hidden.float()
            .pow(2)
            .mean(dim=1, keepdim=True)
            .sqrt()
            .clamp_min(0.10)
        )
        hidden_delta = (
            self.injection_strength
            * base_rms
            * structural_strength[:, None]
            * raw_delta
        )
        enhanced_hidden = stage1_hidden.float() + hidden_delta
        hidden_change_rms = (
            hidden_delta.pow(2).mean(dim=1) + 1.0e-12
        ).sqrt()
        # Preserve the original 13-value CDED forward contract exactly.
        return (enhanced_hidden, hidden_change_rms, *values[2:])


class TunableTrainingModel(
    decomposition_training.ConsensusDissentTrainingModel
):
    active_injection_strength = 0.30
    active_rho_minimum = 0.20

    def __init__(
        self,
        module1,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        variant: str,
    ) -> None:
        super().__init__(module1, input_dim, hidden_dim, dropout, variant)
        self.module2 = TunableCDED(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            injection_strength=self.active_injection_strength,
            rho_minimum=self.active_rho_minimum,
        )


def candidate_m2_args(args, run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        module1_checkpoint=args.module1_checkpoint,
        polaris_cache=args.polaris_cache,
        run_dir=run_dir,
        epochs=args.m2_epochs,
        minimum_epochs=args.m2_minimum_epochs,
        patience=args.m2_patience,
        batch_size=args.m2_batch_size,
        eval_batch_size=args.eval_batch_size,
        pair_batch_size=args.m2_pair_batch_size,
        lr=args.m2_lr,
        weight_decay=args.m2_weight_decay,
        gradient_clip=args.gradient_clip,
        point_loss_weight=1.0,
        rank_loss_weight=args.m2_rank_loss_weight,
        easy_anchor_weight=args.easy_anchor_weight,
        easy_anchor_quantile=args.easy_anchor_quantile,
        hidden_regularization_weight=args.hidden_regularization_weight,
        rank_minimum_gap=args.m2_rank_minimum_gap,
        rank_maximum_per_group=args.m2_rank_maximum_per_group,
        rank_temperature=args.m2_rank_temperature,
        minimum_tau_gain=args.m2_minimum_tau_gain,
        minimum_error_gain=args.m2_minimum_error_gain,
        seed=args.seed,
        amp=args.amp,
        device=args.device,
    )


def beats_all_three(candidate: dict, control: dict) -> bool:
    return bool(
        candidate["tau_x100"] > control["tau_x100"]
        and candidate["mae"] < control["mae"]
        and candidate["rmse"] < control["rmse"]
    )


def select_metrics_candidate(rows: list[dict], metric_key: str) -> dict:
    eligible = [row for row in rows if row["all_three_better"]]
    pool = eligible if eligible else rows
    maximum_tau = max(row[metric_key]["tau_x100"] for row in pool)
    near_best = [
        row
        for row in pool
        if row[metric_key]["tau_x100"] >= maximum_tau - TAU_TIE_X100
    ]
    # Tau is primary; within a practically tied band, protect large-error
    # calibration first (RMSE), then MAE.  This rule is fixed before Expert.
    return min(
        near_best,
        key=lambda row: (
            row[metric_key]["rmse"],
            row[metric_key]["mae"],
            -row[metric_key]["tau_x100"],
        ),
    )


def train_cded_grid(args, run_root: Path) -> tuple[Path, dict, list[dict]]:
    original_class = train_decomposition.ConsensusDissentTrainingModel
    train_decomposition.ConsensusDissentTrainingModel = TunableTrainingModel
    rows: list[dict] = []
    try:
        for injection in args.cded_injection_grid:
            for rho_minimum in args.cded_rho_minimum_grid:
                name = f"lambda_{slug(injection)}__rho_{slug(rho_minimum)}"
                run_dir = run_root / "cded_grid" / name
                checkpoint_path = run_dir / "module2.best.pt"
                configuration = {
                    "injection_strength": float(injection),
                    "rho_minimum": float(rho_minimum),
                }
                reusable = False
                if checkpoint_path.is_file():
                    payload = torch_load(checkpoint_path)
                    reusable = (
                        payload.get("tuning_configuration") == configuration
                        and payload.get("benchmark_labels_used_for_tuning") is False
                    )
                    if not reusable:
                        raise RuntimeError(
                            f"Incompatible generated CDED candidate: {checkpoint_path}"
                        )
                    print(f"Reuse CDED candidate {name}")
                if not reusable:
                    TunableTrainingModel.active_injection_strength = float(injection)
                    TunableTrainingModel.active_rho_minimum = float(rho_minimum)
                    set_seed(args.seed)
                    train_decomposition.train(candidate_m2_args(args, run_dir))
                    payload = torch_load(checkpoint_path)
                    payload["format_base"] = payload.get("format")
                    payload["format"] = FORMAT
                    payload["tuning_configuration"] = configuration
                    payload["tuning_dataset"] = "official Polaris validation"
                    payload["benchmark_labels_used_for_tuning"] = False
                    save_torch(payload, checkpoint_path)
                payload = torch_load(checkpoint_path)
                metrics = payload["best_validation"]
                control = payload["locked_validation"]
                row = {
                    "name": name,
                    "configuration": configuration,
                    "checkpoint": str(checkpoint_path),
                    "metrics": metrics,
                    "control": control,
                    "all_three_better": beats_all_three(metrics, control),
                    "effect": {
                        "tau_gain": metrics["tau_x100"] - control["tau_x100"],
                        "mae_reduction": control["mae"] - metrics["mae"],
                        "rmse_reduction": control["rmse"] - metrics["rmse"],
                    },
                }
                rows.append(row)
    finally:
        train_decomposition.ConsensusDissentTrainingModel = original_class

    selected = select_metrics_candidate(rows, "metrics")
    source_path = Path(selected["checkpoint"])
    selected_payload = torch_load(source_path)
    selected_payload["selection"] = {
        "rule": (
            "all-three improvement over locked RCE; highest Polaris-val Tau; "
            "within 0.05 Tau-x100 choose lower RMSE then MAE"
        ),
        "candidate_count": len(rows),
        "benchmark_labels_used": False,
    }
    selected_path = run_root / "selected_cded" / "module2.best.pt"
    save_torch(selected_payload, selected_path)
    (run_root / "selected_cded" / "selection.json").write_text(
        json.dumps(
            {
                "selected": selected,
                "candidates": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return selected_path, selected, rows


def build_tuned_model(
    module1_path: Path,
    module2_path: Path,
    device: torch.device,
) -> TunableTrainingModel:
    module1, payload1 = train_decomposition.load_locked_module1(
        module1_path, device
    )
    payload2 = torch_load(module2_path)
    if payload2.get("format") != FORMAT:
        raise RuntimeError("Selected CDED checkpoint is not tuning-valid")
    configuration = payload2["tuning_configuration"]
    TunableTrainingModel.active_injection_strength = float(
        configuration["injection_strength"]
    )
    TunableTrainingModel.active_rho_minimum = float(
        configuration["rho_minimum"]
    )
    model = TunableTrainingModel(
        module1=module1,
        input_dim=int(payload1["embedding_dim"]),
        hidden_dim=int(payload1["hidden_dim"]),
        dropout=float(payload1["dropout"]),
        variant="m12",
    ).to(device)
    model.module2.load_state_dict(payload2["module_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def feature_source_split(path: Path, split_name: str) -> tuple[dict, dict]:
    source = torch_load(path)
    if source.get("format") != train_reference.CACHE_FORMAT:
        raise RuntimeError(f"Wrong source cache: {source.get('format')}")
    if source.get("kind") == "polaris":
        if split_name not in {"train", "val"}:
            raise RuntimeError("Polaris cache requires train or val")
        return source, source[split_name]
    if source.get("kind") == "benchmark":
        if source.get("dataset") != split_name:
            raise RuntimeError("Benchmark split mismatch")
        return source, source["split"]
    raise RuntimeError("Unknown source-cache kind")


def configure_dynamic_hashes(module1_path: Path, module2_path: Path) -> tuple[str, str]:
    module1_hash = sha256(module1_path)
    module2_hash = sha256(module2_path)
    feature_tools.m12_runner.LOCKED_M1_SHA256 = module1_hash
    feature_tools.m12_runner.LOCKED_M2_SHA256 = module2_hash
    training_utils.LOCKED_M1_SHA256 = module1_hash
    training_utils.LOCKED_M2_SHA256 = module2_hash
    return module1_hash, module2_hash


def prepare_tuned_features(
    args,
    source_path: Path,
    split_name: str,
    feedback_path: Path,
    module2_path: Path,
    output: Path,
    device: torch.device,
) -> None:
    module1_hash, module2_hash = configure_dynamic_hashes(
        args.module1_checkpoint, module2_path
    )
    expected = {
        "source_cache_sha256": sha256(source_path),
        "feedback_sha256": sha256(feedback_path),
        "module1_sha256": module1_hash,
        "module2_sha256": module2_hash,
    }
    if output.is_file():
        existing = torch_load(output)
        try:
            feature_tools.validate_feature_cache(existing, split_name)
            if all(existing.get(key) == value for key, value in expected.items()):
                print(f"Reuse exact tuned features: {output}")
                return
        except RuntimeError:
            pass
        raise RuntimeError(f"Stale generated feature cache: {output}")
    source, split = feature_source_split(source_path, split_name)
    feedback_rows = verification_utils.read_jsonl_resume(feedback_path)
    verification_utils.validate_rows(feedback_rows, split["sample_ids"])
    feedback = verification_utils.matrix_from_rows(
        feedback_rows, split["sample_ids"]
    )
    model = build_tuned_model(args.module1_checkpoint, module2_path, device)
    anchor, base_features = feature_tools.extract_locked_features(
        model,
        split,
        args.eval_batch_size,
        device,
        f"Selected tuned CDED features {split_name}",
    )
    payload2 = torch_load(module2_path)
    payload = {
        "format": FEATURE_FORMAT,
        "created_unix": time.time(),
        "split": split_name,
        "source_cache": str(source_path),
        "source_signature": source.get("signature"),
        "source_cache_sha256": expected["source_cache_sha256"],
        "feedback_file": str(feedback_path),
        "feedback_sha256": expected["feedback_sha256"],
        "module1_sha256": module1_hash,
        "module2_sha256": module2_hash,
        "feature_names": feature_tools.FEATURE_NAMES,
        "feedback_fields": verification_utils.FIELDS + ("VALID",),
        "sample_ids": list(split["sample_ids"]),
        "groups": list(split["groups"]),
        "records": list(split["records"]),
        "gold": split["gold"].float().cpu(),
        "anchor": anchor.float().cpu(),
        "base_features": base_features.float().cpu(),
        "feedback": torch.from_numpy(feedback).float(),
        "strict_feedback": int(feedback[:, -1].sum()),
        "cded_tuning_configuration": payload2["tuning_configuration"],
        "benchmark_labels_used_for_training_or_selection": False,
    }
    feature_tools.validate_feature_cache(payload, split_name)
    save_torch(payload, output)


def normalize(cache: dict, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (cache["base_features"].float() - mean) / std


@torch.no_grad()
def expectation_predict(model, values, batch_size, device, description):
    model.eval()
    rows = []
    loader = DataLoader(
        TensorDataset(torch.arange(len(values))),
        batch_size=batch_size,
        shuffle=False,
    )
    for (indices,) in tqdm(
        loader, desc=description, dynamic_ncols=True, leave=False
    ):
        rows.append(model(values.index_select(0, indices).to(device)).cpu())
    return torch.cat(rows)


def train_expectation(args, train_cache, val_cache, device):
    mean = train_cache["base_features"].float().mean(0)
    std = train_cache["base_features"].float().std(0).clamp_min(1.0e-5)
    train_x = normalize(train_cache, mean, std)
    val_x = normalize(val_cache, mean, std)
    train_y = train_cache["feedback"][:, :7].float()
    val_y = val_cache["feedback"][:, :7].float()
    model = FeedbackExpectationNet(
        len(feature_tools.FEATURE_NAMES), args.expectation_hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.expectation_lr,
        weight_decay=args.sdmc_weight_decay,
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train_x))),
        batch_size=args.sdmc_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 3101),
    )
    best_loss, best_epoch, best_state = math.inf, 0, None
    for epoch in range(1, args.expectation_epochs + 1):
        model.train()
        for (indices,) in tqdm(
            loader,
            desc=f"Tuned expected feedback epoch {epoch:02d}",
            dynamic_ncols=True,
        ):
            prediction = model(train_x.index_select(0, indices).to(device))
            target = train_y.index_select(0, indices).to(device)
            loss = F.smooth_l1_loss(prediction, target, beta=0.10)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            optimizer.step()
        val_prediction = expectation_predict(
            model,
            val_x,
            args.eval_batch_size,
            device,
            f"Tuned expectation validation {epoch:02d}",
        )
        value = float(F.mse_loss(val_prediction, val_y))
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("No expected-feedback state selected")
    model.load_state_dict(best_state, strict=True)
    train_expected = expectation_predict(
        model, train_x, args.eval_batch_size, device, "Selected expectation train"
    )
    val_expected = expectation_predict(
        model, val_x, args.eval_batch_size, device, "Selected expectation val"
    )
    metadata = {
        "state": best_state,
        "hidden_dim": args.expectation_hidden_dim,
        "best_epoch": best_epoch,
        "validation_mse": best_loss,
        "normalizer_mean": mean,
        "normalizer_std": std,
    }
    return metadata, (train_x, train_y, train_expected), (val_x, val_y, val_expected)


def sdmc_candidate_args(args, gamma: float) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        hidden_dim=args.sdmc_hidden_dim,
        maximum_base_correction=args.maximum_base_correction,
        maximum_feedback_correction=float(gamma),
        base_epochs=args.base_epochs,
        feedback_epochs=args.feedback_epochs,
        batch_size=args.sdmc_batch_size,
        eval_batch_size=args.eval_batch_size,
        base_lr=args.base_lr,
        feedback_lr=args.feedback_lr,
        weight_decay=args.sdmc_weight_decay,
        neutral_width=args.neutral_width,
        huber_beta=args.huber_beta,
        supervision_weight=args.supervision_weight,
        mse_weight=args.mse_weight,
        mae_weight=args.mae_weight,
        rank_weight=args.sdmc_rank_weight,
        tail_weight=args.tail_weight,
        mean_shift_weight=args.mean_shift_weight,
        feedback_mean_shift_weight=args.feedback_mean_shift_weight,
        rank_temperature=args.sdmc_rank_temperature,
        gradient_clip=args.gradient_clip,
        base_alphas=args.base_alphas,
        feedback_alphas=args.feedback_alphas,
        maximum_feedback_mean_shift=args.maximum_feedback_mean_shift,
        minimum_feedback_tau_gain=args.minimum_feedback_tau_gain,
    )


def train_sdmc_grid(
    args,
    run_root: Path,
    module2_path: Path,
    train_path: Path,
    val_path: Path,
    device: torch.device,
) -> tuple[Path, dict, list[dict]]:
    module1_hash, module2_hash = configure_dynamic_hashes(
        args.module1_checkpoint, module2_path
    )
    train_cache = torch_load(train_path)
    val_cache = torch_load(val_path)
    training_utils.validate_feature_cache(train_cache, "train")
    training_utils.validate_feature_cache(val_cache, "val")
    pipeline.require_all_feedback(train_cache, "Polaris train")
    pipeline.require_all_feedback(val_cache, "Polaris validation")
    expectation, train_inputs, val_inputs = train_expectation(
        args, train_cache, val_cache, device
    )
    pairs, pair_weights, pair_audit = training_utils.make_hard_pairs(
        train_cache,
        args.sdmc_rank_minimum_gap,
        args.sdmc_rank_maximum_per_group,
        args.seed,
    )
    anchor_metrics = training_utils.metric_values(
        val_cache["anchor"].numpy(), val_cache
    )
    rows: list[dict] = []
    for gamma in args.sdmc_gamma_grid:
        candidate_dir = run_root / "sdmc_grid" / f"gamma_{slug(gamma)}"
        checkpoint_path = candidate_dir / "sdmc.best.pt"
        if checkpoint_path.is_file():
            payload = torch_load(checkpoint_path)
            if (
                payload.get("format") != SDMC_FORMAT
                or float(payload.get("gamma", -1.0)) != float(gamma)
                or payload.get("module2_sha256") != module2_hash
            ):
                raise RuntimeError(f"Incompatible SDMC candidate: {checkpoint_path}")
            print(f"Reuse SDMC gamma={gamma:.3f}")
        else:
            set_seed(args.seed)
            local = sdmc_candidate_args(args, gamma)
            model = DirectionMagnitudeCorrector(
                hidden_dim=local.hidden_dim,
                maximum_base_correction=local.maximum_base_correction,
                maximum_feedback_correction=local.maximum_feedback_correction,
            ).to(device)
            base_snapshots = pipeline.train_base(
                local, model, train_cache, val_cache, train_inputs, val_inputs,
                pairs, pair_weights, device
            )
            base_selected, base_candidates = pipeline.select_base(
                local, base_snapshots, model, val_cache, val_inputs, device
            )
            feedback_snapshots = pipeline.train_feedback(
                local, model, base_selected["alpha"], train_cache, val_cache,
                train_inputs, val_inputs, pairs, pair_weights, device
            )
            feedback_selected, feedback_candidates = pipeline.select_feedback(
                local, feedback_snapshots, model, base_selected,
                val_cache, val_inputs, device
            )
            payload = {
                "format": SDMC_FORMAT,
                "created_unix": time.time(),
                "gamma": float(gamma),
                "training": "official Polaris train only",
                "selection": "official Polaris validation only",
                "benchmark_labels_used_for_training_or_selection": False,
                "module1_sha256": module1_hash,
                "module2_sha256": module2_hash,
                "train_features_sha256": sha256(train_path),
                "val_features_sha256": sha256(val_path),
                "hidden_dim": local.hidden_dim,
                "maximum_base_correction": local.maximum_base_correction,
                "maximum_feedback_correction": local.maximum_feedback_correction,
                "expectation": expectation,
                "anchor_validation": anchor_metrics,
                "pair_audit": pair_audit,
                "base_selected": base_selected,
                "feedback_selected": feedback_selected,
                "base_candidates": [
                    {k: v for k, v in row.items() if k != "state"}
                    for row in base_candidates
                ],
                "feedback_candidates": [
                    {k: v for k, v in row.items() if k != "state"}
                    for row in feedback_candidates
                ],
            }
            save_torch(payload, checkpoint_path)
        payload = torch_load(checkpoint_path)
        metrics = payload["feedback_selected"]["metrics"]
        row = {
            "name": f"gamma_{slug(gamma)}",
            "gamma": float(gamma),
            "checkpoint": str(checkpoint_path),
            "metrics": metrics,
            "control": anchor_metrics,
            "all_three_better": beats_all_three(metrics, anchor_metrics),
            "effect": training_utils.improvement(metrics, anchor_metrics),
        }
        rows.append(row)
    selected = select_metrics_candidate(rows, "metrics")
    selected_payload = torch_load(Path(selected["checkpoint"]))
    selected_payload["selection"] = {
        "rule": (
            "all-three improvement over selected M12; highest Polaris-val Tau; "
            "within 0.05 Tau-x100 choose lower RMSE then MAE"
        ),
        "candidate_count": len(rows),
        "benchmark_labels_used": False,
    }
    selected_path = run_root / "selected_sdmc" / "sdmc.best.pt"
    save_torch(selected_payload, selected_path)
    (run_root / "selected_sdmc" / "selection.json").write_text(
        json.dumps({"selected": selected, "candidates": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return selected_path, selected, rows


def evaluation_inputs(
    cache, checkpoint, batch_size, device, description="Selected expected feedback"
):
    expectation = checkpoint["expectation"]
    normalized = normalize(
        cache,
        expectation["normalizer_mean"].float(),
        expectation["normalizer_std"].float(),
    )
    model = FeedbackExpectationNet(
        len(feature_tools.FEATURE_NAMES), int(expectation["hidden_dim"])
    ).to(device)
    model.load_state_dict(expectation["state"], strict=True)
    expected = expectation_predict(
        model, normalized, batch_size, device, description
    )
    return normalized, cache["feedback"][:, :7].float(), expected


def metric_cells(values: dict) -> str:
    return (
        f"{values['tau_x100']/100.0:.4f} | "
        f"{values['mae']:.4f} | {values['rmse']:.4f}"
    )


def write_grid_report(
    run_root: Path,
    selected_cded: dict,
    cded_rows: list[dict],
    selected_sdmc: dict,
    sdmc_rows: list[dict],
) -> None:
    lines = [
        "# Polaris-validation hyperparameter selection",
        "",
        "## CDED grid",
        "",
        "| lambda | rho_min | Tau | MAE | RMSE | All-three vs RCE |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in cded_rows:
        cfg, m = row["configuration"], row["metrics"]
        lines.append(
            f"| {cfg['injection_strength']:.2f} | {cfg['rho_minimum']:.2f} | "
            f"{m['tau_x100']/100:.4f} | {m['mae']:.4f} | {m['rmse']:.4f} | "
            f"{'YES' if row['all_three_better'] else 'NO'} |"
        )
    lines += [
        "",
        f"Selected CDED: `{json.dumps(selected_cded['configuration'])}`",
        "",
        "## SDMC grid",
        "",
        "| gamma | Tau | MAE | RMSE | All-three vs selected M12 |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for row in sdmc_rows:
        m = row["metrics"]
        lines.append(
            f"| {row['gamma']:.2f} | {m['tau_x100']/100:.4f} | "
            f"{m['mae']:.4f} | {m['rmse']:.4f} | "
            f"{'YES' if row['all_three_better'] else 'NO'} |"
        )
    lines += ["", f"Selected SDMC gamma: `{selected_sdmc['gamma']:.2f}`"]
    (run_root / "polaris_validation_parameter_selection.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def evaluate_expert(
    args,
    run_root: Path,
    module2_path: Path,
    sdmc_path: Path,
    expert_features: Path,
    device: torch.device,
) -> None:
    configure_dynamic_hashes(args.module1_checkpoint, module2_path)
    cache = torch_load(expert_features)
    training_utils.validate_feature_cache(cache, "expert")
    checkpoint = torch_load(sdmc_path)
    if checkpoint.get("benchmark_labels_used_for_training_or_selection") is not False:
        raise RuntimeError("Benchmark-selected SDMC checkpoint is forbidden")
    inputs = evaluation_inputs(cache, checkpoint, args.eval_batch_size, device)
    model = DirectionMagnitudeCorrector(
        hidden_dim=int(checkpoint["hidden_dim"]),
        maximum_base_correction=float(checkpoint["maximum_base_correction"]),
        maximum_feedback_correction=float(
            checkpoint["maximum_feedback_correction"]
        ),
    ).to(device)
    model.load_state_dict(checkpoint["feedback_selected"]["state"], strict=True)
    prediction = pipeline.predict_split(
        model,
        cache,
        *inputs,
        args.eval_batch_size,
        device,
        "Frozen tuned Full CrediCap expert",
    )
    base_alpha = float(checkpoint["base_selected"]["alpha"])
    feedback_alpha = float(checkpoint["feedback_selected"]["feedback_alpha"])
    final = pipeline.score_from(
        cache["anchor"],
        prediction.base.correction,
        prediction.feedback.correction,
        base_alpha,
        feedback_alpha,
    )
    rce_metrics = training_utils.metric_values(
        cache["base_features"][:, 1].numpy(), cache
    )
    m12_metrics = training_utils.metric_values(cache["anchor"].numpy(), cache)
    full_metrics = training_utils.metric_values(final.numpy(), cache)
    previous_m12 = {"tau_x100": 54.13, "mae": 0.1026, "rmse": 0.1483}
    previous_full = {"tau_x100": 54.28, "mae": 0.1006, "rmse": 0.1467}
    m12_beats_previous = beats_all_three(m12_metrics, previous_m12)
    full_beats_previous = beats_all_three(full_metrics, previous_full)
    payload2 = torch_load(module2_path)
    report = {
        "format": FORMAT,
        "dataset": "expert",
        "training": "Polaris train",
        "selection": "Polaris validation",
        "benchmark_labels_used_for_training_or_selection": False,
        "selected_cded": payload2["tuning_configuration"],
        "selected_sdmc_gamma": float(checkpoint["gamma"]),
        "metrics": {
            "rce": rce_metrics,
            "rce_cded": m12_metrics,
            "full": full_metrics,
        },
        "effects": {
            "cded_vs_rce": training_utils.improvement(m12_metrics, rce_metrics),
            "sdmc_vs_m12": training_utils.improvement(full_metrics, m12_metrics),
        },
        "comparison_with_previous_expert_rows": {
            "previous_rce_cded": previous_m12,
            "new_rce_cded_all_three_better": m12_beats_previous,
            "previous_full": previous_full,
            "new_full_all_three_better": full_beats_previous,
            "used_for_selection": False,
        },
    }
    expert_root = run_root / "expert"
    expert_root.mkdir(parents=True, exist_ok=True)
    (expert_root / "tuned_cumulative_ablation.metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Tuned CrediCap cumulative ablation on Flickr8k-Expert",
        "",
        "| Method | Expert Tau-c ↑ | MAE ↓ | RMSE ↓ |",
        "|---|---:|---:|---:|",
        f"| RCE | {metric_cells(rce_metrics)} |",
        f"| RCE+CDED | {metric_cells(m12_metrics)} |",
        f"| RCE+CDED+SDMC (Full) | {metric_cells(full_metrics)} |",
        "",
        f"- Selected CDED: lambda={payload2['tuning_configuration']['injection_strength']:.2f}, "
        f"rho_min={payload2['tuning_configuration']['rho_minimum']:.2f}.",
        f"- Selected SDMC maximum feedback correction: gamma={float(checkpoint['gamma']):.2f}.",
        "- All hyperparameters were selected on Polaris validation before Expert was loaded.",
        f"- New RCE+CDED beats the strongest previous 0.5413/0.1026/0.1483 on all three: "
        f"**{'YES' if m12_beats_previous else 'NO'}**.",
        f"- New Full beats the previous tuned 0.5428/0.1006/0.1467 on all three: "
        f"**{'YES' if full_beats_previous else 'NO'}**.",
    ]
    summary = run_root / "tuned_cumulative_ablation_summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Summary: {summary}")


def write_all_datasets_summary(run_root: Path) -> None:
    paths = {
        "expert": run_root / "expert" / "tuned_cumulative_ablation.metrics.json",
        "cf": run_root / "benchmarks" / "cf" / "cf_tuned_v2.metrics.json",
        "composite": (
            run_root
            / "benchmarks"
            / "composite"
            / "composite_tuned_v2.metrics.json"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        print("All-dataset summary pending; missing:")
        for path in missing:
            print(f"  {path}")
        return
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    methods = (
        ("RCE", "rce"),
        ("RCE+CDED", "rce_cded"),
        ("RCE+CDED+SDMC (Full)", "full"),
    )
    lines = [
        "# CrediCap v2 frozen-checkpoint results on all three datasets",
        "",
        "| Method | Expert Tau-c ↑ | MAE ↓ | RMSE ↓ | "
        "CF Tau-b ↑ | MAE ↓ | RMSE ↓ | Composite Tau-c ↑ | MAE ↓ | RMSE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in methods:
        cells = []
        for dataset in ("expert", "cf", "composite"):
            metrics = reports[dataset]["metrics"][key]
            cells.extend(
                [
                    f"{metrics['tau_x100']/100.0:.4f}",
                    f"{metrics['mae']:.4f}",
                    f"{metrics['rmse']:.4f}",
                ]
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "- One frozen v2 CDED checkpoint and one frozen v2 SDMC checkpoint are used everywhere.",
        "- No Expert, CF, or Composite labels are used for training or parameter selection.",
    ]
    summary = run_root / "all_datasets_tuned_v2_summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"All-dataset summary: {summary}")


def evaluate_benchmark(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        raise ValueError("--run-name must be one safe directory name")
    args.module1_checkpoint = (
        args.fleur_root
        / "results"
        / "trijudge_formal_v2"
        / "checkpoints"
        / "formal_stage2.best.pt"
    )
    run_root = args.fleur_root / "results" / args.run_name
    module2_path = run_root / "selected_cded" / "module2.best.pt"
    sdmc_path = run_root / "selected_sdmc" / "sdmc.best.pt"
    source_path = (
        args.fleur_root
        / "results"
        / "trijudge_formal_v2"
        / f"{args.dataset}_cache.pt"
    )
    feedback_path = (
        args.fleur_root
        / "results"
        / "trijudge_feedback_calibrator_v6"
        / "feedback"
        / f"{args.dataset}.jsonl"
    )
    for path in (
        args.module1_checkpoint,
        module2_path,
        sdmc_path,
        source_path,
        feedback_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    feature_path = run_root / "features" / f"{args.dataset}.pt"
    prepare_tuned_features(
        args,
        source_path,
        args.dataset,
        feedback_path,
        module2_path,
        feature_path,
        device,
    )
    configure_dynamic_hashes(args.module1_checkpoint, module2_path)
    cache = torch_load(feature_path)
    training_utils.validate_feature_cache(cache, args.dataset)
    checkpoint = torch_load(sdmc_path)
    if checkpoint.get("benchmark_labels_used_for_training_or_selection") is not False:
        raise RuntimeError("Benchmark-selected SDMC checkpoint is forbidden")
    inputs = evaluation_inputs(
        cache,
        checkpoint,
        args.eval_batch_size,
        device,
        f"Frozen v2 expected feedback {args.dataset}",
    )
    model = DirectionMagnitudeCorrector(
        hidden_dim=int(checkpoint["hidden_dim"]),
        maximum_base_correction=float(checkpoint["maximum_base_correction"]),
        maximum_feedback_correction=float(
            checkpoint["maximum_feedback_correction"]
        ),
    ).to(device)
    model.load_state_dict(checkpoint["feedback_selected"]["state"], strict=True)
    prediction = pipeline.predict_split(
        model,
        cache,
        *inputs,
        args.eval_batch_size,
        device,
        f"Frozen tuned v2 Full CrediCap {args.dataset}",
    )
    base_alpha = float(checkpoint["base_selected"]["alpha"])
    feedback_alpha = float(checkpoint["feedback_selected"]["feedback_alpha"])
    final = pipeline.score_from(
        cache["anchor"],
        prediction.base.correction,
        prediction.feedback.correction,
        base_alpha,
        feedback_alpha,
    )
    metrics = {
        "rce": training_utils.metric_values(
            cache["base_features"][:, 1].numpy(), cache
        ),
        "rce_cded": training_utils.metric_values(cache["anchor"].numpy(), cache),
        "full": training_utils.metric_values(final.numpy(), cache),
    }
    payload2 = torch_load(module2_path)
    report = {
        "format": FORMAT,
        "dataset": args.dataset,
        "training": "Polaris train",
        "selection": "Polaris validation",
        "benchmark_labels_used_for_training_or_selection": False,
        "selected_cded": payload2["tuning_configuration"],
        "selected_sdmc_gamma": float(checkpoint["gamma"]),
        "metrics": metrics,
        "effects": {
            "cded_vs_rce": training_utils.improvement(
                metrics["rce_cded"], metrics["rce"]
            ),
            "sdmc_vs_m12": training_utils.improvement(
                metrics["full"], metrics["rce_cded"]
            ),
        },
    }
    output_root = run_root / "benchmarks" / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{args.dataset}_tuned_v2.metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result_path = output_root / f"{args.dataset}_tuned_v2_results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for record, rce, m12, score in zip(
            cache["records"],
            cache["base_features"][:, 1].tolist(),
            cache["anchor"].tolist(),
            final.tolist(),
        ):
            row = dict(record)
            row.update(
                {
                    "rce_score": float(rce),
                    "rce_cded_score": float(m12),
                    "score": float(score),
                    "mode": "credicap_tuned_cded_sdmc_v2_frozen",
                    "benchmark_label_used_for_training_or_selection": False,
                }
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = [
        f"# Frozen CrediCap v2 — {args.dataset}",
        "",
        "| Method | Tau ↑ | MAE ↓ | RMSE ↓ |",
        "|---|---:|---:|---:|",
        f"| RCE | {metric_cells(metrics['rce'])} |",
        f"| RCE+CDED | {metric_cells(metrics['rce_cded'])} |",
        f"| RCE+CDED+SDMC (Full) | {metric_cells(metrics['full'])} |",
        "",
        f"- CDED lambda={payload2['tuning_configuration']['injection_strength']:.2f}, "
        f"rho_min={payload2['tuning_configuration']['rho_minimum']:.2f}.",
        f"- SDMC gamma={float(checkpoint['gamma']):.2f}.",
        "- Training/selection: Polaris only; this benchmark is evaluation only.",
    ]
    summary_path = output_root / f"{args.dataset}_tuned_v2_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Report : {report_path}")
    print(f"Result : {result_path}")
    print(f"Summary: {summary_path}")
    write_all_datasets_summary(run_root)


def run(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    args.module1_checkpoint = (
        args.fleur_root
        / "results"
        / "trijudge_formal_v2"
        / "checkpoints"
        / "formal_stage2.best.pt"
    )
    args.polaris_cache = (
        args.fleur_root
        / "results"
        / "trijudge_formal_v2"
        / "polaris_train_val_cache.pt"
    )
    expert_cache = (
        args.fleur_root
        / "results"
        / "trijudge_formal_v2"
        / "expert_cache.pt"
    )
    feedback_root = (
        args.fleur_root
        / "results"
        / "trijudge_feedback_calibrator_v6"
        / "feedback"
    )
    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        raise ValueError("--run-name must be one safe directory name")
    run_root = args.fleur_root / "results" / args.run_name
    required_before_selection = (
        args.module1_checkpoint,
        args.polaris_cache,
        feedback_root / "polaris_train.jsonl",
        feedback_root / "polaris_val.jsonl",
    )
    for path in required_before_selection:
        if not path.is_file():
            raise FileNotFoundError(path)

    module2_path, selected_cded, cded_rows = train_cded_grid(args, run_root)
    feature_root = run_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    prepare_tuned_features(
        args,
        args.polaris_cache,
        "train",
        feedback_root / "polaris_train.jsonl",
        module2_path,
        feature_root / "train.pt",
        device,
    )
    prepare_tuned_features(
        args,
        args.polaris_cache,
        "val",
        feedback_root / "polaris_val.jsonl",
        module2_path,
        feature_root / "val.pt",
        device,
    )
    sdmc_path, selected_sdmc, sdmc_rows = train_sdmc_grid(
        args,
        run_root,
        module2_path,
        feature_root / "train.pt",
        feature_root / "val.pt",
        device,
    )
    write_grid_report(
        run_root, selected_cded, cded_rows, selected_sdmc, sdmc_rows
    )

    if args.selection_only:
        print("=" * 112)
        print("POLARIS-ONLY PARAMETER EXPLORATION FINISHED")
        print("Expert/CF/Composite caches were not opened.")
        print(
            "Selection: "
            f"{run_root / 'polaris_validation_parameter_selection.md'}"
        )
        print(f"CDED:     {module2_path}")
        print(f"SDMC:     {sdmc_path}")
        print("=" * 112)
        return

    # Expert is deliberately touched only after both selections are final.
    for path in (expert_cache, feedback_root / "expert.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(path)
    prepare_tuned_features(
        args,
        expert_cache,
        "expert",
        feedback_root / "expert.jsonl",
        module2_path,
        feature_root / "expert.pt",
        device,
    )
    evaluate_expert(
        args,
        run_root,
        module2_path,
        sdmc_path,
        feature_root / "expert.pt",
        device,
    )


def selfcheck(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    module = TunableCDED(
        input_dim=32,
        hidden_dim=48,
        dropout=0.0,
        injection_strength=0.15,
        rho_minimum=0.0,
    ).to(device)
    count, references = 5, 4
    batch = {
        "candidate": torch.randn(count, 32, device=device),
        "references": torch.randn(count, references, 32, device=device),
        "reference_mask": torch.ones(
            count, references, dtype=torch.bool, device=device
        ),
        "baseline": torch.rand(count, device=device),
    }
    scalar = torch.randn(count, 13, device=device)
    weights = torch.softmax(torch.randn(count, references, device=device), 1)
    result = module(
        batch,
        torch.rand(count, device=device),
        torch.randn(count, 48, device=device),
        scalar,
        weights,
    )
    loss = result[0].mean()
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Tunable CDED self-check failed")
    print("POLARIS-TUNED CDED+SDMC SELF-CHECK: PASS")
    print("Locked RCE unchanged                : YES")
    print("CDED lambda/rho exposed             : YES")
    print("SDMC gamma exposed                  : YES")
    print("Expert used for selection           : NO")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("selfcheck")
    check.add_argument("--seed", type=int, default=2026)
    check.add_argument("--device", default="cpu")
    check.set_defaults(function=selfcheck)
    command = commands.add_parser("run")
    command.add_argument("--fleur-root", type=Path, required=True)
    command.add_argument(
        "--run-name",
        default="credicap_tuned_cded_sdmc_v2",
        help="New result directory name under FLEUR/results.",
    )
    command.add_argument("--seed", type=int, default=2026)
    command.add_argument("--device", default="cuda")
    command.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    command.add_argument(
        "--cded-injection-grid",
        type=float,
        nargs="+",
        default=(0.04, 0.06, 0.08, 0.10, 0.12),
    )
    command.add_argument(
        "--cded-rho-minimum-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.025, 0.05),
    )
    command.add_argument(
        "--sdmc-gamma-grid",
        type=float,
        nargs="+",
        default=(0.05, 0.06, 0.07, 0.08, 0.10),
    )
    command.add_argument("--m2-epochs", type=int, default=60)
    command.add_argument("--m2-minimum-epochs", type=int, default=8)
    command.add_argument("--m2-patience", type=int, default=14)
    command.add_argument("--m2-batch-size", type=int, default=256)
    command.add_argument("--m2-pair-batch-size", type=int, default=160)
    command.add_argument("--m2-lr", type=float, default=1.0e-4)
    command.add_argument("--m2-weight-decay", type=float, default=1.0e-4)
    command.add_argument("--m2-rank-loss-weight", type=float, default=0.20)
    command.add_argument("--easy-anchor-weight", type=float, default=0.12)
    command.add_argument("--easy-anchor-quantile", type=float, default=0.35)
    command.add_argument(
        "--hidden-regularization-weight", type=float, default=0.02
    )
    command.add_argument("--m2-rank-minimum-gap", type=float, default=0.15)
    command.add_argument(
        "--m2-rank-maximum-per-group", type=int, default=64
    )
    command.add_argument("--m2-rank-temperature", type=float, default=0.08)
    command.add_argument("--m2-minimum-tau-gain", type=float, default=0.05)
    command.add_argument("--m2-minimum-error-gain", type=float, default=5.0e-5)
    command.add_argument("--expectation-hidden-dim", type=int, default=128)
    command.add_argument("--expectation-epochs", type=int, default=8)
    command.add_argument("--expectation-lr", type=float, default=2.0e-4)
    command.add_argument("--sdmc-hidden-dim", type=int, default=128)
    command.add_argument("--maximum-base-correction", type=float, default=0.05)
    command.add_argument("--base-epochs", type=int, default=8)
    command.add_argument("--feedback-epochs", type=int, default=12)
    command.add_argument("--sdmc-batch-size", type=int, default=512)
    command.add_argument("--eval-batch-size", type=int, default=4096)
    command.add_argument("--base-lr", type=float, default=1.5e-4)
    command.add_argument("--feedback-lr", type=float, default=1.5e-4)
    command.add_argument("--sdmc-weight-decay", type=float, default=2.0e-4)
    command.add_argument("--neutral-width", type=float, default=0.015)
    command.add_argument("--huber-beta", type=float, default=0.04)
    command.add_argument("--supervision-weight", type=float, default=0.75)
    command.add_argument("--mse-weight", type=float, default=8.0)
    command.add_argument("--mae-weight", type=float, default=0.30)
    command.add_argument("--sdmc-rank-weight", type=float, default=0.22)
    command.add_argument("--tail-weight", type=float, default=3.0)
    command.add_argument("--mean-shift-weight", type=float, default=12.0)
    command.add_argument(
        "--feedback-mean-shift-weight", type=float, default=8.0
    )
    command.add_argument("--sdmc-rank-temperature", type=float, default=0.055)
    command.add_argument("--sdmc-rank-minimum-gap", type=float, default=0.08)
    command.add_argument(
        "--sdmc-rank-maximum-per-group", type=int, default=96
    )
    command.add_argument("--gradient-clip", type=float, default=2.0)
    command.add_argument(
        "--base-alphas",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    command.add_argument(
        "--feedback-alphas",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50, 0.75, 1.0),
    )
    command.add_argument(
        "--maximum-feedback-mean-shift", type=float, default=0.005
    )
    command.add_argument(
        "--minimum-feedback-tau-gain", type=float, default=0.01
    )
    command.add_argument(
        "--selection-only",
        action="store_true",
        help=(
            "Stop after Polaris train/validation selection and never open "
            "Expert, CF, or Composite."
        ),
    )
    command.set_defaults(function=run)

    benchmark = commands.add_parser(
        "benchmark",
        help="Evaluate one frozen v2 checkpoint on CF or Composite.",
    )
    benchmark.add_argument("--fleur-root", type=Path, required=True)
    benchmark.add_argument(
        "--run-name", default="credicap_tuned_cded_sdmc_v2"
    )
    benchmark.add_argument(
        "--dataset", choices=("cf", "composite"), required=True
    )
    benchmark.add_argument("--seed", type=int, default=2026)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--eval-batch-size", type=int, default=4096)
    benchmark.set_defaults(function=evaluate_benchmark)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

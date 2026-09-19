from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from credicap import training_utils as common
from credicap.expected_feedback import (
    ExpectedFeedbackAdapter,
    FeedbackExpectationNet,
)


V8_FORMAT = "trijudge-feedback-conditional-innovation-checkpoint-v8"
RISK_SCALAR_NAMES = (
    "m12_anchor",
    "v8_proposal_score",
    "v8_proposal_correction",
    "v8_absolute_correction",
    "v8_correction_sign",
    "v8_gate",
    "feedback_novelty",
    "m1_m2_feedback_agreement",
)


def load_v8_checkpoint(path: Path) -> dict:
    checkpoint = common.torch_load(path)
    if checkpoint.get("format") != V8_FORMAT:
        raise RuntimeError(
            f"Expected locked v8 checkpoint, got {checkpoint.get('format')!r}"
        )
    if checkpoint.get("module1_sha256") != common.LOCKED_M1_SHA256:
        raise RuntimeError("v8 checkpoint is not tied to locked M1")
    if checkpoint.get("module2_sha256") != common.LOCKED_M2_SHA256:
        raise RuntimeError("v8 checkpoint is not tied to locked CDED-M2")
    if checkpoint.get("benchmark_labels_used_for_training_or_selection") is not False:
        raise RuntimeError("Benchmark-selected v8 checkpoint is forbidden")
    if checkpoint.get("feedback_active") is not True:
        raise RuntimeError("The supplied v8 did not pass its Polaris gate")
    return checkpoint


def build_frozen_v8(checkpoint: dict, device: torch.device):
    expectation = FeedbackExpectationNet(
        len(common.FEATURE_NAMES),
        int(checkpoint["expectation_hidden_dim"]),
    ).to(device)
    expectation.load_state_dict(checkpoint["expectation"]["state"], strict=True)
    expectation.eval()
    configuration = checkpoint["selected"]["configuration"]
    adapter = ExpectedFeedbackAdapter(
        base_dim=len(common.FEATURE_NAMES),
        hidden_dim=int(checkpoint["hidden_dim"]),
        maximum_correction=float(configuration["maximum_correction"]),
    ).to(device)
    adapter.load_state_dict(checkpoint["selected"]["state"], strict=True)
    adapter.eval()
    for parameter in expectation.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return expectation, adapter


@torch.inference_mode()
def frozen_v8_proposal(
    cache: dict,
    checkpoint: dict,
    batch_size: int,
    device: torch.device,
    description: str,
) -> dict:
    expectation, adapter = build_frozen_v8(checkpoint, device)
    normalized = common.normalized_base(
        cache,
        checkpoint["normalizer_mean"].float(),
        checkpoint["normalizer_std"].float(),
    )
    alpha = float(checkpoint["selected"]["configuration"]["blend_alpha"])
    expected_rows = []
    proposal_scores = []
    proposal_corrections = []
    gates = []
    novelties = []
    agreements = []
    loader = DataLoader(
        TensorDataset(torch.arange(len(cache["sample_ids"]))),
        batch_size=batch_size,
        shuffle=False,
    )
    for (indices,) in tqdm(loader, desc=description, dynamic_ncols=True):
        normalized_batch = normalized.index_select(0, indices).to(device)
        expected = expectation(normalized_batch)
        output = adapter(
            cache["anchor"].index_select(0, indices).to(device),
            normalized_batch,
            cache["base_features"].index_select(0, indices).to(device),
            cache["feedback"].index_select(0, indices).to(device),
            expected,
        )
        anchor = cache["anchor"].index_select(0, indices).to(device)
        correction = alpha * output.correction
        score = (anchor + correction).clamp(0.0, 1.0)
        expected_rows.append(expected.cpu())
        proposal_scores.append(score.cpu())
        proposal_corrections.append((score - anchor).cpu())
        gates.append(output.gate.cpu())
        novelties.append(output.novelty.cpu())
        agreements.append(output.agreement.cpu())
    expected = torch.cat(expected_rows)
    proposal_score = torch.cat(proposal_scores)
    proposal_correction = torch.cat(proposal_corrections)
    gate = torch.cat(gates)
    novelty = torch.cat(novelties)
    agreement = torch.cat(agreements)
    scalars = torch.stack(
        [
            cache["anchor"].float(),
            proposal_score,
            proposal_correction,
            proposal_correction.abs(),
            torch.sign(proposal_correction),
            gate,
            novelty,
            agreement,
        ],
        dim=1,
    )
    risk_features = torch.cat(
        [
            normalized,
            cache["feedback"][:, :7].float(),
            expected,
            scalars,
        ],
        dim=1,
    )
    expected_dim = (
        len(common.FEATURE_NAMES)
        + 7
        + 7
        + len(RISK_SCALAR_NAMES)
    )
    if risk_features.shape != (len(cache["sample_ids"]), expected_dim):
        raise RuntimeError(f"Risk feature mismatch: {tuple(risk_features.shape)}")
    if not torch.isfinite(risk_features).all():
        raise RuntimeError("Non-finite frozen-v8 risk features")
    return {
        "expected_feedback": expected,
        "proposal_score": proposal_score,
        "proposal_correction": proposal_correction,
        "risk_features": risk_features,
        "v8_alpha": alpha,
        "diagnostics": {
            "mean_signed_proposal": float(proposal_correction.mean()),
            "mean_abs_proposal": float(proposal_correction.abs().mean()),
            "maximum_abs_proposal": float(proposal_correction.abs().max()),
            "mean_gate": float(gate.mean()),
            "mean_novelty": float(novelty.mean()),
            "mean_agreement": float(agreement.mean()),
        },
    }


def stable_strata(groups: list[str], count: int = 4) -> list[np.ndarray]:
    result: list[list[int]] = [[] for _ in range(count)]
    for index, group in enumerate(groups):
        digest = hashlib.sha256(str(group).encode("utf-8")).digest()
        result[int.from_bytes(digest[:4], "big") % count].append(index)
    arrays = [np.asarray(indices, dtype=np.int64) for indices in result]
    if any(len(indices) == 0 for indices in arrays):
        raise RuntimeError("Empty validation robustness stratum")
    return arrays


def subset_cache(cache: dict, indices: np.ndarray) -> dict:
    tensor_indices = torch.from_numpy(indices).long()
    return {
        "split": cache["split"],
        "gold": cache["gold"].index_select(0, tensor_indices),
        "records": [cache["records"][int(index)] for index in indices],
    }


def robustness(
    prediction: np.ndarray,
    anchor: np.ndarray,
    cache: dict,
) -> dict:
    rows = []
    for number, indices in enumerate(stable_strata(cache["groups"]), start=1):
        local = subset_cache(cache, indices)
        control = common.metric_values(anchor[indices], local)
        candidate = common.metric_values(prediction[indices], local)
        rows.append(
            {
                "stratum": number,
                "n": int(len(indices)),
                "effect": common.improvement(candidate, control),
            }
        )
    minimum_tau = min(row["effect"]["tau_gain"] for row in rows)
    minimum_mae = min(row["effect"]["mae_reduction"] for row in rows)
    minimum_rmse = min(row["effect"]["rmse_reduction"] for row in rows)
    return {
        "strata": rows,
        "minimum_tau_gain": float(minimum_tau),
        "minimum_mae_reduction": float(minimum_mae),
        "minimum_rmse_reduction": float(minimum_rmse),
        "stable": bool(
            minimum_tau >= -0.05
            and minimum_mae >= -0.00025
            and minimum_rmse >= -0.00025
        ),
    }

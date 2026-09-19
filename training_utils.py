from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats
import torch


SOURCE_FORMAT = "trijudge-feedback-calibrator-features-v6"
LOCKED_M1_SHA256 = (
    "eddd881ce97315bf0274bce01f3bfb222ca5a6cd272bd1e7ae4865d6c78c5e16"
)
LOCKED_M2_SHA256 = (
    "36409390b68d8a38dc64322122e192c3962cccd975ae5e69662a43328088c47e"
)
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
FEEDBACK_FIELDS = (
    "ENTITY_SUPPORT",
    "ACTION_SUPPORT",
    "ATTRIBUTE_SUPPORT",
    "REFERENCE_SUPPORT",
    "REFERENCE_CONFLICT",
    "UNSUPPORTED_DETAIL",
    "UNCERTAINTY",
    "VALID",
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


def validate_feature_cache(cache: dict, required_split: str | None = None) -> None:
    if cache.get("format") != SOURCE_FORMAT:
        raise RuntimeError(
            f"Wrong feature cache format: {cache.get('format')!r}"
        )
    if required_split is not None and cache.get("split") != required_split:
        raise RuntimeError(
            f"Expected {required_split} features, got {cache.get('split')}"
        )
    if tuple(cache.get("feature_names", ())) != FEATURE_NAMES:
        raise RuntimeError("Frozen M1/M2 feature contract mismatch")
    if tuple(cache.get("feedback_fields", ())) != FEEDBACK_FIELDS:
        raise RuntimeError("Structured-feedback field contract mismatch")
    if cache.get("module1_sha256") != LOCKED_M1_SHA256:
        raise RuntimeError("Feature cache was not produced by locked M1")
    if cache.get("module2_sha256") != LOCKED_M2_SHA256:
        raise RuntimeError("Feature cache was not produced by locked CDED-M2")
    count = len(cache.get("sample_ids", ()))
    if count == 0 or len(set(cache["sample_ids"])) != count:
        raise RuntimeError("Empty or duplicate sample IDs")
    for key in (
        "records",
        "groups",
        "gold",
        "anchor",
        "base_features",
        "feedback",
    ):
        if len(cache[key]) != count:
            raise RuntimeError(f"Feature-cache length mismatch: {key}")
    if tuple(cache["base_features"].shape) != (count, len(FEATURE_NAMES)):
        raise RuntimeError("Wrong M1/M2 diagnostic shape")
    if tuple(cache["feedback"].shape) != (count, len(FEEDBACK_FIELDS)):
        raise RuntimeError("Wrong structured-feedback shape")
    for key in ("gold", "anchor", "base_features", "feedback"):
        if not torch.isfinite(cache[key].float()).all():
            raise RuntimeError(f"Non-finite values in {key}")
    feedback = cache["feedback"].float()
    if torch.any(feedback < 0.0) or torch.any(feedback > 1.0):
        raise RuntimeError("Structured-feedback values must stay in [0,1]")


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
    absolute = np.abs(difference)
    square = np.square(difference)
    tail_count = max(1, int(np.ceil(0.10 * len(difference))))
    tail_indices = np.argpartition(absolute, -tail_count)[-tail_count:]
    return {
        "split": cache["split"],
        "n": len(prediction),
        "tau_variant": variant,
        "tau_x100": float(100.0 * tau),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(square.mean())),
        "bias": float(difference.mean()),
        "tail10_rmse": float(np.sqrt(square[tail_indices].mean())),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
    }


def metric_objective(values: dict) -> float:
    return float(
        values["tau_x100"]
        - 20.0 * values["mae"]
        - 20.0 * values["rmse"]
        - 3.0 * values["tail10_rmse"]
    )


def all_three_better(candidate: dict, control: dict) -> bool:
    return bool(
        candidate["tau_x100"] > control["tau_x100"]
        and candidate["mae"] < control["mae"]
        and candidate["rmse"] < control["rmse"]
    )


def improvement(candidate: dict, control: dict) -> dict:
    return {
        "tau_gain": candidate["tau_x100"] - control["tau_x100"],
        "mae_reduction": control["mae"] - candidate["mae"],
        "rmse_reduction": control["rmse"] - candidate["rmse"],
        "tail10_rmse_reduction": (
            control["tail10_rmse"] - candidate["tail10_rmse"]
        ),
        "p95_absolute_error_reduction": (
            control["p95_absolute_error"]
            - candidate["p95_absolute_error"]
        ),
        "all_three_better": all_three_better(candidate, control),
    }


def print_metrics(label: str, values: dict) -> None:
    print(
        f"{label:<40} Tau-{values['tau_variant']}={values['tau_x100']:.6f}  "
        f"MAE={values['mae']:.6f}  RMSE={values['rmse']:.6f}  "
        f"bias={values['bias']:+.6f}  tail10={values['tail10_rmse']:.6f}",
        flush=True,
    )


def metric_row(label: str, values: dict) -> str:
    return (
        f"| {label} | {values['tau_x100']:.6f} | {values['mae']:.6f} | "
        f"{values['rmse']:.6f} | {values['bias']:+.6f} |"
    )


def normalized_base(
    cache: dict,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (cache["base_features"].float() - mean) / std


def make_hard_pairs(
    cache: dict,
    minimum_gap: float,
    maximum_per_group: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(cache["groups"]):
        groups[str(group)].append(index)
    gold = cache["gold"].numpy().astype(np.float64)
    anchor = cache["anchor"].numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    reversed_count = 0
    for indices in groups.values():
        local: list[tuple[int, int, float, bool]] = []
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                gold_difference = float(gold[left] - gold[right])
                if abs(gold_difference) < minimum_gap:
                    continue
                high, low = (
                    (left, right) if gold_difference > 0 else (right, left)
                )
                anchor_margin = float(anchor[high] - anchor[low])
                reversed_pair = anchor_margin <= 0.0
                hardness = (
                    1.0
                    + 3.0 * float(reversed_pair)
                    + np.exp(-abs(anchor_margin) / 0.08)
                )
                local.append((high, low, float(hardness), reversed_pair))
        if len(local) > maximum_per_group:
            local_weights = np.asarray(
                [row[2] for row in local], dtype=np.float64
            )
            # NumPy's weighted choice is needlessly brittle here: on some
            # versions the normalized float64 vector can still trip the
            # "probabilities do not sum to 1" check.  Draw from a cumulative
            # distribution instead and de-duplicate positions explicitly.
            cumulative = np.cumsum(local_weights)
            selected: set[int] = set()
            attempts = 0
            while (
                len(selected) < maximum_per_group
                and attempts < len(local) * 8
            ):
                draws = rng.random(maximum_per_group * 2) * cumulative[-1]
                positions = np.searchsorted(cumulative, draws, side="right")
                selected.update(
                    min(int(position), len(local) - 1)
                    for position in positions
                )
                attempts += len(positions)
            if len(selected) < maximum_per_group:
                remaining = [
                    index
                    for index in range(len(local))
                    if index not in selected
                ]
                selected.update(
                    remaining[: maximum_per_group - len(selected)]
                )
            chosen = sorted(selected)
            local = [local[index] for index in chosen]
        for high, low, hardness, reversed_pair in local:
            pairs.append((high, low))
            weights.append(hardness)
            reversed_count += int(reversed_pair)
    if not pairs:
        raise RuntimeError("No same-image ranking pairs")
    pair_array = np.asarray(pairs, dtype=np.int64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weight_array).all() or weight_array.sum() <= 0.0:
        raise RuntimeError("Invalid hard-pair sampling weights")
    return pair_array, weight_array, {
        "pairs": int(len(pair_array)),
        "reversed_pairs": int(reversed_count),
        "reversed_rate": float(reversed_count / len(pair_array)),
    }


def weighted_pair_positions(
    rng: np.random.Generator,
    cumulative_weight: np.ndarray,
    count: int,
) -> np.ndarray:
    positions = np.searchsorted(
        cumulative_weight,
        rng.random(count) * cumulative_weight[-1],
        side="right",
    )
    return np.minimum(positions, len(cumulative_weight) - 1)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

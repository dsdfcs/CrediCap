from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.stats
import torch


CACHE_FORMAT = "formal-trijudge-reffleur-polaris-v2"
EXPECTED_COUNTS = {
    "expert": 5664,
    "cf": 47830,
    "composite": 11985,
}
TAU_VARIANTS = {
    "expert": "c",
    "cf": "b",
    "composite": "c",
}
LOCKED_REFFLEUR = {
    "expert": (51.93956622103766, 0.11809792809716875, 0.1739409357289407),
    "cf": (38.80332655737372, 0.10413254609381464, 0.16744927198078657),
    "composite": (64.22286305942502, 0.24526952203234187, 0.32390549211776914),
}
PAPER_RESULTS = {
    "expert": {
        "RefCLIP-S": (53.023, 0.3846, 0.4110),
        "RefPAC-S": (55.863, 0.4398, 0.4643),
        "Polos": (56.454, 0.1038, 0.1334),
        "RefFLEUR": (51.940, 0.1181, 0.1739),
    },
    "cf": {
        "RefCLIP-S": (36.404, 0.5462, 0.5662),
        "RefPAC-S": (37.624, 0.6015, 0.6206),
        "Polos": (37.797, 0.2101, 0.2497),
        "RefFLEUR": (38.803, 0.1041, 0.1674),
    },
    "composite": {
        "RefCLIP-S": (56.275, 0.3282, 0.3953),
        "RefPAC-S": (57.960, 0.3419, 0.4207),
        "Polos": (58.379, 0.2596, 0.3120),
        "RefFLEUR": (64.223, 0.2453, 0.3239),
    },
}


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_split(split: dict, dataset: str) -> None:
    required = {
        "sample_ids",
        "records",
        "gold",
        "image",
        "candidate",
        "references",
        "reference_mask",
        "baseline",
        "candidate_length",
    }
    missing = required - set(split)
    if missing:
        raise RuntimeError(f"Cache split missing fields: {sorted(missing)}")
    count = len(split["sample_ids"])
    expected = EXPECTED_COUNTS[dataset]
    if count != expected:
        raise RuntimeError(f"Incomplete {dataset}: {count} != {expected}")
    if len(split["records"]) != count or split["gold"].shape[0] != count:
        raise RuntimeError(f"{dataset} cache row count mismatch")
    if len(set(split["sample_ids"])) != count:
        raise RuntimeError(f"{dataset} contains duplicate sample IDs")


def load_cache(path: Path, expected_kind: str = "benchmark") -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    cache = torch_load(path)
    if cache.get("format") != CACHE_FORMAT:
        raise RuntimeError(f"Wrong formal cache format: {cache.get('format')}")
    if cache.get("kind") != expected_kind:
        raise RuntimeError(
            f"Wrong cache kind: {cache.get('kind')} != {expected_kind}"
        )
    dataset = cache.get("dataset")
    if dataset not in EXPECTED_COUNTS:
        raise RuntimeError(f"Unknown benchmark dataset: {dataset}")
    validate_split(cache["split"], dataset)
    return cache


def make_batch(
    split: dict,
    indices: Sequence[int] | torch.Tensor | np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long).cpu()
    batch = {}
    for key in (
        "image",
        "candidate",
        "references",
        "reference_mask",
        "baseline",
        "candidate_length",
    ):
        value = split[key].index_select(0, index).to(
            device, non_blocking=True
        )
        if value.is_floating_point():
            value = value.float()
        batch[key] = value
    return batch


def metric_values(prediction: np.ndarray, split: dict) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    gold = split["gold"].numpy().astype(np.float64)
    if prediction.shape != gold.shape:
        raise RuntimeError(
            f"Prediction shape {prediction.shape} != gold {gold.shape}"
        )
    tau_prediction = []
    tau_gold = []
    for score, record in zip(prediction.tolist(), split["records"]):
        ratings = [float(value) for value in record["human_ratings"]]
        tau_prediction.extend([score] * len(ratings))
        tau_gold.extend(ratings)
    variant = TAU_VARIANTS[split["dataset"]]
    tau = scipy.stats.kendalltau(
        tau_prediction, tau_gold, variant=variant
    ).statistic
    difference = prediction - gold
    return {
        "dataset": split["dataset"],
        "n": len(prediction),
        "tau_variant": variant,
        "tau_x100": float(tau * 100.0),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "bias": float(np.mean(difference)),
        "pred_mean": float(np.mean(prediction)),
        "gold_mean": float(np.mean(gold)),
    }


def verify_locked_baseline(dataset: str, values: dict) -> None:
    target = LOCKED_REFFLEUR[dataset]
    actual = (values["tau_x100"], values["mae"], values["rmse"])
    tolerance = (0.03, 0.0005, 0.0005)
    failures = [
        f"{name}={value:.6f}, expected={wanted:.6f}"
        for name, value, wanted, allowed in zip(
            ("tau", "mae", "rmse"), actual, target, tolerance
        )
        if abs(value - wanted) > allowed
    ]
    if failures:
        raise RuntimeError(
            f"Wrong {dataset} RefFLEUR baseline: " + "; ".join(failures)
        )


def print_metrics(label: str, values: dict) -> None:
    print(
        f"{label:<30} Tau-{values['tau_variant']}="
        f"{values['tau_x100']:.6f}  MAE={values['mae']:.6f}  "
        f"RMSE={values['rmse']:.6f}  bias={values['bias']:+.6f}",
        flush=True,
    )

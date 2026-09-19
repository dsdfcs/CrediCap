#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from credicap.reference_training import ReferenceCredibilityTrainingModel


ROOT = Path(__file__).resolve().parent
CACHE_FORMAT = "formal-trijudge-reffleur-polaris-v2"
EXPECTED_BENCHMARK_COUNTS = {
    "expert": 5664,
    "cf": 47830,
    "composite": 11985,
}
TAU_VARIANTS = {
    "polaris": "b",
    "expert": "c",
    "cf": "b",
    "composite": "c",
}
LOCKED_REFFLEUR = {
    "expert": {
        "tau_x100": 51.93956622103766,
        "mae": 0.11809792809716875,
        "rmse": 0.1739409357289407,
    },
    "cf": {
        "tau_x100": 38.80332655737372,
        "mae": 0.10413254609381464,
        "rmse": 0.16744927198078657,
    },
    "composite": {
        "tau_x100": 64.22286305942502,
        "mae": 0.24526952203234187,
        "rmse": 0.32390549211776914,
    },
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
FLOAT_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


@dataclass(frozen=True)
class MetricSample:
    sample_id: str
    dataset: str
    subset: str
    image_path: Path
    image_label: str
    candidate: str
    references: tuple[str, ...]
    human_ratings: tuple[float, ...]
    normalized_gold: float
    group: str

    def result_fields(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "subset": self.subset,
            "image": self.image_label,
            "candidate": self.candidate,
            "references": list(self.references),
            "human_ratings": list(self.human_ratings),
            "human_score_normalized": self.normalized_gold,
            "group": self.group,
        }


def clean_text(value: object) -> str:
    return " ".join(str(value).split())


def normalize_text(value: str) -> str:
    return clean_text(value).lower()


def normalize_image(value: str | Path) -> str:
    return Path(value).name.lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chunks(values: Sequence[int], size: int) -> Iterator[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def parse_references(value: str) -> tuple[str, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse Polaris refs: {value[:160]!r}") from exc
    if not isinstance(parsed, (list, tuple)):
        raise RuntimeError("Polaris refs must be a list")
    result = tuple(clean_text(item) for item in parsed if clean_text(item))
    if not result:
        raise RuntimeError("Polaris row has no references")
    return result


def load_polaris_csv(
    path: Path,
    images_dir: Path,
    split: str,
) -> list[MetricSample]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    samples: list[MetricSample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"mt", "refs", "score", "imgid"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
        for index, row in enumerate(reader):
            score = float(row["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RuntimeError(
                    f"Invalid normalized Polaris score at row {index}: {score}"
                )
            image_label = clean_text(row["imgid"])
            image_path = images_dir / image_label
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing Polaris image at row {index}: {image_path}"
                )
            samples.append(
                MetricSample(
                    sample_id=f"polaris:{split}:{index:06d}",
                    dataset="polaris",
                    subset=split,
                    image_path=image_path,
                    image_label=image_label,
                    candidate=clean_text(row["mt"]),
                    references=parse_references(row["refs"]),
                    human_ratings=(score,),
                    normalized_gold=score,
                    group=f"polaris:{normalize_image(image_label)}",
                )
            )
    if not samples:
        raise RuntimeError(f"Empty Polaris CSV: {path}")
    return samples


def resolve_benchmark_image(
    dataset: str,
    subset: str,
    image_label: str,
    datasets_root: Path,
) -> Path:
    if dataset in {"expert", "cf"} or subset == "flickr8k":
        directory = datasets_root / "flickr8k"
    elif subset == "flickr30k":
        directory = datasets_root / "flickr30k"
    elif subset == "coco":
        directory = datasets_root / "coco2014"
    else:
        raise RuntimeError(f"Unknown benchmark image subset: {dataset}/{subset}")
    return directory / image_label


def load_expert(annotations_dir: Path, datasets_root: Path) -> list[MetricSample]:
    path = annotations_dir / "flickr8k.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: list[MetricSample] = []
    for image_key, item in data.items():
        rows = [
            row
            for row in item["human_judgement"]
            if not math.isnan(float(row["rating"]))
        ]
        if len(rows) % 3:
            raise RuntimeError(f"Expert ratings are not triplets: {image_key}")
        references = tuple(clean_text(text) for text in item["ground_truth"])
        for offset in range(0, len(rows), 3):
            block = rows[offset : offset + 3]
            captions = {clean_text(row["caption"]) for row in block}
            image_labels = {row["image_path"] for row in block}
            if len(captions) != 1 or len(image_labels) != 1:
                raise RuntimeError(f"Inconsistent Expert triplet: {image_key}")
            ratings = tuple(float(row["rating"]) for row in block)
            image_label = next(iter(image_labels))
            samples.append(
                MetricSample(
                    sample_id=f"expert:{image_key}:{offset // 3}",
                    dataset="expert",
                    subset="flickr8k",
                    image_path=resolve_benchmark_image(
                        "expert", "flickr8k", image_label, datasets_root
                    ),
                    image_label=image_label,
                    candidate=next(iter(captions)),
                    references=references,
                    human_ratings=ratings,
                    normalized_gold=(float(np.mean(ratings)) - 1.0) / 3.0,
                    group=f"flickr8k:{normalize_image(image_label)}",
                )
            )
    return samples


def load_cf(annotations_dir: Path, datasets_root: Path) -> list[MetricSample]:
    path = annotations_dir / "crowdflower_flickr8k.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: list[MetricSample] = []
    for image_key, item in data.items():
        references = tuple(clean_text(text) for text in item["ground_truth"])
        for row_index, row in enumerate(item["human_judgement"]):
            rating = float(row["rating"])
            if math.isnan(rating):
                continue
            image_label = row["image_path"]
            samples.append(
                MetricSample(
                    sample_id=f"cf:{image_key}:{row_index}",
                    dataset="cf",
                    subset="flickr8k",
                    image_path=resolve_benchmark_image(
                        "cf", "flickr8k", image_label, datasets_root
                    ),
                    image_label=image_label,
                    candidate=clean_text(row["caption"]),
                    references=references,
                    human_ratings=(rating,),
                    normalized_gold=rating,
                    group=f"flickr8k:{normalize_image(image_label)}",
                )
            )
    return samples


def load_composite(
    annotations_dir: Path,
    datasets_root: Path,
) -> list[MetricSample]:
    path = annotations_dir / "composite.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: list[MetricSample] = []
    for subset, rows in data.items():
        for row_index, row in enumerate(rows):
            rating = float(row["human"])
            image_label = row["image"]
            samples.append(
                MetricSample(
                    sample_id=f"composite:{subset}:{row_index}",
                    dataset="composite",
                    subset=subset,
                    image_path=resolve_benchmark_image(
                        "composite", subset, image_label, datasets_root
                    ),
                    image_label=image_label,
                    candidate=clean_text(row["caption"]),
                    references=tuple(clean_text(text) for text in row["reference"]),
                    human_ratings=(rating,),
                    normalized_gold=(rating - 1.0) / 4.0,
                    group=f"{subset}:{normalize_image(image_label)}",
                )
            )
    return samples


def load_benchmark(
    dataset: str,
    annotations_dir: Path,
    datasets_root: Path,
) -> list[MetricSample]:
    if dataset == "expert":
        samples = load_expert(annotations_dir, datasets_root)
    elif dataset == "cf":
        samples = load_cf(annotations_dir, datasets_root)
    else:
        samples = load_composite(annotations_dir, datasets_root)
    expected = EXPECTED_BENCHMARK_COUNTS[dataset]
    if len(samples) != expected:
        raise RuntimeError(f"{dataset} count {len(samples)} != {expected}")
    identifiers = [sample.sample_id for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"Duplicate sample IDs in {dataset}")
    return samples


def parse_original_reffleur(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    parsed: list[dict] = []
    image: str | None = None
    candidate: str | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("image file :"):
                image = line.split(":", 1)[1].strip()
            elif line.startswith("caption :"):
                candidate = line.split(":", 1)[1].strip()
            elif line.startswith("our score :"):
                if image is None or candidate is None:
                    raise RuntimeError(f"Malformed RefFLEUR result: {path}")
                match = FLOAT_PATTERN.search(line.split(":", 1)[1])
                if match is None:
                    raise RuntimeError(f"Cannot parse score: {line}")
                score = float(match.group(0))
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise RuntimeError(f"Invalid RefFLEUR score: {score}")
                parsed.append({"image": image, "candidate": candidate, "score": score})
                image = None
                candidate = None
    if not parsed:
        raise RuntimeError(f"No scores in {path}")
    return parsed


def align_benchmark_baseline(
    samples: Sequence[MetricSample],
    path: Path,
) -> np.ndarray:
    pools: dict[tuple[str, str], deque[float]] = defaultdict(deque)
    for row in parse_original_reffleur(path):
        pools[(normalize_image(row["image"]), normalize_text(row["candidate"]))].append(
            float(row["score"])
        )
    aligned: list[float] = []
    missing: list[tuple[str, str]] = []
    for sample in samples:
        key = (normalize_image(sample.image_label), normalize_text(sample.candidate))
        if not pools[key]:
            missing.append(key)
        else:
            aligned.append(pools[key].popleft())
    leftovers = sum(len(values) for values in pools.values())
    if missing or len(aligned) != len(samples) or leftovers:
        raise RuntimeError(
            "RefFLEUR result does not align one-to-one: "
            f"samples={len(samples)} aligned={len(aligned)} "
            f"missing={len(missing)} leftovers={leftovers} preview={missing[:3]}"
        )
    return np.asarray(aligned, dtype=np.float32)


def align_polaris_baseline(
    samples: Sequence[MetricSample],
    path: Path,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    mapping: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in mapping:
                raise RuntimeError(f"Duplicate {sample_id} in {path}:{line_number}")
            score = float(row["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RuntimeError(f"Invalid score in {path}:{line_number}")
            mapping[sample_id] = score
    expected = [sample.sample_id for sample in samples]
    missing = [sample_id for sample_id in expected if sample_id not in mapping]
    unexpected = set(mapping) - set(expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Polaris baseline mismatch: missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
    return np.asarray([mapping[sample_id] for sample_id in expected], dtype=np.float32)


def metric_values(prediction: np.ndarray, split: dict) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    gold = split["gold"].numpy().astype(np.float64)
    if prediction.shape != gold.shape:
        raise RuntimeError(f"Prediction shape {prediction.shape} != gold {gold.shape}")
    tau_prediction: list[float] = []
    tau_gold: list[float] = []
    for score, record in zip(prediction.tolist(), split["records"]):
        ratings = [float(value) for value in record["human_ratings"]]
        tau_prediction.extend([score] * len(ratings))
        tau_gold.extend(ratings)
    variant = TAU_VARIANTS[split["dataset"]]
    tau = scipy.stats.kendalltau(
        tau_prediction,
        tau_gold,
        variant=variant,
    )[0]
    pearson = scipy.stats.pearsonr(prediction, gold)[0]
    difference = prediction - gold
    return {
        "dataset": split["dataset"],
        "n": len(prediction),
        "tau_variant": variant,
        "tau_x100": float(tau * 100.0),
        "pearson_x100": float(pearson * 100.0),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "bias": float(np.mean(difference)),
        "pred_mean": float(np.mean(prediction)),
        "gold_mean": float(np.mean(gold)),
    }


def print_metrics(label: str, values: dict) -> None:
    print(
        f"{label:<30} Tau-{values['tau_variant']}={values['tau_x100']:.6f}  "
        f"MAE={values['mae']:.6f}  RMSE={values['rmse']:.6f}  "
        f"bias={values['bias']:+.6f}",
        flush=True,
    )


def verify_locked_baseline(dataset: str, values: dict) -> None:
    expected = LOCKED_REFFLEUR[dataset]
    tolerances = {"tau_x100": 0.03, "mae": 0.0005, "rmse": 0.0005}
    failures = []
    for key, target in expected.items():
        if abs(values[key] - target) > tolerances[key]:
            failures.append(f"{key}={values[key]:.6f} expected {target:.6f}")
    if failures:
        raise RuntimeError(
            f"Wrong {dataset} RefFLEUR baseline: " + "; ".join(failures)
        )


def encode_texts(
    texts: Sequence[str],
    model: object,
    processor: object,
    device: torch.device,
    batch_size: int,
    description: str,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    positions = list(range(len(texts)))
    for block in tqdm(
        list(chunks(positions, batch_size)),
        desc=description,
        dynamic_ncols=True,
    ):
        encoded = processor(
            text=[texts[index] for index in block],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            feature = model.get_text_features(**inputs)
            feature = F.normalize(feature.float(), dim=-1)
        values.append(feature.cpu().half())
    return torch.cat(values)


def encode_images(
    paths: Sequence[Path],
    model: object,
    processor: object,
    device: torch.device,
    batch_size: int,
    description: str,
) -> torch.Tensor:
    from PIL import Image

    values: list[torch.Tensor] = []
    positions = list(range(len(paths)))
    for block in tqdm(
        list(chunks(positions, batch_size)),
        desc=description,
        dynamic_ncols=True,
    ):
        images = []
        for index in block:
            path = paths[index]
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(
            device
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            feature = model.get_image_features(pixel_values=pixels)
            feature = F.normalize(feature.float(), dim=-1)
        values.append(feature.cpu().half())
    return torch.cat(values)


def build_feature_split(
    samples: Sequence[MetricSample],
    baseline: np.ndarray,
    model: object,
    processor: object,
    device: torch.device,
    text_batch_size: int,
    image_batch_size: int,
    description: str,
) -> dict:
    unique_texts: list[str] = []
    text_lookup: dict[str, int] = {}
    unique_images: list[Path] = []
    image_lookup: dict[str, int] = {}
    candidate_positions: list[int] = []
    reference_positions: list[list[int]] = []
    image_positions: list[int] = []

    def text_position(text: str) -> int:
        normalized = clean_text(text)
        if normalized not in text_lookup:
            text_lookup[normalized] = len(unique_texts)
            unique_texts.append(normalized)
        return text_lookup[normalized]

    for sample in samples:
        image_key = str(sample.image_path.resolve())
        if image_key not in image_lookup:
            image_lookup[image_key] = len(unique_images)
            unique_images.append(sample.image_path)
        image_positions.append(image_lookup[image_key])
        candidate_positions.append(text_position(sample.candidate))
        reference_positions.append(
            [text_position(reference) for reference in sample.references]
        )

    print(
        f"{description}: rows={len(samples)} groups={len(set(s.group for s in samples))} "
        f"unique_images={len(unique_images)} unique_texts={len(unique_texts)}",
        flush=True,
    )
    text_features = encode_texts(
        unique_texts,
        model,
        processor,
        device,
        text_batch_size,
        f"CLIP text {description}",
    )
    image_features = encode_images(
        unique_images,
        model,
        processor,
        device,
        image_batch_size,
        f"CLIP image {description}",
    )
    max_references = max(len(values) for values in reference_positions)
    dimension = int(text_features.shape[1])
    references = torch.zeros(
        len(samples),
        max_references,
        dimension,
        dtype=torch.float16,
    )
    reference_mask = torch.zeros(
        len(samples),
        max_references,
        dtype=torch.bool,
    )
    for index, positions in enumerate(reference_positions):
        count = len(positions)
        references[index, :count] = text_features[positions]
        reference_mask[index, :count] = True

    return {
        "dataset": samples[0].dataset,
        "sample_ids": [sample.sample_id for sample in samples],
        "groups": [sample.group for sample in samples],
        "records": [sample.result_fields() for sample in samples],
        "gold": torch.tensor(
            [sample.normalized_gold for sample in samples],
            dtype=torch.float32,
        ),
        "baseline": torch.from_numpy(baseline.copy()),
        "image": image_features[image_positions],
        "candidate": text_features[candidate_positions],
        "references": references,
        "reference_mask": reference_mask,
        "candidate_length": torch.tensor(
            [min(len(sample.candidate.split()) / 40.0, 1.5) for sample in samples],
            dtype=torch.float32,
        ),
        "embedding_dim": dimension,
    }


def load_clip(args: argparse.Namespace) -> tuple[object, object, torch.device]:
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Loading frozen CLIP evidence encoder: {args.clip_model}", flush=True)
    processor = CLIPProcessor.from_pretrained(
        args.clip_model,
        local_files_only=True,
    )
    model = CLIPModel.from_pretrained(
        args.clip_model,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    return model, processor, device


def save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    torch.save(cache, temporary)
    temporary.replace(path)
    print(f"Cache saved: {path} ({path.stat().st_size / 2**30:.2f} GiB)", flush=True)


def prepare_polaris(args: argparse.Namespace) -> None:
    if args.cache.exists() and not args.overwrite:
        cache = load_cache(args.cache, expected_kind="polaris")
        print(f"Valid Polaris cache already exists: {args.cache}")
        print_metrics("Polaris train RefFLEUR", cache["train_baseline_metrics"])
        print_metrics("Polaris val RefFLEUR", cache["val_baseline_metrics"])
        return
    train_csv = args.polaris_dir / "polaris_train.csv"
    val_csv = args.polaris_dir / "polaris_val.csv"
    images_dir = args.polaris_dir / "images"
    train_samples = load_polaris_csv(train_csv, images_dir, "train")
    val_samples = load_polaris_csv(val_csv, images_dir, "val")
    train_baseline = align_polaris_baseline(train_samples, args.train_baseline)
    val_baseline = align_polaris_baseline(val_samples, args.val_baseline)

    model, processor, device = load_clip(args)
    try:
        train_split = build_feature_split(
            train_samples,
            train_baseline,
            model,
            processor,
            device,
            args.text_batch_size,
            args.image_batch_size,
            "Polaris train",
        )
        val_split = build_feature_split(
            val_samples,
            val_baseline,
            model,
            processor,
            device,
            args.text_batch_size,
            args.image_batch_size,
            "Polaris val",
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_metrics = metric_values(train_baseline, train_split)
    val_metrics = metric_values(val_baseline, val_split)
    print_metrics("Polaris train RefFLEUR", train_metrics)
    print_metrics("Polaris val RefFLEUR", val_metrics)
    signature_data = {
        "format": CACHE_FORMAT,
        "kind": "polaris",
        "train_csv_sha256": sha256_file(train_csv),
        "val_csv_sha256": sha256_file(val_csv),
        "train_baseline_sha256": sha256_file(args.train_baseline),
        "val_baseline_sha256": sha256_file(args.val_baseline),
        "clip_model": str(args.clip_model),
        "runner_sha256": sha256_file(Path(__file__)),
        "model_sha256": sha256_file(ROOT / "trijudge_formal_model.py"),
    }
    cache = {
        "format": CACHE_FORMAT,
        "kind": "polaris",
        "created_unix": time.time(),
        "signature": sha256_text(json.dumps(signature_data, sort_keys=True)),
        "signature_data": signature_data,
        "clip_model": str(args.clip_model),
        "embedding_dim": int(train_split["embedding_dim"]),
        "train": train_split,
        "val": val_split,
        "train_baseline_metrics": train_metrics,
        "val_baseline_metrics": val_metrics,
        "protocol": {
            "train": "official Polaris train split",
            "model_selection": "official Polaris validation split",
            "polaris_test_used": False,
            "benchmark_labels_used": False,
            "cross_validation": False,
            "seed_ensemble": False,
            "reffleur_backbone": "LLaVA-v1.5-13B exact released prompt/smoothing",
            "evidence_encoder": "frozen CLIP ViT-L/14@336",
        },
    }
    validate_polaris_cache(cache)
    save_cache(cache, args.cache)


def prepare_benchmark(args: argparse.Namespace) -> None:
    if args.cache.exists() and not args.overwrite:
        cache = load_cache(args.cache, expected_kind="benchmark")
        if cache["dataset"] != args.dataset:
            raise RuntimeError(
                f"Cache dataset {cache['dataset']} != requested {args.dataset}"
            )
        print(f"Valid {args.dataset} cache already exists: {args.cache}")
        print_metrics("Locked RefFLEUR", cache["baseline_metrics"])
        return
    samples = load_benchmark(args.dataset, args.annotations_dir, args.datasets_root)
    baseline = align_benchmark_baseline(samples, args.baseline_result)
    model, processor, device = load_clip(args)
    try:
        split = build_feature_split(
            samples,
            baseline,
            model,
            processor,
            device,
            args.text_batch_size,
            args.image_batch_size,
            args.dataset,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline_metrics = metric_values(baseline, split)
    print_metrics("Locked RefFLEUR", baseline_metrics)
    verify_locked_baseline(args.dataset, baseline_metrics)
    annotation_file = {
        "expert": args.annotations_dir / "flickr8k.json",
        "cf": args.annotations_dir / "crowdflower_flickr8k.json",
        "composite": args.annotations_dir / "composite.json",
    }[args.dataset]
    signature_data = {
        "format": CACHE_FORMAT,
        "kind": "benchmark",
        "dataset": args.dataset,
        "annotation_sha256": sha256_file(annotation_file),
        "baseline_sha256": sha256_file(args.baseline_result),
        "clip_model": str(args.clip_model),
    }
    cache = {
        "format": CACHE_FORMAT,
        "kind": "benchmark",
        "dataset": args.dataset,
        "created_unix": time.time(),
        "signature": sha256_text(json.dumps(signature_data, sort_keys=True)),
        "signature_data": signature_data,
        "clip_model": str(args.clip_model),
        "embedding_dim": int(split["embedding_dim"]),
        "split": split,
        "baseline_metrics": baseline_metrics,
        "protocol": {
            "dataset": args.dataset,
            "published_split": True,
            "training": False,
            "checkpoint_selection": False,
            "labels_used_only_after_prediction": True,
        },
    }
    validate_benchmark_cache(cache)
    save_cache(cache, args.cache)


def validate_split(split: dict) -> None:
    required = {
        "dataset",
        "sample_ids",
        "groups",
        "records",
        "gold",
        "baseline",
        "image",
        "candidate",
        "references",
        "reference_mask",
        "candidate_length",
        "embedding_dim",
    }
    missing = required - set(split)
    if missing:
        raise RuntimeError(f"Cache split missing: {sorted(missing)}")
    count = len(split["sample_ids"])
    if not count or len(set(split["sample_ids"])) != count:
        raise RuntimeError("Empty or duplicate cache sample IDs")
    for key in (
        "groups",
        "records",
        "gold",
        "baseline",
        "image",
        "candidate",
        "references",
        "reference_mask",
        "candidate_length",
    ):
        if len(split[key]) != count:
            raise RuntimeError(f"Wrong split length for {key}: {len(split[key])} != {count}")
    if split["references"].shape[:2] != split["reference_mask"].shape:
        raise RuntimeError("Reference tensor/mask mismatch")


def validate_polaris_cache(cache: dict) -> None:
    if cache.get("format") != CACHE_FORMAT or cache.get("kind") != "polaris":
        raise RuntimeError("Not a formal Polaris cache")
    validate_split(cache["train"])
    validate_split(cache["val"])
    if cache["train"]["dataset"] != "polaris" or cache["val"]["dataset"] != "polaris":
        raise RuntimeError("Wrong Polaris cache dataset")
    if int(cache["embedding_dim"]) != int(cache["train"]["embedding_dim"]):
        raise RuntimeError("Polaris embedding dimension mismatch")
    # Use the official split exactly as supplied. Report image overlap rather
    # than silently resplitting it into folds.
    train_groups = set(cache["train"]["groups"])
    val_groups = set(cache["val"]["groups"])
    cache["protocol"]["official_train_val_image_overlap"] = len(
        train_groups & val_groups
    )


def validate_benchmark_cache(cache: dict) -> None:
    if cache.get("format") != CACHE_FORMAT or cache.get("kind") != "benchmark":
        raise RuntimeError("Not a formal benchmark cache")
    validate_split(cache["split"])
    dataset = cache["dataset"]
    expected = EXPECTED_BENCHMARK_COUNTS[dataset]
    if len(cache["split"]["sample_ids"]) != expected:
        raise RuntimeError(f"Incomplete {dataset} cache")
    if cache["split"]["dataset"] != dataset:
        raise RuntimeError("Benchmark dataset mismatch")


def load_cache(path: Path, expected_kind: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    cache = torch_load(path)
    if cache.get("format") != CACHE_FORMAT:
        raise RuntimeError(
            f"Wrong cache format {cache.get('format')}; old OOF caches are blocked"
        )
    if expected_kind is not None and cache.get("kind") != expected_kind:
        raise RuntimeError(f"Cache kind {cache.get('kind')} != {expected_kind}")
    if cache["kind"] == "polaris":
        validate_polaris_cache(cache)
    else:
        validate_benchmark_cache(cache)
    return cache


def make_batch(
    split: dict,
    indices: Sequence[int] | torch.Tensor | np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long)
    batch = {}
    for key in (
        "image",
        "candidate",
        "references",
        "reference_mask",
        "baseline",
        "candidate_length",
    ):
        value = split[key].index_select(0, index).to(device, non_blocking=True)
        if value.is_floating_point():
            value = value.float()
        batch[key] = value
    return batch


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    split: dict,
    batch_size: int,
    device: torch.device,
    amp: bool,
    description: str,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.arange(len(split["sample_ids"]))),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    output: list[torch.Tensor] = []
    for (indices,) in tqdm(loader, desc=description, dynamic_ncols=True, leave=False):
        batch = make_batch(split, indices, device)
        with torch.cuda.amp.autocast(enabled=amp):
            output.append(model(batch).score.float().cpu())
    return torch.cat(output).numpy()


def metric_objective(values: dict) -> float:
    # A 0.01 MAE reduction is worth 0.60 Tau points, and a 0.01 RMSE
    # reduction is worth 0.30 Tau points. Tau remains the primary signal.
    return float(values["tau_x100"] - 60.0 * values["mae"] - 30.0 * values["rmse"])


def make_ranking_pairs(
    split: dict,
    minimum_gap: float,
    maximum_per_group: int,
    seed: int,
) -> np.ndarray:
    by_group: dict[str, list[int]] = defaultdict(list)
    gold = split["gold"].numpy()
    for index, group in enumerate(split["groups"]):
        by_group[group].append(index)
    generator = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    for indices in by_group.values():
        local: list[tuple[int, int]] = []
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                difference = float(gold[left] - gold[right])
                if abs(difference) < minimum_gap:
                    continue
                local.append((left, right) if difference > 0 else (right, left))
        if len(local) > maximum_per_group:
            selected = generator.choice(
                len(local),
                size=maximum_per_group,
                replace=False,
            )
            local = [local[int(index)] for index in selected]
        pairs.extend(local)
    if not pairs:
        raise RuntimeError("No Polaris ranking pairs were created")
    return np.asarray(pairs, dtype=np.int64)


def model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ReferenceCredibilityTrainingModel, dict]:
    payload = torch_load(checkpoint_path)
    required = {"model", "stage", "embedding_dim", "hidden_dim", "dropout"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {sorted(missing)}")
    model = ReferenceCredibilityTrainingModel(
        input_dim=int(payload["embedding_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        dropout=float(payload["dropout"]),
        stage=int(payload["stage"]),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def training_signature(
    args: argparse.Namespace,
    cache: dict,
    previous_checkpoint: Path | None,
) -> str:
    values = {
        "cache_signature": cache["signature"],
        "stage": args.stage,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "rank_minimum_gap": args.rank_minimum_gap,
        "rank_maximum_per_group": args.rank_maximum_per_group,
        "rank_loss_weight": args.rank_loss_weight,
        "reference_corruption_weight": args.reference_corruption_weight,
        "runner_sha256": sha256_file(Path(__file__)),
        "model_sha256": sha256_file(ROOT / "trijudge_formal_model.py"),
        "previous_checkpoint_sha256": (
            sha256_file(previous_checkpoint) if previous_checkpoint else None
        ),
    }
    return sha256_text(json.dumps(values, sort_keys=True))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cache = load_cache(args.polaris_cache, expected_kind="polaris")
    train_split = cache["train"]
    val_split = cache["val"]
    device = torch.device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / f"formal_stage{args.stage}.best.pt"
    report_path = args.run_dir / f"formal_stage{args.stage}.validation.json"
    previous_path = (
        args.run_dir / f"formal_stage{args.stage - 1}.best.pt"
        if args.stage > 1
        else None
    )
    if previous_path is not None and not previous_path.is_file():
        raise FileNotFoundError(
            f"Stage {args.stage} requires the frozen Stage {args.stage - 1}: "
            f"{previous_path}"
        )
    signature = training_signature(args, cache, previous_path)
    if checkpoint_path.exists() or report_path.exists():
        if not checkpoint_path.is_file() or not report_path.is_file():
            raise RuntimeError("Partial existing stage output; inspect it before rerunning")
        payload = torch_load(checkpoint_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if payload.get("training_signature") == signature and report.get("training_signature") == signature:
            print(f"Stage {args.stage} already complete and signature-valid")
            print_metrics("Best Polaris validation", report["model"])
            return
        if not args.overwrite:
            raise RuntimeError(
                f"Existing Stage {args.stage} output has a different signature; "
                "use a new run directory or --overwrite"
            )

    model = ReferenceCredibilityTrainingModel(
        input_dim=int(cache["embedding_dim"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        stage=args.stage,
    ).to(device)
    if previous_path is not None:
        previous = torch_load(previous_path)
        if previous.get("polaris_cache_signature") != cache["signature"]:
            raise RuntimeError("Previous stage was trained with another Polaris cache")
        if int(previous["stage"]) != args.stage - 1:
            raise RuntimeError("Wrong previous-stage checkpoint")
        if int(previous["hidden_dim"]) != args.hidden_dim:
            raise RuntimeError("hidden-dim changed between stages")
        model.load_state_dict(previous["model"], strict=True)
        print(f"Loaded frozen previous stage: {previous_path}", flush=True)
    model.set_trainable_stage(args.stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if not trainable_count:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loader = DataLoader(
        TensorDataset(torch.arange(len(train_split["sample_ids"]))),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + args.stage),
    )

    ranking_pairs = None
    pair_generator = np.random.default_rng(args.seed + 13007)
    if args.stage == 3:
        ranking_pairs = make_ranking_pairs(
            train_split,
            minimum_gap=args.rank_minimum_gap,
            maximum_per_group=args.rank_maximum_per_group,
            seed=args.seed,
        )
        print(f"Polaris same-image ranking pairs: {len(ranking_pairs)}", flush=True)

    if args.stage == 1:
        reference_metrics = cache["val_baseline_metrics"]
        reference_label = "RefFLEUR"
    else:
        previous_model, _ = model_from_checkpoint(previous_path, device)
        previous_prediction = predict(
            previous_model,
            val_split,
            args.eval_batch_size,
            device,
            amp,
            f"Stage {args.stage - 1} validation reference",
        )
        reference_metrics = metric_values(previous_prediction, val_split)
        reference_label = f"Stage {args.stage - 1}"
        del previous_model

    initial_prediction = predict(
        model,
        val_split,
        args.eval_batch_size,
        device,
        amp,
        f"Stage {args.stage} initialization check",
    )
    maximum_initial_difference = float(
        np.max(
            np.abs(
                initial_prediction
                - (
                    val_split["baseline"].numpy()
                    if args.stage == 1
                    else previous_prediction
                )
            )
        )
    )
    if maximum_initial_difference > 2.0e-5:
        raise RuntimeError(
            f"Stage {args.stage} does not start from {reference_label}: "
            f"max difference={maximum_initial_difference}"
        )

    print("=" * 92)
    print(f"FORMAL EXTERNAL TRAINING — STAGE {args.stage}")
    print("=" * 92)
    print("Train split       : official Polaris train")
    print("Selection split   : official Polaris validation")
    print("Polaris test      : NOT USED")
    print("Expert/CF/Composite labels: NOT LOADED BY THIS PROCESS")
    print("Five-fold OOF     : NO")
    print("Seed ensemble     : NO")
    print(f"Trainable params  : {trainable_count:,}")
    print_metrics(f"Validation {reference_label}", reference_metrics)

    best_objective = metric_objective(reference_metrics)
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    best_metrics = reference_metrics
    best_epoch = 0
    stale = 0
    gold = train_split["gold"]

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.stage >= 2:
            model.reference_trust.eval()
        if args.stage >= 3:
            model.evidence_router.eval()
        total_loss = 0.0
        total_point = 0.0
        total_rank = 0.0
        total_reference = 0.0
        seen = 0
        progress = tqdm(
            loader,
            desc=f"Formal Stage {args.stage} epoch {epoch:02d}",
            dynamic_ncols=True,
        )
        for (indices,) in progress:
            batch = make_batch(train_split, indices, device)
            target = gold.index_select(0, indices).to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                score = model(batch).score
                point_loss = (
                    0.65 * F.smooth_l1_loss(score, target, beta=0.10)
                    + 0.35 * F.mse_loss(score, target)
                )
                anchor_loss = F.mse_loss(score, batch["baseline"])
                loss = point_loss + args.anchor_loss_weight * anchor_loss
                rank_loss = torch.zeros((), device=device)
                reference_loss = torch.zeros((), device=device)
                if args.stage == 1 and len(indices) > 1:
                    corrupted = dict(batch)
                    corrupted_references = batch["references"].clone()
                    valid_counts = batch["reference_mask"].sum(dim=1).long()
                    corrupt_slots = torch.floor(
                        torch.rand(len(indices), device=device)
                        * valid_counts.clamp_min(1).float()
                    ).long()
                    donors = torch.roll(batch["references"][:, 0], shifts=1, dims=0)
                    rows = torch.arange(len(indices), device=device)
                    corrupted_references[rows, corrupt_slots] = donors
                    corrupted["references"] = corrupted_references
                    corrupted_output = model(corrupted)
                    bad_weights = corrupted_output.reference_weights[rows, corrupt_slots]
                    reference_loss = bad_weights.mean()
                    loss = loss + args.reference_corruption_weight * reference_loss
                if ranking_pairs is not None:
                    chosen = pair_generator.choice(
                        len(ranking_pairs),
                        size=min(args.pair_batch_size, len(ranking_pairs)),
                        replace=False,
                    )
                    pair = ranking_pairs[chosen]
                    better = make_batch(train_split, pair[:, 0], device)
                    worse = make_batch(train_split, pair[:, 1], device)
                    better_score = model(better).score
                    worse_score = model(worse).score
                    rank_loss = F.softplus(
                        (worse_score - better_score) / args.rank_temperature
                    ).mean()
                    loss = loss + args.rank_loss_weight * rank_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            count = len(indices)
            seen += count
            total_loss += float(loss.detach()) * count
            total_point += float(point_loss.detach()) * count
            total_rank += float(rank_loss.detach()) * count
            total_reference += float(reference_loss.detach()) * count
            progress.set_postfix(loss=f"{float(loss.detach()):.5f}")

        prediction = predict(
            model,
            val_split,
            args.eval_batch_size,
            device,
            amp,
            f"Polaris validation epoch {epoch:02d}",
        )
        values = metric_values(prediction, val_split)
        objective = metric_objective(values)
        print(
            f"stage={args.stage} epoch={epoch:02d} "
            f"loss={total_loss / seen:.6f} point={total_point / seen:.6f} "
            f"ref={total_reference / seen:.6f} rank={total_rank / seen:.6f} "
            f"objective={objective:.6f}",
            flush=True,
        )
        print_metrics("Polaris validation", values)
        if objective > best_objective + args.minimum_objective_gain:
            best_objective = objective
            best_metrics = values
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    validation_pass = bool(
        best_metrics["tau_x100"] >= reference_metrics["tau_x100"] + args.minimum_tau_gain
        and best_metrics["mae"] < reference_metrics["mae"]
        and best_metrics["rmse"] < reference_metrics["rmse"]
    )
    payload = {
        "format": CACHE_FORMAT,
        "model": best_state,
        "stage": args.stage,
        "embedding_dim": int(cache["embedding_dim"]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "seed": args.seed,
        "seed_ensemble": False,
        "cross_validation": False,
        "training_dataset": "Polaris train",
        "selection_dataset": "Polaris validation",
        "polaris_test_used": False,
        "benchmark_labels_used": False,
        "polaris_train_image_names": sorted(
            {
                normalize_image(record["image"])
                for record in train_split["records"]
            }
        ),
        "polaris_val_image_names": sorted(
            {
                normalize_image(record["image"])
                for record in val_split["records"]
            }
        ),
        "polaris_cache_signature": cache["signature"],
        "clip_model": cache["clip_model"],
        "training_signature": signature,
        "best_epoch": best_epoch,
        "reference_validation": reference_metrics,
        "best_validation": best_metrics,
        "validation_pass": validation_pass,
    }
    temporary = checkpoint_path.with_suffix(".pt.writing")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    report = {
        key: value
        for key, value in payload.items()
        if key != "model"
    }
    report.update(
        {
            "reference_label": reference_label,
            "reference": reference_metrics,
            "model": best_metrics,
            "delta": {
                "tau_x100": best_metrics["tau_x100"] - reference_metrics["tau_x100"],
                "mae_reduction": reference_metrics["mae"] - best_metrics["mae"],
                "rmse_reduction": reference_metrics["rmse"] - best_metrics["rmse"],
            },
            "checkpoint": str(checkpoint_path),
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("=" * 92)
    print(f"STAGE {args.stage} EXTERNAL VALIDATION RESULT")
    print_metrics(reference_label, reference_metrics)
    print_metrics(f"Stage {args.stage}", best_metrics)
    print(f"Best epoch      : {best_epoch}")
    print(f"VALIDATION PASS : {validation_pass}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Report          : {report_path}")


def gate(args: argparse.Namespace) -> None:
    report_path = args.run_dir / f"formal_stage{args.stage}.validation.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print_metrics(report["reference_label"], report["reference"])
    print_metrics(f"Stage {args.stage}", report["model"])
    if not report["validation_pass"]:
        print(
            f"Stage {args.stage} failed on external Polaris validation. "
            "Formal pipeline stops before touching benchmark results.",
            flush=True,
        )
        raise SystemExit(3)
    print(f"Stage {args.stage} passed external validation", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    cache = load_cache(args.benchmark_cache, expected_kind="benchmark")
    dataset = cache["dataset"]
    split = cache["split"]
    device = torch.device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    model, checkpoint = model_from_checkpoint(args.checkpoint, device)
    if checkpoint.get("format") != CACHE_FORMAT:
        raise RuntimeError("Wrong formal checkpoint format")
    if checkpoint.get("benchmark_labels_used") is not False:
        raise RuntimeError("Checkpoint does not certify benchmark-label isolation")
    if checkpoint.get("cross_validation") is not False:
        raise RuntimeError("Cross-validation checkpoint is forbidden")
    if checkpoint.get("seed_ensemble") is not False:
        raise RuntimeError("Seed-ensemble checkpoint is forbidden")
    if checkpoint.get("clip_model") != cache.get("clip_model"):
        raise RuntimeError("Checkpoint/cache CLIP backbone mismatch")
    if int(checkpoint["embedding_dim"]) != int(cache["embedding_dim"]):
        raise RuntimeError("Checkpoint/cache embedding dimension mismatch")
    benchmark_image_names = {
        normalize_image(record["image"])
        for record in split["records"]
    }
    train_image_overlap = benchmark_image_names & set(
        checkpoint.get("polaris_train_image_names", [])
    )
    validation_image_overlap = benchmark_image_names & set(
        checkpoint.get("polaris_val_image_names", [])
    )

    print("=" * 92)
    print(f"FROZEN DIRECT BENCHMARK TEST — {dataset.upper()}")
    print("=" * 92)
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Checkpoint SHA256: {sha256_file(args.checkpoint)}")
    print("Training         : Polaris train only")
    print("Model selection  : Polaris validation only")
    print(f"{dataset} labels : metrics only, after frozen prediction")
    print("Five-fold OOF    : NO")
    print("Seed ensemble    : NO")
    print(f"Filename overlap with Polaris train/val: "
          f"{len(train_image_overlap)}/{len(validation_image_overlap)}")
    prediction = predict(
        model,
        split,
        args.eval_batch_size,
        device,
        amp,
        f"Frozen Stage {checkpoint['stage']} {dataset}",
    )
    model_metrics = metric_values(prediction, split)
    baseline_metrics = cache["baseline_metrics"]
    verify_locked_baseline(dataset, baseline_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, (record, score) in enumerate(zip(split["records"], prediction)):
            row = dict(record)
            row.update(
                {
                    "mode": "formal_polaris_trained_frozen_direct_test",
                    "stage": int(checkpoint["stage"]),
                    "score": float(np.clip(score, 0.0, 1.0)),
                    "reffleur_score": float(split["baseline"][index]),
                    "checkpoint_sha256": sha256_file(args.checkpoint),
                    "benchmark_label_used_for_training": False,
                    "cross_validation": False,
                    "seed_ensemble": False,
                }
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    strict = bool(
        model_metrics["tau_x100"] > baseline_metrics["tau_x100"]
        and model_metrics["mae"] < baseline_metrics["mae"]
        and model_metrics["rmse"] < baseline_metrics["rmse"]
    )
    report = {
        "format": CACHE_FORMAT,
        "dataset": dataset,
        "stage": int(checkpoint["stage"]),
        "protocol": "Polaris external train/val; one frozen checkpoint; published benchmark direct test",
        "training_dataset": "Polaris train",
        "selection_dataset": "Polaris validation",
        "benchmark_labels_used_for_training_or_selection": False,
        "cross_validation": False,
        "seed_ensemble": False,
        "benchmark_filename_overlap_with_polaris_train": len(train_image_overlap),
        "benchmark_filename_overlap_with_polaris_validation": len(validation_image_overlap),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "reffleur": baseline_metrics,
        "model": model_metrics,
        "delta": {
            "tau_x100": model_metrics["tau_x100"] - baseline_metrics["tau_x100"],
            "mae_reduction": baseline_metrics["mae"] - model_metrics["mae"],
            "rmse_reduction": baseline_metrics["rmse"] - model_metrics["rmse"],
        },
        "strict_all_three_improvement": strict,
        "paper_results": PAPER_RESULTS[dataset],
        "result": str(args.output),
    }
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("=" * 92)
    print(f"FORMAL {dataset.upper()} RESULT")
    print_metrics("Reproduced RefFLEUR", baseline_metrics)
    print_metrics(f"Formal Stage {checkpoint['stage']}", model_metrics)
    print(f"STRICT ALL-THREE IMPROVEMENT: {strict}")
    print(f"Result: {args.output}")
    print(f"Report: {report_path}")


def compare(args: argparse.Namespace) -> None:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    if not reports:
        raise RuntimeError("No reports")
    datasets = {report["dataset"] for report in reports}
    if len(datasets) != 1:
        raise RuntimeError(f"Mixed report datasets: {datasets}")
    dataset = next(iter(datasets))
    print("=" * 86)
    print(f"PUBLISHED-PROTOCOL COMPARISON — {dataset.upper()}")
    print("=" * 86)
    print(f"{'Method':<28}{'Tau':>12}{'MAE':>14}{'RMSE':>14}{'Training':>18}")
    print("-" * 86)
    published_training = {
        "RefCLIP-S": "zero-shot",
        "RefPAC-S": "COCO synthetic",
        "Polos": "Polaris",
        "RefFLEUR": "zero-shot",
    }
    for name, values in PAPER_RESULTS[dataset].items():
        training = published_training[name]
        print(
            f"{name:<28}{values[0]:>12.3f}{values[1]:>14.6f}"
            f"{values[2]:>14.6f}{training:>18}"
        )
    baseline = reports[0]["reffleur"]
    print(
        f"{'RefFLEUR reproduced':<28}{baseline['tau_x100']:>12.3f}"
        f"{baseline['mae']:>14.6f}{baseline['rmse']:>14.6f}{'zero-shot':>18}"
    )
    for report in sorted(reports, key=lambda value: value["stage"]):
        values = report["model"]
        print(
            f"{'TriJudge Stage ' + str(report['stage']):<28}"
            f"{values['tau_x100']:>12.3f}{values['mae']:>14.6f}"
            f"{values['rmse']:>14.6f}{'Polaris':>18}"
        )
    print("-" * 86)
    print("No benchmark label was used for training, early stopping, or checkpoint choice.")
    print("This is a supervised external-data metric; it is not described as zero-shot.")


def selfcheck(args: argparse.Namespace) -> None:
    set_seed(7)
    device = torch.device(args.device)
    batch = {
        "image": torch.randn(6, 64, device=device),
        "candidate": torch.randn(6, 64, device=device),
        "references": torch.randn(6, 5, 64, device=device),
        "reference_mask": torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]] * 2,
            dtype=torch.bool,
            device=device,
        ),
        "baseline": torch.tensor([0.1, 0.25, 0.4, 0.55, 0.7, 0.9], device=device),
        "candidate_length": torch.rand(6, device=device),
    }
    for stage in (1, 2, 3):
        model = ReferenceCredibilityTrainingModel(
            input_dim=64,
            hidden_dim=32,
            dropout=0.0,
            stage=stage,
        ).to(device)
        model.set_trainable_stage(stage)
        output = model(batch)
        if output.score.shape != (6,) or not torch.isfinite(output.score).all():
            raise RuntimeError(f"Stage {stage} forward failed")
        if not torch.allclose(output.score, batch["baseline"], atol=2.0e-6):
            raise RuntimeError(f"Stage {stage} is not baseline-preserving at initialization")
        output.score.mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"Stage {stage} backward failed")
        print(f"Formal Stage {stage} self-check: OK", flush=True)


def add_clip_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--image-batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal Polaris-trained TriJudge-RefFLEUR"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("selfcheck")
    check.add_argument("--device", default="cuda")
    check.set_defaults(function=selfcheck)

    polaris = commands.add_parser("prepare-polaris")
    polaris.add_argument("--polaris-dir", type=Path, required=True)
    polaris.add_argument("--train-baseline", type=Path, required=True)
    polaris.add_argument("--val-baseline", type=Path, required=True)
    add_clip_arguments(polaris)
    polaris.set_defaults(function=prepare_polaris)

    benchmark = commands.add_parser("prepare-benchmark")
    benchmark.add_argument(
        "--dataset",
        choices=("expert", "cf", "composite"),
        required=True,
    )
    benchmark.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    benchmark.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/home/xgd/FLEUR_reproduction/04_datasets"),
    )
    benchmark.add_argument("--baseline-result", type=Path, required=True)
    add_clip_arguments(benchmark)
    benchmark.set_defaults(function=prepare_benchmark)

    training = commands.add_parser("train")
    training.add_argument("--polaris-cache", type=Path, required=True)
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--patience", type=int, default=8)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--eval-batch-size", type=int, default=1024)
    training.add_argument("--pair-batch-size", type=int, default=192)
    training.add_argument("--hidden-dim", type=int, default=192)
    training.add_argument("--dropout", type=float, default=0.10)
    training.add_argument("--lr", type=float, default=3.0e-4)
    training.add_argument("--weight-decay", type=float, default=1.0e-4)
    training.add_argument("--anchor-loss-weight", type=float, default=0.02)
    training.add_argument("--reference-corruption-weight", type=float, default=0.08)
    training.add_argument("--rank-minimum-gap", type=float, default=0.20)
    training.add_argument("--rank-maximum-per-group", type=int, default=48)
    training.add_argument("--rank-temperature", type=float, default=0.10)
    training.add_argument("--rank-loss-weight", type=float, default=0.12)
    training.add_argument("--gradient-clip", type=float, default=1.0)
    training.add_argument("--minimum-objective-gain", type=float, default=1.0e-4)
    training.add_argument("--minimum-tau-gain", type=float, default=0.05)
    training.add_argument("--seed", type=int, default=2026)
    training.add_argument("--device", default="cuda")
    training.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    training.add_argument("--overwrite", action="store_true")
    training.set_defaults(function=train)

    stage_gate = commands.add_parser("gate")
    stage_gate.add_argument("--run-dir", type=Path, required=True)
    stage_gate.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    stage_gate.set_defaults(function=gate)

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--benchmark-cache", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--eval-batch-size", type=int, default=1024)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    evaluation.set_defaults(function=evaluate)

    comparison = commands.add_parser("compare")
    comparison.add_argument("--reports", nargs="+", type=Path, required=True)
    comparison.set_defaults(function=compare)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

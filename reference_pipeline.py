#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from credicap import data_utils as base
from credicap.reference_credibility import ReferenceCredibilityModel
from credicap.evidence_decomposition import IntegratedEvidenceScorer


FORMAT_M2 = "formal-trijudge-m2cded-v7"
LOCKED_M1_SHA256 = (
    "eddd881ce97315bf0274bce01f3bfb222ca5a6cd272bd1e7ae4865d6c78c5e16"
)
LOCKED_M2_SHA256 = (
    "36409390b68d8a38dc64322122e192c3962cccd975ae5e69662a43328088c47e"
)
EXPECTED = {
    "expert": {
        "m1": (53.978, 0.102870, 0.148277),
        "m12": (54.128, 0.102607, 0.148305),
    },
    "cf": {
        "m1": (38.112, 0.176078, 0.260192),
        "m12": (38.197, 0.171827, 0.256939),
    },
    "composite": {
        "m1": (63.420, 0.231223, 0.289928),
        "m12": (63.544, 0.229768, 0.289671),
    },
}
REFPACPP = {
    "expert": (55.765, 0.3320, 0.3591),
    "cf": (37.978, 0.5040, 0.5257),
    "composite": (59.490, 0.3170, 0.3746),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: {actual}; expected {expected}"
        )
    print(f"{label} SHA256: OK ({actual})")


def build_model(
    module1_path: Path,
    module2_path: Path,
    variant: str,
    device: torch.device,
) -> IntegratedEvidenceScorer:
    validate_hash(module1_path, LOCKED_M1_SHA256, "LOCKED M1")
    validate_hash(module2_path, LOCKED_M2_SHA256, "LOCKED CDED-M2")
    module1_payload = base.torch_load(module1_path)
    module2_payload = base.torch_load(module2_path)
    required_m1 = {"model", "embedding_dim", "hidden_dim", "dropout"}
    missing = required_m1 - set(module1_payload)
    if missing:
        raise RuntimeError(f"M1 checkpoint missing: {sorted(missing)}")
    if module2_payload.get("format") != FORMAT_M2:
        raise RuntimeError(
            f"Wrong M2 format: {module2_payload.get('format')}"
        )
    if module2_payload.get("module1_sha256") != LOCKED_M1_SHA256:
        raise RuntimeError("M2 was not trained on the locked M1 checkpoint")
    if module2_payload.get("active") is not True:
        raise RuntimeError("Locked M2 checkpoint is not active")

    module1 = ReferenceCredibilityModel(
        input_dim=int(module1_payload["embedding_dim"]),
        hidden_dim=int(module1_payload["hidden_dim"]),
        dropout=float(module1_payload["dropout"]),
    ).to(device)
    retained_state = {
        name: value
        for name, value in module1_payload["model"].items()
        if name.startswith("reference_trust.")
        or name.startswith("evidence_router.")
    }
    module1.load_state_dict(retained_state, strict=True)
    model = IntegratedEvidenceScorer(
        module1=module1,
        input_dim=int(module1_payload["embedding_dim"]),
        hidden_dim=int(module1_payload["hidden_dim"]),
        dropout=float(module1_payload["dropout"]),
        variant=variant,
    ).to(device)
    model.module2.load_state_dict(module2_payload["module_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.inference_mode()
def predict(
    model: IntegratedEvidenceScorer,
    split: dict,
    batch_size: int,
    device: torch.device,
    description: str,
) -> dict[str, np.ndarray]:
    fields = (
        "score",
        "module1_score",
        "stage1_score",
        "hidden_change_rms",
        "structural_strength",
        "consensus_support",
        "dissent_support",
        "consensus_dissent_gap",
        "support_dispersion",
        "reference_disagreement",
    )
    output = {field: [] for field in fields}
    loader = DataLoader(
        TensorDataset(torch.arange(len(split["sample_ids"]))),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    for (indices,) in tqdm(
        loader,
        desc=description,
        dynamic_ncols=True,
    ):
        prediction = model(base.make_batch(split, indices, device))
        for field in fields:
            output[field].append(
                getattr(prediction, field).detach().float().cpu()
            )
    return {
        field: torch.cat(values).numpy()
        for field, values in output.items()
    }


def check_expected(dataset: str, variant: str, values: dict) -> None:
    target = EXPECTED[dataset][variant]
    actual = (values["tau_x100"], values["mae"], values["rmse"])
    tolerance = (0.06, 0.0006, 0.0006)
    failures = [
        f"{name}={value:.6f}, expected≈{wanted:.6f}"
        for name, value, wanted, allowed in zip(
            ("tau", "mae", "rmse"), actual, target, tolerance
        )
        if abs(value - wanted) > allowed
    ]
    if failures:
        raise RuntimeError(
            f"Locked {dataset}/{variant} result changed: " + "; ".join(failures)
        )


def selfcheck(args) -> None:
    device = torch.device(args.device)
    for variant in ("m1", "m12"):
        model = build_model(
            args.module1_checkpoint,
            args.module2_checkpoint,
            variant,
            device,
        )
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError(f"{variant}: a locked parameter is trainable")
        print(f"{variant}: locked model load PASS")
    for dataset, path in (
        ("expert", args.expert_cache),
        ("cf", args.cf_cache),
        ("composite", args.composite_cache),
    ):
        cache = base.load_cache(path, expected_kind="benchmark")
        if cache["dataset"] != dataset:
            raise RuntimeError(
                f"Wrong dataset in {path}: {cache['dataset']} != {dataset}"
            )
        baseline = base.metric_values(
            cache["split"]["baseline"].numpy(), cache["split"]
        )
        base.verify_locked_baseline(dataset, baseline)
        print(
            f"{dataset}: cache and reproduced RefFLEUR baseline PASS "
            f"(n={len(cache['split']['sample_ids'])})"
        )
    print("FINAL M1+CDED-M2 SELF-CHECK: PASS")


def evaluate(args) -> None:
    device = torch.device(args.device)
    cache = base.load_cache(args.cache, expected_kind="benchmark")
    dataset = cache["dataset"]
    split = cache["split"]
    model = build_model(
        args.module1_checkpoint,
        args.module2_checkpoint,
        args.variant,
        device,
    )
    prediction = predict(
        model,
        split,
        args.batch_size,
        device,
        f"Locked {args.variant.upper()} {dataset}",
    )
    values = base.metric_values(prediction["score"], split)
    baseline = base.metric_values(split["baseline"].numpy(), split)
    base.verify_locked_baseline(dataset, baseline)
    check_expected(dataset, args.variant, values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(split["records"]):
            row = dict(record)
            row["mode"] = f"locked_trijudge_{args.variant}_final"
            row["score"] = float(prediction["score"][index])
            row["audit"] = {
                "module1_score": float(prediction["module1_score"][index]),
                "stage1_score": float(prediction["stage1_score"][index]),
                "hidden_change_rms": float(
                    prediction["hidden_change_rms"][index]
                ),
                "structural_strength": float(
                    prediction["structural_strength"][index]
                ),
                "consensus_support": float(
                    prediction["consensus_support"][index]
                ),
                "dissent_support": float(
                    prediction["dissent_support"][index]
                ),
                "consensus_dissent_gap": float(
                    prediction["consensus_dissent_gap"][index]
                ),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "method": "TriJudge-M1+CDED-M2 (final locked method)",
        "variant": args.variant,
        "dataset": dataset,
        "module1_sha256": LOCKED_M1_SHA256,
        "module2_sha256": LOCKED_M2_SHA256,
        "training": "official Polaris train",
        "checkpoint_selection": "official Polaris validation",
        "benchmark_labels_used_for_training_or_selection": False,
        "five_fold_oof": False,
        "seed_ensemble": False,
        "reffleur_reproduced": baseline,
        "metrics": values,
        "diagnostics": {
            "mean_hidden_change_rms": float(
                prediction["hidden_change_rms"].mean()
            ),
            "mean_structural_strength": float(
                prediction["structural_strength"].mean()
            ),
            "mean_consensus_support": float(
                prediction["consensus_support"].mean()
            ),
            "mean_dissent_support": float(
                prediction["dissent_support"].mean()
            ),
        },
    }
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("=" * 100)
    print(f"LOCKED FINAL RESULT — {dataset.upper()} / {args.variant.upper()}")
    base.print_metrics("Reproduced RefFLEUR", baseline)
    base.print_metrics(args.variant.upper(), values)
    print("Result:", args.output)
    print("Report:", report_path)
    print("=" * 100)


def paper_rows(dataset: str):
    rows = []
    for name in ("RefCLIP-S", "RefPAC-S"):
        rows.append((name, base.PAPER_RESULTS[dataset][name]))
    rows.append(("RefPAC-S++", REFPACPP[dataset]))
    for name in ("Polos", "RefFLEUR"):
        rows.append((name, base.PAPER_RESULTS[dataset][name]))
    return rows


def summarize(args) -> None:
    datasets = ("expert", "cf", "composite")
    reports = {
        dataset: {
            variant: json.loads(
                (
                    args.root
                    / dataset
                    / f"trijudge_{variant}_{dataset}.metrics.json"
                ).read_text(encoding="utf-8")
            )
            for variant in ("m1", "m12")
        }
        for dataset in datasets
    }
    lines = [
        "| Method | Expert Tau-c | Expert MAE | Expert RMSE | CF Tau-b | CF MAE | CF RMSE | Composite Tau-c | Composite MAE | Composite RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "RefCLIP-S",
        "RefPAC-S",
        "RefPAC-S++",
        "Polos",
        "RefFLEUR",
    ):
        values = []
        for dataset in datasets:
            values.extend(dict(paper_rows(dataset))[name])
        rendered = [
            f"{value:.3f}" if index % 3 == 0 else f"{value:.6f}"
            for index, value in enumerate(values)
        ]
        lines.append(f"| {name} | " + " | ".join(rendered) + " |")

    for variant, name in (
        ("m1", "TriJudge-M1"),
        ("m12", "TriJudge-M1+CDED-M2 (Final)"),
    ):
        values = []
        for dataset in datasets:
            metric = reports[dataset][variant]["metrics"]
            values.extend((metric["tau_x100"], metric["mae"], metric["rmse"]))
        rendered = [
            f"{value:.3f}" if index % 3 == 0 else f"{value:.6f}"
            for index, value in enumerate(values)
        ]
        lines.append(f"| {name} | " + " | ".join(rendered) + " |")

    lines.extend(
        [
            "",
            "Final method: TriJudge-M1+CDED-M2.",
            f"Locked M1 SHA256: `{LOCKED_M1_SHA256}`",
            f"Locked M2 SHA256: `{LOCKED_M2_SHA256}`",
            "M3: abandoned and excluded from the final method.",
        ]
    )
    text = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)

    command = commands.add_parser("selfcheck")
    command.add_argument("--module1-checkpoint", type=Path, required=True)
    command.add_argument("--module2-checkpoint", type=Path, required=True)
    command.add_argument("--expert-cache", type=Path, required=True)
    command.add_argument("--cf-cache", type=Path, required=True)
    command.add_argument("--composite-cache", type=Path, required=True)
    command.add_argument("--device", default="cuda")
    command.set_defaults(function=selfcheck)

    command = commands.add_parser("evaluate")
    command.add_argument("--variant", choices=("m1", "m12"), required=True)
    command.add_argument("--module1-checkpoint", type=Path, required=True)
    command.add_argument("--module2-checkpoint", type=Path, required=True)
    command.add_argument("--cache", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--batch-size", type=int, default=1024)
    command.add_argument("--device", default="cuda")
    command.set_defaults(function=evaluate)

    command = commands.add_parser("summarize")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=summarize)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)

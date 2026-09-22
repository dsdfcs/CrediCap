#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PUBLISHED = (
    ("RefCLIP-S", (53.023, 0.384600, 0.411000), (36.404, 0.546200, 0.566200), (56.275, 0.328200, 0.395300)),
    ("RefPAC-S", (55.863, 0.439800, 0.464300), (37.624, 0.601500, 0.620600), (57.960, 0.341900, 0.420700)),
    ("RefPAC-S++", (55.765, 0.332000, 0.359100), (37.978, 0.504000, 0.525700), (59.490, 0.317000, 0.374600)),
    ("Polos", (56.454, 0.103800, 0.133400), (37.797, 0.210100, 0.249700), (58.379, 0.259600, 0.312000)),
    ("RefFLEUR", (51.940, 0.118100, 0.173900), (38.803, 0.104100, 0.167400), (64.223, 0.245300, 0.323900)),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def triple(metric: dict) -> tuple[float, float, float]:
    return (
        float(metric["tau_x100"]),
        float(metric["mae"]),
        float(metric["rmse"]),
    )


def row(label: str, values: tuple[tuple[float, float, float], ...]) -> str:
    cells = [label]
    for tau, mae, rmse in values:
        cells.extend((f"{tau:.6f}", f"{mae:.6f}", f"{rmse:.6f}"))
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/trijudge_feedback_signed_error_v9"),
    )
    parser.add_argument("--if-complete", action="store_true")
    args = parser.parse_args()

    paths = {
        split: args.root / split / f"{split}_feedback_signed_error.metrics.json"
        for split in ("expert", "cf", "composite")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        message = "Cannot summarize before all reports exist: " + ", ".join(missing)
        if args.if_complete:
            print(message)
            return
        raise FileNotFoundError(message)

    reports = {split: read_json(path) for split, path in paths.items()}
    for split, report in reports.items():
        if report.get("dataset") != split:
            raise RuntimeError(f"Report dataset mismatch: {paths[split]}")
        if report.get("benchmark_labels_used_for_training_or_selection") is not False:
            raise RuntimeError(f"Benchmark selection leakage flag in {split}")
    checkpoint_hashes = {
        report.get("checkpoint_sha256") for report in reports.values()
    }
    if len(checkpoint_hashes) != 1:
        raise RuntimeError("The three reports did not use the same locked v9 checkpoint")

    order = ("expert", "cf", "composite")
    m12 = tuple(
        triple(reports[split]["metrics"]["locked_m12_no_feedback"])
        for split in order
    )
    v9 = tuple(
        triple(reports[split]["metrics"]["all_feedback_second_stage"])
        for split in order
    )
    lines = [
        "# TriJudge Feedback-SignedError v9：三个数据集总表",
        "",
        "| Method | Expert Tau-c ↑ | Expert MAE ↓ | Expert RMSE ↓ | CF Tau-b ↑ | CF MAE ↓ | CF RMSE ↓ | Composite Tau-c ↑ | Composite MAE ↓ | Composite RMSE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, expert, cf, composite in PUBLISHED:
        lines.append(row(label, (expert, cf, composite)))
    lines.extend(
        (
            row("Locked M1+CDED-M2 / no feedback", m12),
            row("TriJudge-v9 / all-feedback signed second stage", v9),
            "",
            "## v9 相对锁定 M1+CDED-M2",
            "",
            "| Dataset | Tau gain ↑ | MAE reduction ↑ | RMSE reduction ↑ | 三项同时提升 |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for split in order:
        effect = reports[split]["effects"]["feedback_vs_locked_m12"]
        lines.append(
            f"| {split} | {effect['tau_gain']:+.6f} | "
            f"{effect['mae_reduction']:+.6f} | "
            f"{effect['rmse_reduction']:+.6f} | "
            f"{effect['all_three_better']} |"
        )
    lines.extend(
        (
            "",
            f"- 三个测试均使用同一个 v9 checkpoint：`{next(iter(checkpoint_hashes))}`",
            "- v9 只在 Polaris train 训练、Polaris validation 选型；三个基准标签只用于最终指标。",
            "- 每个数据集的每一行都处理全部七个结构化反馈字段。",
        )
    )
    output = args.root / "all_datasets_feedback_signed_error_v9_summary.md"
    temporary = output.with_suffix(output.suffix + ".writing")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output)
    print("\n".join(lines))
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()

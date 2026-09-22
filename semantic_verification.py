#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from credicap import verification_utils as common


CACHE_FORMAT = "formal-trijudge-reffleur-polaris-v2"


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_split(path: Path, split_name: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    cache = torch_load(path)
    if cache.get("format") != CACHE_FORMAT:
        raise RuntimeError(f"Wrong source cache format: {cache.get('format')}")
    if cache.get("kind") == "polaris":
        if split_name not in {"train", "val"}:
            raise RuntimeError("Polaris cache accepts only train or val")
        split = cache[split_name]
    elif cache.get("kind") == "benchmark":
        if split_name != cache.get("dataset"):
            raise RuntimeError(
                f"Benchmark cache contains {cache.get('dataset')}, not {split_name}"
            )
        split = cache["split"]
    else:
        raise RuntimeError(f"Unknown cache kind: {cache.get('kind')}")
    if len(split["sample_ids"]) != len(split["records"]):
        raise RuntimeError("Source cache row mismatch")
    return split


class FrozenQwenVL:
    """Qwen2.5-VL loader that does not depend on qwen-vl-utils."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Qwen feedback generation")
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Activate the Qwen environment with Qwen2_5_VL support"
            ) from exc
        print(f"Loading frozen Qwen2.5-VL: {args.model_path}", flush=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(args.model_path),
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            str(args.model_path),
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            local_files_only=True,
        )
        self.print_memory("After Qwen load")

    @staticmethod
    def print_memory(label: str) -> None:
        print(
            f"{label}: GPU allocated/reserved/peak="
            f"{torch.cuda.memory_allocated() / 1024**3:.2f}/"
            f"{torch.cuda.memory_reserved() / 1024**3:.2f}/"
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
            flush=True,
        )

    def generate(self, image_path: Path, prompt: str, max_new_tokens: int) -> str:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = self.processor(
            text=[rendered],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def feedback_input_key(record: dict) -> tuple:
    """Identity of the information supplied to the frozen feedback model."""
    return (
        str(record.get("dataset", "")),
        str(record.get("subset", "")),
        str(record.get("image", "")),
        common.clean(record.get("candidate", "")),
        tuple(common.clean(value) for value in record.get("references", ())),
    )


def image_path_for(
    record: dict,
    image_root: Path | None,
    datasets_root: Path | None,
) -> Path:
    label = record.get("image")
    if not label:
        raise RuntimeError(f"Record has no image label: {record.get('sample_id')}")
    if datasets_root is not None:
        dataset = str(record.get("dataset", ""))
        subset = str(record.get("subset", ""))
        if dataset in {"expert", "cf"} or subset == "flickr8k":
            directory = datasets_root / "flickr8k"
        elif subset == "flickr30k":
            directory = datasets_root / "flickr30k"
        elif subset == "coco":
            directory = datasets_root / "coco2014"
        else:
            raise RuntimeError(
                f"Cannot resolve image subset for {dataset}/{subset}: "
                f"{record.get('sample_id')}"
            )
        return directory / str(label)
    if image_root is None:
        raise RuntimeError("Either --image-root or --datasets-root is required")
    return image_root / str(label)


def audit_one(
    engine: FrozenQwenVL,
    image_path: Path,
    record: dict,
    max_new_tokens: int,
    maximum_attempts: int,
    require_all_strict: bool,
) -> dict:
    raw = ""
    errors: list[str] = []
    for attempt in range(maximum_attempts):
        raw = engine.generate(
            image_path,
            common.feedback_prompt(
                record["references"],
                record["candidate"],
                retry=attempt > 0,
            ),
            max_new_tokens,
        )
        try:
            codes, values = common.parse_feedback(raw)
            return {
                "parse_mode": "strict",
                "codes": codes,
                "values": values,
                "raw_output": raw,
                "attempt": attempt + 1,
            }
        except ValueError as exc:
            errors.append(str(exc))
    if require_all_strict:
        raise RuntimeError(
            "Qwen did not satisfy the seven-field feedback contract after "
            f"{maximum_attempts} attempts for {record.get('sample_id')}: "
            f"{raw!r}"
        )
    return {
        "parse_mode": "safe_fallback",
        "codes": ["P"] * len(common.FIELDS),
        "values": common.fallback_vector(),
        "raw_output": raw,
        "attempt": maximum_attempts,
        "errors": errors,
    }


def generate(args: argparse.Namespace) -> None:
    if (args.image_root is None) == (args.datasets_root is None):
        raise RuntimeError(
            "Supply exactly one of --image-root and --datasets-root"
        )
    split = load_split(args.source_cache, args.split)
    records = split["records"]
    sample_ids = split["sample_ids"]
    rows = common.read_jsonl_resume(args.output)
    common.validate_rows(rows, sample_ids)
    for index, row in enumerate(rows):
        if row.get("split") != args.split:
            raise RuntimeError(
                f"Existing feedback split mismatch at row {index}: "
                f"{row.get('split')} != {args.split}"
            )
    if args.require_all_strict:
        invalid = sum(row["parse_mode"] != "strict" for row in rows)
        if invalid:
            raise RuntimeError(
                f"Existing feedback contains {invalid} fallback rows. "
                "v9 requires strict seven-field feedback for every row; "
                "use a new output path."
            )
    unique_inputs = len({feedback_input_key(record) for record in records})
    print("=" * 112)
    print(f"STRUCTURED SCORE-FREE FEEDBACK — {args.split.upper()}")
    print("=" * 112)
    print(f"Rows             : {len(records)}")
    print(f"Unique VLM inputs: {unique_inputs}")
    print(f"Exact duplicates : {len(records) - unique_inputs}")
    print(f"Resume           : {len(rows)}/{len(records)}")
    print("Output contract  : E/A/T/R/C/U/Q, each N/P/S")
    print("Caption score    : NEVER requested")
    print("Score direction  : NEVER requested")
    print("Human gold       : NEVER placed in the prompt")
    print("qwen-vl-utils    : NOT REQUIRED")
    print("=" * 112, flush=True)
    if len(rows) == len(records):
        print("STRUCTURED FEEDBACK ALREADY COMPLETE", flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fallback_count = sum(row["parse_mode"] != "strict" for row in rows)
    seen: dict[tuple, tuple[int, dict]] = {}
    for index, row in enumerate(rows):
        key = feedback_input_key(records[index])
        seen.setdefault(key, (index, row))
    engine: FrozenQwenVL | None = None
    generated_count = 0
    reused_count = 0
    first_generated_printed = False
    with args.output.open("a", encoding="utf-8") as handle:
        progress = tqdm(
            range(len(rows), len(records)),
            initial=len(rows),
            total=len(records),
            desc=f"Qwen structured feedback {args.split}",
            dynamic_ncols=True,
        )
        for index in progress:
            record = records[index]
            key = feedback_input_key(record)
            if key in seen:
                source_index, source = seen[key]
                result = {
                    "parse_mode": source["parse_mode"],
                    "codes": list(source["codes"]),
                    "values": list(source["values"]),
                    "raw_output": source.get("raw_output", ""),
                    "attempt": 0,
                    "generation_mode": "exact_input_reuse",
                    "reuse_source_index": source_index,
                }
                reused_count += 1
            else:
                if engine is None:
                    engine = FrozenQwenVL(args)
                result = audit_one(
                    engine,
                    image_path_for(
                        record,
                        args.image_root,
                        args.datasets_root,
                    ),
                    record,
                    args.max_new_tokens,
                    args.maximum_attempts,
                    args.require_all_strict,
                )
                result["generation_mode"] = "qwen2.5-vl"
                seen[key] = (index, result)
                generated_count += 1
            fallback_count += result["parse_mode"] != "strict"
            row = {
                "format": common.FORMAT,
                "split": args.split,
                "index": index,
                "sample_id": sample_ids[index],
                **result,
            }
            common.write_row(handle, row)
            rows.append(row)
            progress.set_postfix(
                generated=generated_count,
                reused=reused_count,
                fallback=fallback_count,
            )
            if result["generation_mode"] == "qwen2.5-vl" and not first_generated_printed:
                first_generated_printed = True
                print(
                    "FIRST STRUCTURED FEEDBACK\n"
                    f"  raw   : {result['raw_output']!r}\n"
                    f"  codes : {result['codes']}\n"
                    f"  mode  : {result['parse_mode']}",
                    flush=True,
                )
                assert engine is not None
                engine.print_memory("After first feedback")
    common.validate_rows(rows, sample_ids)
    if len(rows) != len(records):
        raise RuntimeError("Feedback generation ended with an incomplete file")
    if args.require_all_strict and fallback_count:
        raise RuntimeError(
            f"Strict v9 feedback contract violated: {fallback_count} fallbacks"
        )
    print(
        f"STRUCTURED FEEDBACK COMPLETE: strict={len(rows)-fallback_count} "
        f"fallback={fallback_count}\n"
        f"New Qwen calls={generated_count} exact-input reuses={reused_count}",
        flush=True,
    )


def selfcheck(_: argparse.Namespace) -> None:
    common.selfcheck()
    left = {
        "dataset": "cf",
        "subset": "flickr8k",
        "image": "sample.jpg",
        "candidate": "A dog runs.",
        "references": ["A dog is outside."],
        "human_ratings": [0.2],
    }
    right = {**left, "human_ratings": [0.8]}
    if feedback_input_key(left) != feedback_input_key(right):
        raise RuntimeError("Exact-input feedback reuse key is unstable")
    expected = Path("/datasets/flickr8k/sample.jpg")
    if image_path_for(left, None, Path("/datasets")) != expected:
        raise RuntimeError("CF image resolution self-check failed")
    composite = {**left, "dataset": "composite", "subset": "coco"}
    expected = Path("/datasets/coco2014/sample.jpg")
    if image_path_for(composite, None, Path("/datasets")) != expected:
        raise RuntimeError("Composite image resolution self-check failed")
    print("STRUCTURED FEEDBACK SELF-CHECK: PASS")
    print("CF/Composite mixed image routing: PASS")
    print("Exact-input duplicate reuse      : PASS")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(required=True)
    check = sub.add_parser("selfcheck")
    check.set_defaults(function=selfcheck)
    run = sub.add_parser("generate")
    run.add_argument("--source-cache", type=Path, required=True)
    run.add_argument(
        "--split",
        choices=("train", "val", "expert", "cf", "composite"),
        required=True,
    )
    run.add_argument("--image-root", type=Path)
    run.add_argument("--datasets-root", type=Path)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-new-tokens", type=int, default=40)
    run.add_argument("--maximum-attempts", type=int, default=4)
    run.add_argument("--require-all-strict", action="store_true")
    run.add_argument("--min-pixels", type=int, default=200704)
    run.add_argument("--max-pixels", type=int, default=1003520)
    run.set_defaults(function=generate)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)

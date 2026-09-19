#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
FORMAT = "formal-polaris-reffleur-v1"
NUMBER = re.compile(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")


@dataclass(frozen=True)
class PolarisPromptSample:
    sample_id: str
    image_path: Path
    candidate: str
    references: tuple[str, ...]


def clean_text(value: object) -> str:
    return " ".join(str(value).split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_references(value: str) -> tuple[str, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse Polaris refs: {value[:160]!r}") from exc
    if not isinstance(parsed, (list, tuple)):
        raise RuntimeError("Polaris refs must be a list")
    references = tuple(clean_text(item) for item in parsed if clean_text(item))
    if not references:
        raise RuntimeError("Polaris row has no reference caption")
    return references


def load_polaris_csv(
    csv_path: Path,
    images_dir: Path,
    split: str,
) -> list[PolarisPromptSample]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    samples: list[PolarisPromptSample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"mt", "refs", "imgid"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"{csv_path} missing columns: {sorted(missing)}; "
                "expected legacy Polaris CSV columns mt, refs, score, imgid"
            )
        for index, row in enumerate(reader):
            image_id = clean_text(row["imgid"])
            image_path = images_dir / image_id
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing Polaris image at row {index}: {image_path}"
                )
            samples.append(
                PolarisPromptSample(
                    sample_id=f"polaris:{split}:{index:06d}",
                    image_path=image_path,
                    candidate=clean_text(row["mt"]),
                    references=parse_references(row["refs"]),
                )
            )
    if not samples:
        raise RuntimeError(f"Empty Polaris split: {csv_path}")
    return samples


def exact_reffleur_instruction(sample: PolarisPromptSample) -> str:
    references = "\n".join(sample.references)
    return (
        "Your task is to evaluate and rate the candidate caption on a scale of "
        "0.0 to 1.0 based on the given Grading Criteria. (Print Real Number "
        "Score ONLY)\n\n"
        "Grading Criteria:\n\n"
        "0.0: The caption does not describe the image at all.\n"
        "1.0: The caption accurately and clearly describes the image.\n\n"
        f"Reference Captions: {references}\n\n"
        f"Candidate Caption: {sample.candidate}\n\n"
        "Score(Choose a rating from 0.0 to 1.0):"
    )


class ExactRefFLEURScorer:
    def __init__(
        self,
        project_root: Path,
        model_path: Path,
        image_aspect_ratio: str,
        max_new_tokens: int,
    ) -> None:
        llava_root = project_root / "LLaVA"
        if str(llava_root) not in sys.path:
            sys.path.insert(0, str(llava_root))

        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from llava.model.builder import load_pretrained_model

        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.image_args = SimpleNamespace(image_aspect_ratio=image_aspect_ratio)
        self.max_new_tokens = max_new_tokens

        print(f"Loading exact RefFLEUR LLaVA backbone: {model_path}", flush=True)
        model_name = get_model_name_from_path(str(model_path))
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            model_path=str(model_path),
            model_base=None,
            model_name=model_name,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = self.model.device
        self.dtype = getattr(self.model, "dtype", torch.float16)
        self.rate_to_token = {
            digit: self.tokenizer.encode(str(digit))[-1]
            for digit in range(10)
        }
        self.first = True
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _smoothed_score(
        self,
        generated_ids: torch.Tensor,
        step_logits: tuple[torch.Tensor, ...],
        decoded: str,
    ) -> tuple[float, float]:
        match = NUMBER.search(decoded)
        if match is None:
            raise RuntimeError(f"No valid numeric score in output: {decoded!r}")
        number_text = match.group(0)
        raw_score = float(number_text)
        if not 0.0 <= raw_score <= 1.0:
            raise RuntimeError(f"Out-of-range RefFLEUR score: {raw_score}")

        def positions_for_digit(digit: int) -> list[int]:
            token = self.rate_to_token[digit]
            return (
                (generated_ids == token)
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )

        if raw_score < 1.0:
            decimals = number_text.split(".", 1)[1] if "." in number_text else "0"
            first_digit = int(decimals[0])
            positions = positions_for_digit(first_digit)
            if not positions:
                raise RuntimeError(
                    f"Cannot find first decimal digit token in {decoded!r}"
                )
            # This is the same selection rule used by the released RefFLEUR
            # script: for 0.x, the second zero belongs to the decimal place.
            first_position = positions[1] if first_digit == 0 and len(positions) > 1 else positions[0]
            if first_position >= len(step_logits):
                raise RuntimeError("Generated-token/logit alignment failure")
            probabilities = F.softmax(step_logits[first_position].float(), dim=-1)[0]
            score = sum(
                float(probabilities[token]) * digit * 0.1
                for digit, token in self.rate_to_token.items()
            )

            if len(decimals) >= 2:
                second_digit = int(decimals[1])
                second_positions = positions_for_digit(second_digit)
                if not second_positions:
                    raise RuntimeError(
                        f"Cannot find second decimal digit token in {decoded!r}"
                    )
                if second_digit == first_digit and len(second_positions) > 1:
                    second_position = second_positions[1]
                elif second_digit == 0 and len(second_positions) > 1:
                    second_position = second_positions[-1]
                else:
                    second_position = second_positions[-1]
                if second_position >= len(step_logits):
                    raise RuntimeError("Generated-token/logit alignment failure")
                probabilities2 = F.softmax(
                    step_logits[second_position].float(),
                    dim=-1,
                )[0]
                score += sum(
                    float(probabilities2[token]) * digit * 0.01
                    for digit, token in self.rate_to_token.items()
                )
        else:
            positions = positions_for_digit(1)
            if not positions:
                raise RuntimeError(f"Cannot find 1 token in {decoded!r}")
            position = positions[0]
            probabilities = F.softmax(step_logits[position].float(), dim=-1)[0]
            score = (
                0.9 * float(probabilities[self.rate_to_token[0]])
                + float(probabilities[self.rate_to_token[1]])
            )

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(f"Invalid smoothed score {score} from {decoded!r}")
        return raw_score, score

    @torch.inference_mode()
    def score(self, sample: PolarisPromptSample) -> tuple[float, float, str]:
        instruction = exact_reffleur_instruction(sample)
        conv = self.conv_templates["llava_v1"].copy()
        if self.model.config.mm_use_im_start_end:
            user_text = (
                self.DEFAULT_IM_START_TOKEN
                + self.DEFAULT_IMAGE_TOKEN
                + self.DEFAULT_IM_END_TOKEN
                + "\n"
                + instruction
            )
        else:
            user_text = self.DEFAULT_IMAGE_TOKEN + "\n" + instruction
        conv.append_message(conv.roles[0], user_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        if input_ids.shape[1] > self.context_len:
            raise RuntimeError(
                f"Prompt too long for {sample.sample_id}: "
                f"{input_ids.shape[1]} > {self.context_len}"
            )

        with Image.open(sample.image_path) as opened:
            image = opened.convert("RGB")
            image_tensor = self.process_images(
                [image],
                self.image_processor,
                self.image_args,
            )
        if isinstance(image_tensor, list):
            image_tensor = [value.to(self.device, dtype=self.dtype) for value in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        stop = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        stopping = self.KeywordsStoppingCriteria([stop], self.tokenizer, input_ids)
        output = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.2,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping],
            output_scores=True,
            return_dict_in_generate=True,
        )
        generated_ids = output.sequences[0, input_ids.shape[1] :]
        decoded = self.tokenizer.decode(generated_ids).strip()
        raw_score, smoothed_score = self._smoothed_score(
            generated_ids,
            output.scores,
            decoded,
        )

        if self.first:
            allocated = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            reserved = torch.cuda.memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            print(
                "FIRST FORMAL POLARIS REFFLEUR OUTPUT\n"
                f"  sample       : {sample.sample_id}\n"
                f"  raw text     : {decoded!r}\n"
                f"  raw score    : {raw_score:.6f}\n"
                f"  smooth score : {smoothed_score:.6f}\n"
                f"  GPU GiB allocated/reserved/peak: "
                f"{allocated:.2f}/{reserved:.2f}/{peak:.2f}",
                flush=True,
            )
            self.first = False
        return raw_score, smoothed_score, decoded

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_completed(path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in completed:
                raise RuntimeError(f"Duplicate {sample_id} in {path}:{line_number}")
            score = float(row["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RuntimeError(f"Invalid saved score in {path}:{line_number}")
            completed[sample_id] = row
    return completed


def score_split(
    scorer: ExactRefFLEURScorer,
    split: str,
    csv_path: Path,
    images_dir: Path,
    output: Path,
    model_path: Path,
    limit: int | None,
) -> None:
    samples = load_polaris_csv(csv_path, images_dir, split)
    selected = samples if limit is None else samples[:limit]
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    expected_meta = {
        "format": FORMAT,
        "split": split,
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "images_dir": str(images_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "prompt": "exact released RefFLEUR prompt",
        "scorer_sha256": sha256_file(Path(__file__)),
        "expected_rows": len(samples),
    }
    if output.exists() and not meta_path.is_file():
        raise RuntimeError(f"Refusing unverified resume file without metadata: {output}")
    if meta_path.is_file():
        saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved_meta != expected_meta:
            raise RuntimeError(
                f"Resume metadata mismatch for {output}; move the old file first"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(expected_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    completed = load_completed(output)
    expected_ids = {sample.sample_id for sample in selected}
    unexpected = set(completed) - {sample.sample_id for sample in samples}
    if unexpected:
        raise RuntimeError(f"Unexpected IDs in {output}: {sorted(unexpected)[:3]}")

    remaining = [sample for sample in selected if sample.sample_id not in completed]
    print(
        f"Polaris {split}: total={len(samples)} selected={len(selected)} "
        f"resume={len(selected) - len(remaining)} remaining={len(remaining)}",
        flush=True,
    )
    if remaining:
        with output.open("a", encoding="utf-8", buffering=1) as handle:
            for sample in tqdm(
                remaining,
                desc=f"Formal RefFLEUR Polaris {split}",
                dynamic_ncols=True,
            ):
                raw_score, score, raw_text = scorer.score(sample)
                row = {
                    "sample_id": sample.sample_id,
                    "image": sample.image_path.name,
                    "candidate": sample.candidate,
                    "raw_score": raw_score,
                    "score": score,
                    "raw_text": raw_text,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    final = load_completed(output)
    completed_selected = expected_ids & set(final)
    if len(completed_selected) != len(selected):
        raise RuntimeError(
            f"Incomplete {split}: {len(completed_selected)} != {len(selected)}"
        )
    print(f"Saved {split} RefFLEUR scores: {output}", flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Generate exact released RefFLEUR scores for Polaris train/val"
    )
    value.add_argument("--polaris-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/home/xgd/FLEUR_reproduction/03_models/liuhaotian_llava-v1.5-13b"
        ),
    )
    value.add_argument("--project-root", type=Path, default=ROOT)
    value.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    value.add_argument("--image-aspect-ratio", default="pad")
    value.add_argument("--max-new-tokens", type=int, default=32)
    value.add_argument("--limit", type=int)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    csv_paths = {
        "train": args.polaris_dir / "polaris_train.csv",
        "val": args.polaris_dir / "polaris_val.csv",
    }
    images_dir = args.polaris_dir / "images"
    for split in args.splits:
        if not csv_paths[split].is_file():
            raise FileNotFoundError(csv_paths[split])
    scorer = ExactRefFLEURScorer(
        project_root=args.project_root,
        model_path=args.model_path,
        image_aspect_ratio=args.image_aspect_ratio,
        max_new_tokens=args.max_new_tokens,
    )
    try:
        for split in args.splits:
            score_split(
                scorer=scorer,
                split=split,
                csv_path=csv_paths[split],
                images_dir=images_dir,
                output=args.output_dir / f"polaris_{split}_reffleur.jsonl",
                model_path=args.model_path,
                limit=args.limit,
            )
    finally:
        scorer.close()


if __name__ == "__main__":
    main()

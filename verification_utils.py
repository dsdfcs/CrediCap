#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np


FORMAT = "trijudge-structured-feedback-v6"

# A compact, score-free audit.  Positive and negative evidence are kept in
# separate fields instead of asking the VLM for another absolute score.
FIELDS = (
    "ENTITY_SUPPORT",
    "ACTION_SUPPORT",
    "ATTRIBUTE_SUPPORT",
    "REFERENCE_SUPPORT",
    "REFERENCE_CONFLICT",
    "UNSUPPORTED_DETAIL",
    "UNCERTAINTY",
)
SHORT_KEYS = ("E", "A", "T", "R", "C", "U", "Q")
LEVELS = {"N": 0.0, "P": 0.5, "S": 1.0}

OUTPUT_PATTERN = re.compile(
    r"\bE\s*=\s*([NPS])\s+"
    r"A\s*=\s*([NPS])\s+"
    r"T\s*=\s*([NPS])\s+"
    r"R\s*=\s*([NPS])\s+"
    r"C\s*=\s*([NPS])\s+"
    r"U\s*=\s*([NPS])\s+"
    r"Q\s*=\s*([NPS])\b",
    flags=re.IGNORECASE,
)


def clean(value: object) -> str:
    return " ".join(str(value).split())


def format_references(references: Sequence[str]) -> str:
    return "\n".join(
        f"{index}. {clean(reference)}"
        for index, reference in enumerate(references, start=1)
    )


def feedback_prompt(
    references: Sequence[str],
    candidate: str,
    retry: bool = False,
) -> str:
    retry_text = ""
    if retry:
        retry_text = (
            "\nFORMAT RETRY: print exactly one line and no explanation.\n"
        )
    return f"""Audit one candidate image caption using the image and the references.
References are evidence, not guaranteed truth.  A detail is wrong only when
the image or several reliable references contradict it.  Missing details are
not errors.  Do not give a caption score and do not recommend raising or
lowering a score.

For every field choose one letter:
N = no evidence, P = partial/uncertain evidence, S = strong evidence.

E: candidate entities visibly supported by the image.
A: candidate actions/relations visibly supported by the image.
T: candidate attributes/counts/locations visibly supported by the image.
R: candidate meaning supported by mutually consistent references.
C: candidate meaning contradicted by the image or reliable references.
U: candidate details unsupported by both image and reliable references.
Q: overall ambiguity of the available evidence.

References:
{format_references(references)}

Candidate: {clean(candidate)}
{retry_text}
Use this exact one-line syntax, replacing every question mark with N, P, or S:
E=? A=? T=? R=? C=? U=? Q=?"""


def parse_feedback(text: str) -> tuple[list[str], list[float]]:
    normalized = clean(text).upper()
    match = OUTPUT_PATTERN.search(normalized)
    if match is None:
        raise ValueError(f"Invalid feedback contract: {text!r}")
    codes = list(match.groups())
    return codes, [LEVELS[code] for code in codes]


def fallback_vector() -> list[float]:
    # Exact zero plus validity=0 is the same input used by the no-feedback
    # control, so malformed generations cannot become a hidden feedback cue.
    return [0.0] * len(FIELDS)


def read_jsonl_resume(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    valid_lines: list[str] = []
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if line_index != len(lines) - 1:
                raise RuntimeError(
                    f"Corrupt non-final JSONL line {line_index + 1}: {path}"
                )
            break
        rows.append(row)
        valid_lines.append(json.dumps(row, ensure_ascii=False))
    if len(valid_lines) != len([line for line in lines if line.strip()]):
        path.write_text(
            "\n".join(valid_lines) + ("\n" if valid_lines else ""),
            encoding="utf-8",
        )
    return rows


def write_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def validate_rows(rows: Sequence[dict], sample_ids: Sequence[str]) -> None:
    if len(rows) > len(sample_ids):
        raise RuntimeError("Feedback file contains too many rows")
    for index, row in enumerate(rows):
        if row.get("format") != FORMAT:
            raise RuntimeError(f"Wrong feedback format at row {index}")
        if row.get("sample_id") != sample_ids[index]:
            raise RuntimeError(f"Feedback order mismatch at row {index}")
        values = row.get("values")
        if not isinstance(values, list) or len(values) != len(FIELDS):
            raise RuntimeError(f"Wrong feedback vector at row {index}")
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise RuntimeError(f"Invalid feedback values at row {index}")
        if row.get("parse_mode") not in {"strict", "safe_fallback"}:
            raise RuntimeError(f"Unknown feedback parse mode at row {index}")


def matrix_from_rows(
    rows: Sequence[dict],
    sample_ids: Sequence[str],
) -> np.ndarray:
    validate_rows(rows, sample_ids)
    if len(rows) != len(sample_ids):
        raise RuntimeError(
            f"Incomplete feedback: {len(rows)} != {len(sample_ids)}"
        )
    values = np.asarray([row["values"] for row in rows], dtype=np.float32)
    valid = np.asarray(
        [row["parse_mode"] == "strict" for row in rows],
        dtype=np.float32,
    )[:, None]
    return np.concatenate([values, valid], axis=1)


def selfcheck() -> None:
    codes, values = parse_feedback("E=S A=P T=N R=S C=N U=P Q=P")
    if codes != ["S", "P", "N", "S", "N", "P", "P"]:
        raise RuntimeError("Feedback code parser failed")
    if values != [1.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.5]:
        raise RuntimeError("Feedback value parser failed")

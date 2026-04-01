from __future__ import annotations

import re
from typing import Optional

from datasets import DatasetDict, load_dataset

from config import DataConfig

_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def load_gsm8k(config: DataConfig) -> DatasetDict:
    return load_dataset(config.gsm8k_dataset_name, config.gsm8k_dataset_config)


def format_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step.\n"
        "At the end, write your final answer as a single number.\n"
        f"Problem: {question.strip()}\n"
        "Solution:"
    )


def normalize_numeric_string(value: str) -> Optional[str]:
    candidate = value.strip().replace(",", "")
    if not candidate:
        return None
    if candidate.startswith("+"):
        candidate = candidate[1:]
    if candidate.endswith("."):
        candidate = candidate[:-1]
    try:
        parsed = float(candidate)
    except ValueError:
        return None
    if parsed.is_integer():
        return str(int(parsed))
    normalized = f"{parsed:.10f}".rstrip("0").rstrip(".")
    return normalized


def extract_answer(text: str) -> Optional[str]:
    stripped = text.strip()
    boxed = _BOXED_PATTERN.findall(stripped)
    if boxed:
        normalized = normalize_numeric_string(boxed[-1])
        if normalized is not None:
            return normalized

    if "####" in stripped:
        candidate = stripped.split("####")[-1].strip().splitlines()[0]
        normalized = normalize_numeric_string(candidate)
        if normalized is not None:
            return normalized

    answer_lines = re.findall(r"(?:final answer|answer is)\s*[:\-]?\s*([^\n]+)", stripped, flags=re.I)
    for candidate in reversed(answer_lines):
        number_match = _NUMBER_PATTERN.search(candidate)
        if number_match:
            normalized = normalize_numeric_string(number_match.group(0))
            if normalized is not None:
                return normalized

    matches = _NUMBER_PATTERN.findall(stripped)
    if not matches:
        return None
    return normalize_numeric_string(matches[-1])


def extract_gold_answer(solution: str) -> Optional[str]:
    return extract_answer(solution)

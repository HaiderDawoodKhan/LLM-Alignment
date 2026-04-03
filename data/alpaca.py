from __future__ import annotations

from typing import List

from datasets import Dataset, load_dataset

from config import AppConfig


def _format_alpaca_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def _map_alpaca_row(row: dict) -> dict:
    prompt = _format_alpaca_prompt(row.get("instruction", ""), row.get("input", ""))
    response = (row.get("output") or "").strip()
    return {"prompt": prompt, "response": response}


def load_alpaca_sft_datasets(config: AppConfig) -> dict[str, Dataset]:
    if config.sft.dataset_config:
        raw = load_dataset(config.sft.dataset_name, config.sft.dataset_config)
    else:
        raw = load_dataset(config.sft.dataset_name)

    train_split = raw[config.sft.train_split]
    mapped = train_split.map(_map_alpaca_row, remove_columns=train_split.column_names)
    mapped = mapped.filter(lambda row: bool(row["prompt"].strip()) and bool(row["response"].strip()))

    eval_size = min(config.sft.eval_split_size, max(len(mapped) // 100, 1), len(mapped) - 1) if len(mapped) > 1 else 0
    if eval_size > 0:
        split = mapped.train_test_split(test_size=eval_size, seed=config.runtime.seed, shuffle=True)
        sft_train = split["train"]
        sft_eval = split["test"]
    else:
        sft_train = mapped
        sft_eval = mapped.select(range(min(len(mapped), 1)))
    return {"sft_train": sft_train, "sft_eval": sft_eval}


def preview_alpaca_examples(dataset: Dataset, n: int = 3) -> List[dict]:
    preview = []
    for idx in range(min(n, len(dataset))):
        row = dataset[idx]
        preview.append({"prompt": row["prompt"], "response": row["response"]})
    return preview

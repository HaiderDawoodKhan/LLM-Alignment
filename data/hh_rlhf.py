from __future__ import annotations

from typing import Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

from config import DataConfig
from data.parsing import split_prompt_response


def load_hh_rlhf_raw(config: DataConfig) -> DatasetDict:
    if config.hh_dataset_config:
        return load_dataset(config.hh_dataset_name, config.hh_dataset_config)
    return load_dataset(config.hh_dataset_name)


def parse_hh_example(row: dict) -> Optional[dict]:
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not chosen or not rejected:
        return None
    parsed = split_prompt_response(chosen, rejected)
    if parsed is None:
        return None
    return parsed


def _build_dataset(rows: Iterable[dict]) -> Dataset:
    return Dataset.from_list([row for row in rows if row is not None])


def build_preference_datasets(config: DataConfig) -> dict[str, Dataset]:
    raw = load_hh_rlhf_raw(config)
    train_rows = [parse_hh_example(row) for row in raw[config.train_split]]
    eval_rows = [parse_hh_example(row) for row in raw[config.eval_split]]

    train_dataset = _build_dataset(train_rows)
    eval_dataset = _build_dataset(eval_rows)
    return {
        "rm_train": train_dataset,
        "rm_eval": eval_dataset,
        "dpo_train": train_dataset,
        "dpo_eval": eval_dataset,
        "sft_train": train_dataset.remove_columns(["rejected"]).rename_column("chosen", "response"),
        "sft_eval": eval_dataset.remove_columns(["rejected"]).rename_column("chosen", "response"),
        "prompt_train": train_dataset.remove_columns(["chosen", "rejected"]),
        "prompt_eval": eval_dataset.remove_columns(["chosen", "rejected"]),
    }


def preview_parsed_examples(dataset: Dataset, n: int = 3) -> List[dict]:
    preview = []
    limit = min(n, len(dataset))
    for idx in range(limit):
        row = dataset[idx]
        preview.append(
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
        )
    return preview

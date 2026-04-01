from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


def _response_start_indices(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: List[int],
) -> torch.Tensor:
    full_lengths = attention_mask.sum(dim=1)
    seq_len = input_ids.size(1)
    starts = []
    for idx, prompt_len in enumerate(prompt_lengths):
        left_pad = seq_len - int(full_lengths[idx].item())
        starts.append(left_pad + prompt_len)
    return torch.tensor(starts, dtype=torch.long)


@dataclass
class SFTCollator:
    tokenizer: object
    max_length: int

    def __call__(self, batch: List[dict]) -> dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        responses = [item["response"] for item in batch]
        full_text = [prompt + response for prompt, response in zip(prompts, responses)]

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_tokens = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        prompt_lengths = [len(ids) for ids in prompt_tokens["input_ids"]]
        response_starts = _response_start_indices(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            prompt_lengths,
        )

        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100
        for idx, start in enumerate(response_starts.tolist()):
            labels[idx, :start] = -100
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "response_starts": response_starts,
        }


@dataclass
class RMCollator:
    tokenizer: object
    max_length: int

    def __call__(self, batch: List[dict]) -> dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        chosen = [item["prompt"] + item["chosen"] for item in batch]
        rejected = [item["prompt"] + item["rejected"] for item in batch]
        prompt_tokens = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        prompt_lengths = [len(ids) for ids in prompt_tokens["input_ids"]]

        chosen_tokens = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        rejected_tokens = self.tokenizer(
            rejected,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return {
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "chosen_last_indices": chosen_tokens["attention_mask"].sum(dim=1) - 1,
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
            "rejected_last_indices": rejected_tokens["attention_mask"].sum(dim=1) - 1,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        }


@dataclass
class DPOCollator:
    tokenizer: object
    max_length: int

    def __call__(self, batch: List[dict]) -> dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        chosen = [item["prompt"] + item["chosen"] for item in batch]
        rejected = [item["prompt"] + item["rejected"] for item in batch]

        prompt_tokens = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        prompt_lengths = [len(ids) for ids in prompt_tokens["input_ids"]]
        chosen_tokens = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        rejected_tokens = self.tokenizer(
            rejected,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return {
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "chosen_response_starts": _response_start_indices(
                chosen_tokens["input_ids"],
                chosen_tokens["attention_mask"],
                prompt_lengths,
            ),
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
            "rejected_response_starts": _response_start_indices(
                rejected_tokens["input_ids"],
                rejected_tokens["attention_mask"],
                prompt_lengths,
            ),
        }

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from config import AppConfig, OptimizerConfig
from data.collators import DPOCollator, RMCollator, SFTCollator
from data.hh_rlhf import build_preference_datasets, preview_parsed_examples
from model.lora import has_lora_adapter, load_lora_adapter
from model.policy import build_policy_tokenizer
from model.reward_model import build_reward_tokenizer
from model.utils import get_torch_device, require_hf_access, resolve_dtype


def build_optimizer(parameters: Iterable[torch.nn.Parameter], config: OptimizerConfig) -> torch.optim.Optimizer:
    params = [param for param in parameters if param.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def build_hh_datasets(config: AppConfig) -> dict[str, object]:
    return build_preference_datasets(config.data)


def preview_examples(datasets: dict[str, object], n: int) -> List[dict]:
    return preview_parsed_examples(datasets["rm_train"], n=n)


def build_sft_dataloader(dataset, tokenizer, config: AppConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.sft.batch_size,
        shuffle=shuffle,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory,
        collate_fn=SFTCollator(tokenizer=tokenizer, max_length=config.data.max_seq_len),
    )


def build_rm_dataloader(dataset, tokenizer, config: AppConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.rm.batch_size,
        shuffle=shuffle,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory,
        collate_fn=RMCollator(tokenizer=tokenizer, max_length=config.data.max_seq_len),
    )


def build_dpo_dataloader(dataset, tokenizer, config: AppConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.dpo.batch_size,
        shuffle=shuffle,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory,
        collate_fn=DPOCollator(tokenizer=tokenizer, max_length=config.data.max_seq_len),
    )


def sample_prompts(prompt_dataset, batch_size: int) -> List[str]:
    if len(prompt_dataset) <= batch_size:
        return [row["prompt"] for row in prompt_dataset]
    indices = random.sample(range(len(prompt_dataset)), batch_size)
    return [prompt_dataset[idx]["prompt"] for idx in indices]


def load_policy_checkpoint(checkpoint_dir: str | Path, config: AppConfig, trainable: bool) -> tuple[torch.nn.Module, object]:
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = build_policy_tokenizer(config.model)
    dtype = resolve_dtype(config.model.use_bf16)
    try:
        if has_lora_adapter(str(checkpoint_dir)):
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model.policy_name,
                torch_dtype=dtype,
                trust_remote_code=config.model.trust_remote_code,
            )
            model = load_lora_adapter(base_model, str(checkpoint_dir))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_dir),
                torch_dtype=dtype,
                trust_remote_code=config.model.trust_remote_code,
            )
    except Exception as error:  # pragma: no cover - requires network/HF auth
        raise require_hf_access(error, config.model.policy_name)
    if not trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model, tokenizer


def load_reward_checkpoint(checkpoint_dir: str | Path, config: AppConfig, trainable: bool) -> tuple[torch.nn.Module, object]:
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = build_reward_tokenizer(config.model)
    dtype = resolve_dtype(config.model.use_bf16)
    try:
        if has_lora_adapter(str(checkpoint_dir)):
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.model.rm_name,
                num_labels=1,
                torch_dtype=dtype,
                trust_remote_code=config.model.trust_remote_code,
            )
            model = load_lora_adapter(base_model, str(checkpoint_dir))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                str(checkpoint_dir),
                torch_dtype=dtype,
                trust_remote_code=config.model.trust_remote_code,
            )
    except Exception as error:  # pragma: no cover - requires network/HF auth
        raise require_hf_access(error, config.model.rm_name)
    if not trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model, tokenizer


def default_device(config: AppConfig) -> torch.device:
    return get_torch_device(config.runtime.device)

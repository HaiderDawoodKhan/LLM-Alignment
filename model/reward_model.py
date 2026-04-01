from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LoRAConfig, ModelConfig
from model.lora import apply_lora, enable_grad_checkpointing_for_peft
from model.utils import ensure_tokenizer_padding, freeze_model, require_hf_access, resolve_dtype


def build_reward_tokenizer(config: ModelConfig) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.rm_name, trust_remote_code=config.trust_remote_code)
    except Exception as error:  # pragma: no cover - requires network/HF auth
        raise require_hf_access(error, config.rm_name)
    return ensure_tokenizer_padding(tokenizer, config.rm_padding_side)


def build_reward_model(
    config: ModelConfig,
    trainable: bool = True,
    lora_config: Optional[LoRAConfig] = None,
) -> nn.Module:
    dtype = resolve_dtype(config.use_bf16)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.rm_name,
            num_labels=1,
            torch_dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as error:  # pragma: no cover - requires network/HF auth
        raise require_hf_access(error, config.rm_name)
    if config.gradient_checkpointing:
        enable_grad_checkpointing_for_peft(model)
    if trainable and lora_config and lora_config.enabled:
        model = apply_lora(model, lora_config, task_type="SEQ_CLS")
        if config.gradient_checkpointing:
            enable_grad_checkpointing_for_peft(model)
    elif not trainable:
        freeze_model(model)
    return model


def _find_last_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1) - 1


def score_sequences(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    base_model = getattr(model, "model", None)
    score_head = getattr(model, "score", None)
    if base_model is not None and score_head is not None:
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state
        last_indices = _find_last_indices(attention_mask)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, last_indices]
        logits = score_head(pooled)
        return logits.squeeze(-1)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    if logits.ndim == 2 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    return logits

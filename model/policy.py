from __future__ import annotations

from typing import Tuple

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig
from model.lora import apply_lora, clone_reference_model, create_frozen_copy_without_lora, enable_grad_checkpointing_for_peft
from model.utils import ensure_tokenizer_padding, freeze_model, require_hf_access, resolve_dtype


def build_policy_tokenizer(config: ModelConfig) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.policy_name, trust_remote_code=config.trust_remote_code)
    except Exception as error:  # pragma: no cover - requires network/HF auth
        raise require_hf_access(error, config.policy_name)
    return ensure_tokenizer_padding(tokenizer, config.policy_padding_side)


def build_policy_model(config: ModelConfig, trainable: bool = True):
    dtype = resolve_dtype(config.use_bf16)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.policy_name,
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as error:  # pragma: no cover - requires network/HF auth
        raise require_hf_access(error, config.policy_name)

    if config.gradient_checkpointing:
        enable_grad_checkpointing_for_peft(model)
    if trainable and config.lora.enabled:
        model = apply_lora(model, config.lora, task_type="CAUSAL_LM")
        if config.gradient_checkpointing:
            enable_grad_checkpointing_for_peft(model)
    elif not trainable:
        freeze_model(model)
    return model


def build_policy_and_reference(config: ModelConfig) -> Tuple[object, object, AutoTokenizer]:
    tokenizer = build_policy_tokenizer(config)
    policy = build_policy_model(config, trainable=True)
    reference = clone_reference_model(policy)
    freeze_model(reference)
    return policy, reference, tokenizer


def build_frozen_reference_policy(policy: object) -> object:
    return create_frozen_copy_without_lora(policy)


def build_llama_backbone(config: ModelConfig, model_name: str | None = None):
    dtype = resolve_dtype(config.use_bf16)
    target_name = model_name or config.rm_name
    try:
        backbone = AutoModel.from_pretrained(
            target_name,
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as error:  # pragma: no cover - requires network/HF auth
        raise require_hf_access(error, target_name)
    freeze_model(backbone)
    return backbone

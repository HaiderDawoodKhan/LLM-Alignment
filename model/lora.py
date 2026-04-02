from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterable

import torch

from config import LoRAConfig
from model.utils import freeze_model

try:
    from peft import LoraConfig as PeftLoraConfig
    from peft import PeftModel
    from peft import TaskType, get_peft_model
except Exception:  # pragma: no cover - exercised via runtime guard tests
    PeftLoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None


def peft_available() -> bool:
    return get_peft_model is not None and PeftLoraConfig is not None and PeftModel is not None


def require_peft() -> None:
    if not peft_available():
        raise RuntimeError("PEFT is required for LoRA. Install it with `pip install peft`.")


def apply_lora(model: torch.nn.Module, config: LoRAConfig, task_type: str) -> torch.nn.Module:
    require_peft()
    task_enum = getattr(TaskType, task_type)
    lora_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        task_type=task_enum,
    )
    return get_peft_model(model, lora_config)


def clone_reference_model(model: torch.nn.Module) -> torch.nn.Module:
    cloned = copy.deepcopy(model)
    return freeze_model(cloned)


def create_frozen_copy_without_lora(model: torch.nn.Module) -> torch.nn.Module:
    return clone_reference_model(model)


@contextmanager
def disable_adapters_for_reference(model: torch.nn.Module):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield model
        return
    if hasattr(model, "disable_adapters"):
        model.disable_adapters()
        try:
            yield model
        finally:
            if hasattr(model, "enable_adapters"):
                model.enable_adapters()
        return
    yield model


def trainable_parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return {"total": total, "trainable": trainable}


def has_lora_adapter(checkpoint_dir: str) -> bool:
    from pathlib import Path

    return (Path(checkpoint_dir) / "adapter_config.json").exists()


def load_lora_adapter(base_model: torch.nn.Module, checkpoint_dir: str) -> torch.nn.Module:
    require_peft()
    return PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=True)


def enable_grad_checkpointing_for_peft(model: torch.nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

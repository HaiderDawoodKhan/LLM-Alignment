from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel

from config import LoRAConfig, ModelConfig
from model.lora import apply_lora, enable_grad_checkpointing_for_peft
from model.utils import require_hf_access, resolve_dtype


class ValueModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size")
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)
        backbone_param = next(backbone.parameters(), None)
        if backbone_param is not None:
            self.value_head.to(device=backbone_param.device, dtype=backbone_param.dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        values = self.value_head(outputs.last_hidden_state).squeeze(-1)
        return values


def build_value_model(
    config: ModelConfig,
    train_backbone: bool = False,
    lora_config: Optional[LoRAConfig] = None,
) -> ValueModel:
    dtype = resolve_dtype(config.use_bf16)
    try:
        backbone = AutoModel.from_pretrained(
            config.value_name,
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as error:  # pragma: no cover - requires network/HF auth
        raise require_hf_access(error, config.value_name)
    if config.gradient_checkpointing and train_backbone:
        enable_grad_checkpointing_for_peft(backbone)
    if train_backbone and lora_config and lora_config.enabled:
        backbone = apply_lora(backbone, lora_config, task_type="FEATURE_EXTRACTION")
        if config.gradient_checkpointing:
            enable_grad_checkpointing_for_peft(backbone)
    else:
        if hasattr(backbone, "gradient_checkpointing_disable"):
            backbone.gradient_checkpointing_disable()
        for param in backbone.parameters():
            param.requires_grad_(False)
    return ValueModel(backbone)

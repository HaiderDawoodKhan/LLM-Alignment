from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer


def get_torch_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(use_bf16: bool) -> torch.dtype:
    if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def gpu_memory_snapshot() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 0.0}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 4),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 4),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
    }


def torch_cuda_memory_allocated_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_allocated() / 1024**3, 4)


def report_model_footprint(name: str, model: torch.nn.Module) -> Dict[str, float | int | str]:
    params = count_parameters(model)
    backbone_dtype = next(model.parameters()).dtype if any(True for _ in model.parameters()) else "unknown"
    return {
        "model_name": name,
        "total_params": params["total"],
        "trainable_params": params["trainable"],
        "cuda_memory_allocated_gb": torch_cuda_memory_allocated_gb(),
        "dtype": str(backbone_dtype),
    }


def ensure_tokenizer_padding(tokenizer: AutoTokenizer, padding_side: str) -> AutoTokenizer:
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_artifact(
    model: torch.nn.Module,
    tokenizer: Optional[AutoTokenizer],
    save_dir: str | Path,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    target = Path(save_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(target))
    if extra_metadata is not None:
        with (target / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(extra_metadata, handle, indent=2, sort_keys=True)


def require_hf_access(error: Exception, model_name: str) -> RuntimeError:
    wrapped = RuntimeError(
        f"Failed to load '{model_name}'. Make sure you are logged into Hugging Face "
        "and have access to the gated model if required."
    )
    wrapped.__cause__ = error
    return wrapped

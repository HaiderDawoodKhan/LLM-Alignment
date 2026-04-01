from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return value


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += param.grad.detach().float().norm(2).item() ** 2
    return total ** 0.5


class RunLogger:
    def __init__(self, run_dir: str | Path, config: Optional[Mapping[str, Any]] = None) -> None:
        self.run_dir = ensure_dir(run_dir)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        self.start_time = time.perf_counter()
        if config is not None:
            self.write_json("config.json", config)

    def write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        with (self.run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)

    def log_metrics(self, step: int, metrics: Mapping[str, Any]) -> None:
        serializable = {"step": step, **to_serializable(metrics)}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable, sort_keys=True) + "\n")
        for key, value in serializable.items():
            if key == "step" or not isinstance(value, (int, float)):
                continue
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        self.log_metrics(-1, {"wall_time_seconds": elapsed})
        self.writer.flush()
        self.writer.close()


class StepTimer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start


from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from alignment.losses import bradley_terry_loss
from config import default_config
from model.utils import freeze_model
from model.reward_model import build_reward_model, build_reward_tokenizer, score_sequences
from model.utils import count_parameters, gpu_memory_snapshot, save_artifact
from runtime import RunLogger, StepTimer
from seed import set_seed
from train_helpers import build_hh_datasets, build_optimizer, build_rm_dataloader, default_device, preview_examples


def _reward_histogram(values: torch.Tensor, bins: int = 10) -> dict[str, list[float] | float]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"bin_edges": [], "counts": []}
    min_value = float(values.min().item())
    max_value = float(values.max().item())
    if min_value == max_value:
        max_value = min_value + 1e-6
    counts = torch.histc(values, bins=bins, min=min_value, max=max_value)
    edges = torch.linspace(min_value, max_value, bins + 1)
    return {
        "bin_edges": [float(edge.item()) for edge in edges],
        "counts": [float(count.item()) for count in counts],
    }


def evaluate_reward_model(model, dataloader, device: torch.device, lambda_reg: float) -> dict[str, object]:
    model.eval()
    totals = {"loss": 0.0, "preference_accuracy": 0.0, "reward_gap_mean": 0.0, "steps": 0.0}
    chosen_reward_values = []
    rejected_reward_values = []
    with torch.no_grad():
        for batch in dataloader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            chosen_rewards = score_sequences(model, chosen_input_ids, chosen_attention_mask)
            rejected_rewards = score_sequences(model, rejected_input_ids, rejected_attention_mask)
            loss, metrics = bradley_terry_loss(chosen_rewards, rejected_rewards, lambda_reg=lambda_reg)
            totals["loss"] += float(loss.item())
            totals["preference_accuracy"] += float(metrics["preference_accuracy"].item())
            totals["reward_gap_mean"] += float(metrics["reward_gap_mean"].item())
            totals["steps"] += 1.0
            chosen_reward_values.append(chosen_rewards.detach().cpu())
            rejected_reward_values.append(rejected_rewards.detach().cpu())
    steps = max(totals["steps"], 1.0)
    chosen_rewards = torch.cat(chosen_reward_values) if chosen_reward_values else torch.empty(0)
    rejected_rewards = torch.cat(rejected_reward_values) if rejected_reward_values else torch.empty(0)
    return {
        **{key: value / steps for key, value in totals.items() if key != "steps"},
        "chosen_reward_mean": float(chosen_rewards.mean().item()) if chosen_rewards.numel() else 0.0,
        "chosen_reward_std": float(chosen_rewards.std().item()) if chosen_rewards.numel() > 1 else 0.0,
        "rejected_reward_mean": float(rejected_rewards.mean().item()) if rejected_rewards.numel() else 0.0,
        "rejected_reward_std": float(rejected_rewards.std().item()) if rejected_rewards.numel() > 1 else 0.0,
        "chosen_reward_histogram": _reward_histogram(chosen_rewards),
        "rejected_reward_histogram": _reward_histogram(rejected_rewards),
    }


def main() -> None:
    config = default_config(run_name="reward_model")
    set_seed(config.runtime.seed)
    device = default_device(config)
    datasets = build_hh_datasets(config)
    for idx, example in enumerate(preview_examples(datasets, config.data.print_examples), start=1):
        print(f"[parsed-example-{idx}]")
        print("PROMPT:\n", example["prompt"])
        print("CHOSEN:\n", example["chosen"])
        print("REJECTED:\n", example["rejected"])

    tokenizer = build_reward_tokenizer(config.model)
    train_loader = build_rm_dataloader(datasets["rm_train"], tokenizer, config, shuffle=True)
    eval_loader = build_rm_dataloader(datasets["rm_eval"], tokenizer, config, shuffle=False)

    model = build_reward_model(config.model, trainable=True, lora_config=config.model.lora).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    optimizer = build_optimizer(model.parameters(), config.rm.optimizer)
    logger = RunLogger(config.log_dir("reward_model"), config.to_dict())
    logger.log_metrics(0, {**count_parameters(model), **gpu_memory_snapshot()})

    best_accuracy = float("-inf")
    global_step = 0
    for epoch in range(config.rm.epochs):
        model.train()
        for batch in train_loader:
            timer = StepTimer()
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            chosen_rewards = score_sequences(model, chosen_input_ids, chosen_attention_mask)
            rejected_rewards = score_sequences(model, rejected_input_ids, rejected_attention_mask)
            loss, metrics = bradley_terry_loss(chosen_rewards, rejected_rewards, lambda_reg=config.rm.lambda_reg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.rm.optimizer.grad_clip)
            optimizer.step()

            global_step += 1
            logger.log_metrics(
                global_step,
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "step_time_seconds": timer.elapsed(),
                    **{key: float(value.item()) for key, value in metrics.items()},
                    **gpu_memory_snapshot(),
                },
            )

        eval_metrics = evaluate_reward_model(model, eval_loader, device, config.rm.lambda_reg)
        scalar_eval_metrics = {f"eval_{key}": value for key, value in eval_metrics.items() if isinstance(value, (int, float))}
        logger.log_metrics(global_step, scalar_eval_metrics)
        logger.write_json(
            "reward_model_eval_summary.json",
            {
                "global_step": global_step,
                **eval_metrics,
                "target_preference_accuracy": 0.60,
                "meets_target": bool(eval_metrics["preference_accuracy"] >= 0.60),
            },
        )
        print(
            f"[rm eval] step={global_step} "
            f"loss={eval_metrics['loss']:.4f} "
            f"pref_acc={eval_metrics['preference_accuracy']:.4f} "
            f"reward_gap={eval_metrics['reward_gap_mean']:.4f}"
        )
        if eval_metrics["preference_accuracy"] >= 0.60:
            print("[rm eval] Target met: preference accuracy is at least 60%.")
        else:
            print("[rm eval] Target not met yet: preference accuracy is below 60%.")
        if eval_metrics["preference_accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["preference_accuracy"]
            frozen_model = freeze_model(copy.deepcopy(model))
            save_artifact(
                frozen_model,
                tokenizer,
                config.checkpoint_dir("reward_model"),
                extra_metadata={
                    "base_model_name": config.model.rm_name,
                    "task": "reward_model",
                    "frozen_after_training": True,
                    "eval_preference_accuracy": eval_metrics["preference_accuracy"],
                },
            )
    logger.close()


if __name__ == "__main__":
    main()

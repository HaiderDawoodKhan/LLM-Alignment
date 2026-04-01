from __future__ import annotations

from pathlib import Path

import torch

from alignment.losses import bradley_terry_loss
from config import default_config
from model.reward_model import build_reward_model, build_reward_tokenizer, score_sequences
from model.utils import count_parameters, gpu_memory_snapshot, save_artifact
from runtime import RunLogger, StepTimer
from seed import set_seed
from train_helpers import build_hh_datasets, build_optimizer, build_rm_dataloader, default_device, preview_examples


def evaluate_reward_model(model, dataloader, device: torch.device, lambda_reg: float) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "preference_accuracy": 0.0, "reward_gap_mean": 0.0, "steps": 0.0}
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
    steps = max(totals["steps"], 1.0)
    return {key: value / steps for key, value in totals.items() if key != "steps"}


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
        logger.log_metrics(global_step, {f"eval_{key}": value for key, value in eval_metrics.items()})
        if eval_metrics["preference_accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["preference_accuracy"]
            save_artifact(
                model,
                tokenizer,
                config.checkpoint_dir("reward_model"),
                extra_metadata={"base_model_name": config.model.rm_name, "task": "reward_model"},
            )
    logger.close()


if __name__ == "__main__":
    main()

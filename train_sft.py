from __future__ import annotations

import math

import torch

from alignment.rollout import generate_batch
from config import default_config
from model.policy import build_policy_model, build_policy_tokenizer
from model.utils import count_parameters, gpu_memory_snapshot, save_artifact
from runtime import RunLogger, StepTimer
from seed import set_seed
from train_helpers import build_hh_datasets, build_optimizer, build_sft_dataloader, default_device, preview_examples


def evaluate_sft(model, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += float(outputs.loss.item())
            steps += 1
    avg_loss = total_loss / max(steps, 1)
    return {"loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20))}


def generate_sft_samples(model, tokenizer, prompts: list[str], config, device: torch.device) -> list[dict[str, object]]:
    with torch.no_grad():
        _, _, _, _, responses = generate_batch(
            policy=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_length=config.data.max_seq_len,
            max_new_tokens=config.data.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            device=device,
        )
    return [
        {"prompt": prompt, "response": response}
        for prompt, response in zip(prompts, responses)
    ]


def main() -> None:
    config = default_config(run_name="policy_sft")
    set_seed(config.runtime.seed)
    device = default_device(config)
    datasets = build_hh_datasets(config)
    for idx, example in enumerate(preview_examples(datasets, config.data.print_examples), start=1):
        print(f"[parsed-example-{idx}]")
        print("PROMPT:\n", example["prompt"])
        print("CHOSEN:\n", example["chosen"])
        print("REJECTED:\n", example["rejected"])

    tokenizer = build_policy_tokenizer(config.model)
    train_loader = build_sft_dataloader(datasets["sft_train"], tokenizer, config, shuffle=True)
    eval_loader = build_sft_dataloader(datasets["sft_eval"], tokenizer, config, shuffle=False)

    model = build_policy_model(config.model, trainable=True).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    save_artifact(
        model,
        tokenizer,
        config.checkpoint_dir("policy_base"),
        extra_metadata={"base_model_name": config.model.policy_name, "task": "policy_base"},
    )

    optimizer = build_optimizer(model.parameters(), config.sft.optimizer)
    logger = RunLogger(config.log_dir("policy_sft"), config.to_dict())
    logger.log_metrics(0, {**count_parameters(model), **gpu_memory_snapshot()})
    global_step = 0

    for epoch in range(config.sft.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            timer = StepTimer()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config.sft.grad_accum_steps
            loss.backward()

            global_step += 1
            if global_step % config.sft.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.sft.optimizer.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % config.sft.eval_every == 0:
                eval_metrics = evaluate_sft(model, eval_loader, device)
                logger.log_metrics(global_step, {f"eval_{key}": value for key, value in eval_metrics.items()})

            logger.log_metrics(
                global_step,
                {
                    "epoch": epoch,
                    "loss": float(loss.item() * config.sft.grad_accum_steps),
                    "step_time_seconds": timer.elapsed(),
                    **gpu_memory_snapshot(),
                },
            )

    save_artifact(
        model,
        tokenizer,
        config.checkpoint_dir("policy_sft"),
        extra_metadata={"base_model_name": config.model.policy_name, "task": "policy_sft", "trainable_init": True},
    )
    sample_prompts = [datasets["sft_eval"][idx]["prompt"] for idx in range(min(5, len(datasets["sft_eval"])))]
    samples = generate_sft_samples(model, tokenizer, sample_prompts, config, device)
    print("[sft-samples]")
    for idx, sample in enumerate(samples, start=1):
        print(f"[sample-{idx}]")
        print("PROMPT:")
        print(sample["prompt"])
        print("RESPONSE:")
        print(sample["response"])
    logger.write_json(
        "sft_sample_generations.json",
        {"samples": samples},
    )
    save_artifact(
        model,
        tokenizer,
        config.checkpoint_dir("policy_ref"),
        extra_metadata={
            "base_model_name": config.model.policy_name,
            "task": "policy_ref",
            "frozen_reference": True,
            "reference_mode": "disable_adapters",
        },
    )
    logger.close()


if __name__ == "__main__":
    main()

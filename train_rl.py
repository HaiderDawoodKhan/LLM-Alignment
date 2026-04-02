from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List

import torch

from alignment.dpo import evaluate_dpo_policy, train_dpo_epoch
from alignment.grpo import collect_grpo_rollouts, grpo_update_epoch
from alignment.ppo import (
    collect_ppo_rollouts,
    ppo_sanity_ratio_test,
    ppo_update_epoch,
    prepare_ppo_targets,
    slice_prepared_output,
    slice_rollout_batch,
)
from alignment.rlvr import compute_verifiable_rewards, gsm8k_reward_fn, verify_gsm8k_verifier
from config import AppConfig, default_config
from data.gsm8k import extract_answer, extract_gold_answer, format_gsm8k_prompt, load_gsm8k
from eval import (
    build_gsm8k_sample_table,
    build_reward_scorer,
    compute_kl_to_reference,
    evaluate_gsm8k_pass_at_one,
    generate_responses,
)
from model.policy import build_policy_model
from model.reward_model import build_reward_model
from model.utils import count_parameters, gpu_memory_snapshot, save_artifact
from model.value_model import build_value_model
from runtime import RunLogger, StepTimer, ensure_dir
from seed import set_seed
from train_helpers import (
    build_dpo_dataloader,
    build_hh_datasets,
    build_optimizer,
    default_device,
    load_policy_checkpoint,
    load_reward_checkpoint,
    sample_prompts,
)


def configure_method(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.beta is not None:
        if args.method == "dpo":
            config.dpo.beta = args.beta
        elif args.method == "ppo":
            config.ppo.beta_kl = args.beta
        elif args.method == "grpo":
            config.grpo.beta_kl = args.beta
        elif args.method == "rlvr":
            config.rlvr.beta_kl = args.beta
    if args.clip_epsilon is not None:
        if args.method == "ppo":
            config.ppo.clip_epsilon = args.clip_epsilon
        elif args.method in {"grpo", "rlvr"}:
            getattr(config, args.method).clip_epsilon = args.clip_epsilon
    if args.group_size is not None and args.method in {"grpo", "rlvr"}:
        getattr(config, args.method).group_size = args.group_size
    return config


def reward_fn_from_model(scorer):
    def reward_fn(prompts: List[str], responses: List[str]) -> torch.Tensor:
        return scorer(prompts, responses)

    return reward_fn


def mean_group_reward(rewards: torch.Tensor, group_ids: torch.Tensor | None) -> float:
    if group_ids is None or rewards.numel() == 0:
        return float(rewards.mean().item()) if rewards.numel() else 0.0
    means = []
    for group_id in torch.unique(group_ids):
        group_rewards = rewards[group_ids == group_id]
        means.append(group_rewards.mean())
    return float(torch.stack(means).mean().item()) if means else 0.0


def _sweep_suffix(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def evaluate_reward_score_mean(
    policy,
    tokenizer,
    scorer,
    prompts: List[str],
    config: AppConfig,
    device: torch.device,
) -> float:
    responses = generate_responses(policy, tokenizer, prompts, config, device)
    return float(scorer(prompts, responses).mean().item())


def run_dpo(config: AppConfig, logger: RunLogger, device: torch.device) -> None:
    datasets = build_hh_datasets(config)
    policy, tokenizer = load_policy_checkpoint(config.checkpoint_dir("policy_sft"), config, trainable=True)
    reference, _ = load_policy_checkpoint(config.checkpoint_dir("policy_ref"), config, trainable=False)
    policy.to(device)
    reference.to(torch.device("cpu"))
    reward_model, reward_tokenizer = load_reward_checkpoint(config.checkpoint_dir("reward_model"), config, trainable=False)
    reward_model.to(device)
    scorer = build_reward_scorer(reward_model, reward_tokenizer, device, config.data.max_seq_len)
    optimizer = build_optimizer(policy.parameters(), config.dpo.optimizer)
    dataloader = build_dpo_dataloader(datasets["dpo_train"], tokenizer, config, shuffle=True)
    eval_dataloader = build_dpo_dataloader(datasets["dpo_eval"], tokenizer, config, shuffle=False)
    eval_prompts = [
        datasets["prompt_eval"][idx]["prompt"]
        for idx in range(min(config.data.eval_subset_size, len(datasets["prompt_eval"])))
    ]

    def evaluation_callback() -> Dict[str, float]:
        return evaluate_dpo_policy(
            policy=policy,
            reference=reference,
            eval_dataloader=eval_dataloader,
            eval_prompts=eval_prompts,
            tokenizer=tokenizer,
            scorer=scorer,
            max_seq_len=config.data.max_seq_len,
            max_new_tokens=config.data.max_new_tokens,
            device=device,
        )

    metrics = train_dpo_epoch(
        policy,
        reference,
        dataloader,
        optimizer,
        config.dpo.beta,
        device,
        grad_accum_steps=config.dpo.grad_accum_steps,
        logger=logger,
        log_every=config.dpo.log_every,
        eval_every=config.dpo.eval_every,
        checkpoint_dir=config.checkpoint_dir("dpo_policy"),
        checkpoint_model=policy,
        checkpoint_tokenizer=tokenizer,
        evaluation_callback=evaluation_callback,
    )
    logger.log_metrics(1, metrics | gpu_memory_snapshot())
    save_artifact(
        policy,
        tokenizer,
        config.checkpoint_dir("dpo_policy"),
        extra_metadata={"base_model_name": config.model.policy_name, "task": "dpo_policy"},
    )


def run_ppo(config: AppConfig, logger: RunLogger, device: torch.device) -> None:
    datasets = build_hh_datasets(config)
    policy, tokenizer = load_policy_checkpoint(config.checkpoint_dir("policy_sft"), config, trainable=True)
    reference, _ = load_policy_checkpoint(config.checkpoint_dir("policy_ref"), config, trainable=False)
    reward_model, reward_tokenizer = load_reward_checkpoint(config.checkpoint_dir("reward_model"), config, trainable=False)
    value_model = build_value_model(config.model, train_backbone=False, lora_config=None)
    policy.to(device)
    reference.to(device)
    reward_model.to(device)
    value_model.to(device)

    scorer = build_reward_scorer(reward_model, reward_tokenizer, device, config.data.max_seq_len)
    reward_fn = reward_fn_from_model(scorer)
    policy_optimizer = build_optimizer(policy.parameters(), config.ppo.optimizer)
    value_optimizer = build_optimizer(value_model.parameters(), config.ppo.value_optimizer)
    prompt_dataset = datasets["prompt_train"]
    logger.log_metrics(0, {**count_parameters(policy), **gpu_memory_snapshot()})

    for step in range(1, config.ppo.num_updates + 1):
        timer = StepTimer()
        prompts = sample_prompts(prompt_dataset, config.ppo.prompts_per_step)
        rollout = collect_ppo_rollouts(
            policy=policy,
            reference=reference,
            value_model=value_model,
            tokenizer=tokenizer,
            prompts=prompts,
            reward_fn=reward_fn,
            max_length=config.data.max_seq_len,
            max_new_tokens=config.data.max_new_tokens,
            temperature=config.ppo.temperature,
            top_p=config.ppo.top_p,
            device=device,
        )
        prepared = prepare_ppo_targets(
            rollout,
            beta_kl=config.ppo.beta_kl,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
        )
        assert ppo_sanity_ratio_test(rollout.old_logprobs, rollout.old_logprobs)
        metrics = {}
        epoch_metrics = []
        rollout_size = rollout.full_input_ids.size(0)
        for _ in range(config.ppo.minibatch_epochs):
            permutation = torch.randperm(rollout_size)
            for start in range(0, rollout_size, config.ppo.minibatch_size):
                indices = permutation[start : start + config.ppo.minibatch_size]
                minibatch = slice_rollout_batch(rollout, indices)
                minibatch_prepared = slice_prepared_output(prepared, indices)
                metrics = ppo_update_epoch(
                    policy=policy,
                    value_model=value_model,
                    batch=minibatch,
                    prepared=minibatch_prepared,
                    policy_optimizer=policy_optimizer,
                    value_optimizer=value_optimizer,
                    clip_epsilon=config.ppo.clip_epsilon,
                    kl_loss_coef=config.ppo.kl_loss_coef,
                    value_coef=config.ppo.value_coef,
                    entropy_coef=config.ppo.entropy_coef,
                    device=device,
                )
                epoch_metrics.append(metrics)
        averaged_metrics = {}
        if epoch_metrics:
            for key in epoch_metrics[0]:
                averaged_metrics[key] = sum(metric[key] for metric in epoch_metrics) / len(epoch_metrics)
        logger.log_metrics(step, averaged_metrics | {"step_time_seconds": timer.elapsed()} | gpu_memory_snapshot())

    save_artifact(
        policy,
        tokenizer,
        config.checkpoint_dir("ppo_policy"),
        extra_metadata={"base_model_name": config.model.policy_name, "task": "ppo_policy"},
    )


def run_grpo_like(
    config: AppConfig,
    logger: RunLogger,
    device: torch.device,
    method: str,
    artifact_name: str | None = None,
) -> tuple[torch.nn.Module, object, torch.nn.Module]:
    policy, tokenizer = load_policy_checkpoint(config.checkpoint_dir("policy_sft"), config, trainable=True)
    reference, _ = load_policy_checkpoint(config.checkpoint_dir("policy_ref"), config, trainable=False)
    policy.to(device)
    reference.to(torch.device("cpu"))

    if method == "rlvr":
        gsm8k = load_gsm8k(config.data)
        train_dataset = gsm8k[config.data.train_split]
        prompts = [
            format_gsm8k_prompt(train_dataset[idx]["question"], tokenizer=tokenizer, max_question_tokens=200)
            for idx in range(len(train_dataset))
        ]
        gold_answers = [extract_gold_answer(train_dataset[idx]["answer"]) for idx in range(len(train_dataset))]
        verifier_check = verify_gsm8k_verifier(train_dataset, num_examples=20)
        logger.write_json("rlvr_verifier_check", verifier_check)
        print(
            "[rlvr verifier]",
            {
                "gold_all_correct": verifier_check["gold_all_correct"],
                "wrong_all_zero": verifier_check["wrong_all_zero"],
            },
        )
    else:
        datasets = build_hh_datasets(config)
        train_dataset = datasets["prompt_train"]
        prompts = [train_dataset[idx]["prompt"] for idx in range(len(train_dataset))]
        gold_answers = []

    optimizer = build_optimizer(policy.parameters(), getattr(config, method).optimizer)

    if method == "rlvr":
        reward_fn = gsm8k_reward_fn(gold_answers)
    else:
        reward_model, reward_tokenizer = load_reward_checkpoint(config.checkpoint_dir("reward_model"), config, trainable=False)
        reward_model.to(device)
        scorer = build_reward_scorer(reward_model, reward_tokenizer, device, config.data.max_seq_len)
        reward_fn = reward_fn_from_model(scorer)

    cfg = getattr(config, method)
    logger.log_metrics(0, {**count_parameters(policy), **gpu_memory_snapshot()})
    for step in range(1, cfg.num_updates + 1):
        timer = StepTimer()
        if len(prompts) <= cfg.prompts_per_step:
            indices = list(range(len(prompts)))
        else:
            indices = random.sample(range(len(prompts)), cfg.prompts_per_step)
        batch_prompts = [prompts[idx] for idx in indices]
        if method == "rlvr":
            gold_batch = [gold_answers[idx] for idx in indices for _ in range(cfg.group_size)]
            reward_fn = gsm8k_reward_fn(gold_batch)
        rollout = collect_grpo_rollouts(
            policy=policy,
            reference=reference,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            reward_fn=reward_fn,
            max_length=config.data.max_seq_len,
            max_new_tokens=config.data.rlvr_max_new_tokens if method == "rlvr" else config.data.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            group_size=cfg.group_size,
            device=device,
        )
        epoch_metrics = []
        for _ in range(cfg.minibatch_epochs):
            epoch_metrics.append(
                grpo_update_epoch(
                    policy=policy,
                    batch=rollout,
                    optimizer=optimizer,
                    clip_epsilon=cfg.clip_epsilon,
                    beta_kl=cfg.beta_kl,
                    device=device,
                    minibatch_size=cfg.minibatch_size,
                    update_chunk_size=cfg.update_chunk_size,
                )
            )
        metrics = {}
        if epoch_metrics:
            for key in epoch_metrics[0]:
                metrics[key] = sum(metric[key] for metric in epoch_metrics) / len(epoch_metrics)
        metrics["mean_response_length"] = float(rollout.response_lengths.to(torch.float32).mean().item())
        metrics["group_reward_mean"] = mean_group_reward(rollout.rm_rewards, rollout.group_ids)
        if method == "rlvr":
            batch_rewards = compute_verifiable_rewards(rollout.responses, gold_batch)
            metrics["parseable_fraction"] = float(
                torch.tensor([extract_answer(response) is not None for response in rollout.responses], dtype=torch.float32)
                .mean()
                .item()
            )
            metrics["batch_pass_at_one"] = float(batch_rewards.mean().item())
            if step % cfg.eval_every == 0:
                metrics.update(
                    evaluate_gsm8k_pass_at_one(
                        policy,
                        tokenizer,
                        config,
                        device,
                        max_examples=config.data.eval_subset_size,
                    )
                )
        if metrics.get("degenerate_warning", 0.0) > 0:
            print(
                f"[{method} warning] Degenerate group fraction is {metrics['degenerate_fraction']:.2%}, "
                "which is above the 30% threshold and may indicate a weak reward signal."
            )
        logger.log_metrics(step, metrics | {"step_time_seconds": timer.elapsed()} | gpu_memory_snapshot())

    save_artifact(
        policy,
        tokenizer,
        config.checkpoint_dir(artifact_name or f"{method}_policy"),
        extra_metadata={"base_model_name": config.model.policy_name, "task": artifact_name or f"{method}_policy"},
    )
    if method == "rlvr":
        sample_table = build_gsm8k_sample_table(policy, tokenizer, config, device, max_examples=5)
        logger.write_json("rlvr_sample_table", {"examples": sample_table})
    return policy, tokenizer, reference


def run_grpo_beta_sweep(base_config: AppConfig, device: torch.device) -> None:
    summary = {"runs": []}
    for beta in base_config.ablations.ppo_grpo_beta:
        cfg = copy.deepcopy(base_config)
        cfg.runtime.run_name = f"grpo_beta_{_sweep_suffix(beta)}"
        cfg.grpo.beta_kl = beta
        cfg.grpo.num_updates = 200

        logger = RunLogger(cfg.log_dir(cfg.runtime.run_name), cfg.to_dict())
        set_seed(cfg.runtime.seed)
        policy, tokenizer, reference = run_grpo_like(
            cfg,
            logger,
            device,
            "grpo",
            artifact_name=f"grpo_beta_{_sweep_suffix(beta)}_policy",
        )

        datasets = build_hh_datasets(cfg)
        eval_prompts = [
            datasets["prompt_eval"][idx]["prompt"]
            for idx in range(min(cfg.data.eval_subset_size, len(datasets["prompt_eval"])))
        ]
        reward_model, reward_tokenizer = load_reward_checkpoint(cfg.checkpoint_dir("reward_model"), cfg, trainable=False)
        reward_model.to(device)
        scorer = build_reward_scorer(reward_model, reward_tokenizer, device, cfg.data.max_seq_len)

        rm_score_mean = evaluate_reward_score_mean(policy, tokenizer, scorer, eval_prompts, cfg, device)
        kl_to_reference = compute_kl_to_reference(policy, reference, tokenizer, eval_prompts, cfg, device)
        sample_prompts = eval_prompts[:5]
        sample_responses = generate_responses(policy, tokenizer, sample_prompts, cfg, device)
        sample_scores = scorer(sample_prompts, sample_responses)
        manual_samples = [
            {
                "prompt": prompt,
                "response": response,
                "rm_score": float(score.item()),
            }
            for prompt, response, score in zip(sample_prompts, sample_responses, sample_scores)
        ]

        result = {
            "beta": beta,
            "run_name": cfg.runtime.run_name,
            "checkpoint_name": f"grpo_beta_{_sweep_suffix(beta)}_policy",
            "rm_score_mean": rm_score_mean,
            "kl_to_reference": kl_to_reference,
            "manual_inspection_samples": manual_samples,
        }
        logger.write_json("grpo_beta_sweep_result.json", result)
        logger.close()
        summary["runs"].append(result)

    summary_dir = ensure_dir(Path(base_config.runtime.logs_dir) / "grpo_beta_sweep")
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO, DPO, GRPO, or RLVR.")
    parser.add_argument("--method", required=True, choices=["ppo", "dpo", "grpo", "rlvr"])
    parser.add_argument("--beta", type=float)
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--beta-sweep", action="store_true", help="Run the configured beta sweep for GRPO.")
    args = parser.parse_args()

    config = configure_method(default_config(run_name=args.method), args)
    set_seed(config.runtime.seed)
    device = default_device(config)

    if args.beta_sweep:
        if args.method != "grpo":
            raise ValueError("--beta-sweep is currently supported only for --method grpo.")
        run_grpo_beta_sweep(config, device)
        return

    logger = RunLogger(config.log_dir(args.method), config.to_dict())

    if args.method == "dpo":
        run_dpo(config, logger, device)
    elif args.method == "ppo":
        run_ppo(config, logger, device)
    else:
        run_grpo_like(config, logger, device, args.method)

    logger.close()


if __name__ == "__main__":
    main()

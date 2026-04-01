from __future__ import annotations

import argparse
import random
from typing import Dict, List

import torch

from alignment.dpo import train_dpo_epoch
from alignment.grpo import collect_grpo_rollouts, grpo_update_epoch
from alignment.ppo import collect_ppo_rollouts, ppo_sanity_ratio_test, ppo_update_epoch, prepare_ppo_targets
from alignment.rlvr import gsm8k_reward_fn
from config import AppConfig, default_config
from data.gsm8k import extract_gold_answer, format_gsm8k_prompt, load_gsm8k
from eval import build_reward_scorer, rm_win_rate_vs_sft
from model.policy import build_policy_model
from model.reward_model import build_reward_model
from model.utils import count_parameters, gpu_memory_snapshot, save_artifact
from model.value_model import build_value_model
from runtime import RunLogger, StepTimer
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


def run_dpo(config: AppConfig, logger: RunLogger, device: torch.device) -> None:
    datasets = build_hh_datasets(config)
    policy, tokenizer = load_policy_checkpoint(config.checkpoint_dir("policy_sft"), config, trainable=True)
    reference, _ = load_policy_checkpoint(config.checkpoint_dir("policy_ref"), config, trainable=False)
    policy.to(device)
    reference.to(device)
    optimizer = build_optimizer(policy.parameters(), config.dpo.optimizer)
    dataloader = build_dpo_dataloader(datasets["dpo_train"], tokenizer, config, shuffle=True)
    metrics = train_dpo_epoch(policy, reference, dataloader, optimizer, config.dpo.beta, device)
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
        for _ in range(config.ppo.minibatch_epochs):
            metrics = ppo_update_epoch(
                policy=policy,
                value_model=value_model,
                batch=rollout,
                prepared=prepared,
                policy_optimizer=policy_optimizer,
                value_optimizer=value_optimizer,
                clip_epsilon=config.ppo.clip_epsilon,
                value_coef=config.ppo.value_coef,
                entropy_coef=config.ppo.entropy_coef,
                device=device,
            )
        logger.log_metrics(step, metrics | {"step_time_seconds": timer.elapsed()} | gpu_memory_snapshot())

    save_artifact(
        policy,
        tokenizer,
        config.checkpoint_dir("ppo_policy"),
        extra_metadata={"base_model_name": config.model.policy_name, "task": "ppo_policy"},
    )


def run_grpo_like(config: AppConfig, logger: RunLogger, device: torch.device, method: str) -> None:
    if method == "rlvr":
        gsm8k = load_gsm8k(config.data)
        train_dataset = gsm8k[config.data.train_split]
        prompts = [format_gsm8k_prompt(train_dataset[idx]["question"]) for idx in range(len(train_dataset))]
        gold_answers = [extract_gold_answer(train_dataset[idx]["answer"]) for idx in range(len(train_dataset))]
    else:
        datasets = build_hh_datasets(config)
        train_dataset = datasets["prompt_train"]
        prompts = [train_dataset[idx]["prompt"] for idx in range(len(train_dataset))]
        gold_answers = []

    policy, tokenizer = load_policy_checkpoint(config.checkpoint_dir("policy_sft"), config, trainable=True)
    reference, _ = load_policy_checkpoint(config.checkpoint_dir("policy_ref"), config, trainable=False)
    policy.to(device)
    reference.to(device)
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
            gold_batch = [gold_answers[idx] for idx in indices]
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
        metrics = {}
        for _ in range(cfg.minibatch_epochs):
            metrics = grpo_update_epoch(
                policy=policy,
                batch=rollout,
                optimizer=optimizer,
                clip_epsilon=cfg.clip_epsilon,
                beta_kl=cfg.beta_kl,
                device=device,
            )
        logger.log_metrics(step, metrics | {"step_time_seconds": timer.elapsed()} | gpu_memory_snapshot())

    save_artifact(
        policy,
        tokenizer,
        config.checkpoint_dir(f"{method}_policy"),
        extra_metadata={"base_model_name": config.model.policy_name, "task": f"{method}_policy"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO, DPO, GRPO, or RLVR.")
    parser.add_argument("--method", required=True, choices=["ppo", "dpo", "grpo", "rlvr"])
    parser.add_argument("--beta", type=float)
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--group-size", type=int)
    args = parser.parse_args()

    config = configure_method(default_config(run_name=args.method), args)
    set_seed(config.runtime.seed)
    device = default_device(config)
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

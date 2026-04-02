from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import torch

from alignment.kl import sampled_token_kl
from alignment.losses import (
    entropy_bonus_from_logprobs,
    ppo_clipped_loss,
    sampled_token_kl_penalty,
    value_function_loss,
)
from alignment.rollout import RolloutBatch, collect_rollout_batch, forward_response_logprobs, forward_values
from runtime import gradient_norm


@dataclass
class PPOStepOutput:
    advantages: torch.Tensor
    targets: torch.Tensor
    token_rewards: torch.Tensor


def slice_rollout_batch(batch: RolloutBatch, indices: torch.Tensor) -> RolloutBatch:
    rows = indices.tolist()
    return RolloutBatch(
        prompts=[batch.prompts[idx] for idx in rows],
        prompt_input_ids=batch.prompt_input_ids.index_select(0, indices),
        prompt_attention_mask=batch.prompt_attention_mask.index_select(0, indices),
        full_input_ids=batch.full_input_ids.index_select(0, indices),
        full_attention_mask=batch.full_attention_mask.index_select(0, indices),
        response_ids=batch.response_ids.index_select(0, indices),
        response_starts=batch.response_starts.index_select(0, indices),
        response_mask=batch.response_mask.index_select(0, indices),
        response_lengths=batch.response_lengths.index_select(0, indices),
        responses=[batch.responses[idx] for idx in rows],
        old_logprobs=batch.old_logprobs.index_select(0, indices),
        ref_logprobs=batch.ref_logprobs.index_select(0, indices),
        rewards=batch.rewards.index_select(0, indices),
        rm_rewards=batch.rm_rewards.index_select(0, indices),
        values=batch.values.index_select(0, indices) if batch.values is not None else None,
        group_ids=batch.group_ids.index_select(0, indices) if batch.group_ids is not None else None,
    )


def slice_prepared_output(prepared: PPOStepOutput, indices: torch.Tensor) -> PPOStepOutput:
    return PPOStepOutput(
        advantages=prepared.advantages.index_select(0, indices),
        targets=prepared.targets.index_select(0, indices),
        token_rewards=prepared.token_rewards.index_select(0, indices),
    )


def build_terminal_rewards(
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    sequence_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    beta_kl: float,
) -> torch.Tensor:
    rewards = sampled_token_kl_penalty(old_logprobs, ref_logprobs, beta_kl)
    last_indices = response_mask.sum(dim=1).long() - 1
    for idx, last_index in enumerate(last_indices.tolist()):
        if last_index >= 0:
            rewards[idx, last_index] += sequence_rewards[idx]
    rewards = rewards * response_mask.to(rewards.dtype)
    return rewards


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)
    next_values = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)
    for step in reversed(range(rewards.size(1))):
        valid = mask[:, step].to(rewards.dtype)
        delta = rewards[:, step] + gamma * next_values - values[:, step]
        last_advantage = (delta + gamma * gae_lambda * last_advantage) * valid
        advantages[:, step] = last_advantage
        next_values = torch.where(mask[:, step], values[:, step], next_values)
    targets = values + advantages
    valid_advantages = advantages[mask]
    if valid_advantages.numel() > 1:
        mean = valid_advantages.mean()
        std = valid_advantages.std().clamp_min(1e-8)
        advantages = torch.where(mask, (advantages - mean) / std, advantages)
    return advantages, targets


def ppo_sanity_ratio_test(logp_old: torch.Tensor, logp_new: torch.Tensor) -> bool:
    return torch.allclose(torch.exp(logp_new - logp_old), torch.ones_like(logp_old), atol=1e-5)


def ppo_clipping_reference_value(ratio: float, advantage: float, clip_epsilon: float) -> float:
    unclipped = ratio * advantage
    clipped = max(min(ratio, 1.0 + clip_epsilon), 1.0 - clip_epsilon) * advantage
    return min(unclipped, clipped)


def collect_ppo_rollouts(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    value_model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    reward_fn: Callable[[Sequence[str], list[str]], torch.Tensor],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> RolloutBatch:
    return collect_rollout_batch(
        policy=policy,
        reference=reference,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        device=device,
        reward_fn=reward_fn,
        value_model=value_model,
    )


def prepare_ppo_targets(batch: RolloutBatch, beta_kl: float, gamma: float, gae_lambda: float) -> PPOStepOutput:
    assert batch.values is not None, "PPO requires cached value predictions."
    rewards = build_terminal_rewards(batch.old_logprobs, batch.ref_logprobs, batch.rm_rewards, batch.response_mask, beta_kl)
    advantages, targets = compute_gae(rewards, batch.values, batch.response_mask.bool(), gamma=gamma, gae_lambda=gae_lambda)
    return PPOStepOutput(advantages=advantages, targets=targets, token_rewards=rewards)


def ppo_update_epoch(
    policy: torch.nn.Module,
    value_model: torch.nn.Module,
    batch: RolloutBatch,
    prepared: PPOStepOutput,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    clip_epsilon: float,
    kl_loss_coef: float,
    value_coef: float,
    entropy_coef: float,
    device: torch.device,
) -> Dict[str, float]:
    local = batch.to(device)
    advantages = prepared.advantages.to(device)
    targets = prepared.targets.to(device)
    token_rewards = prepared.token_rewards.to(device)

    policy.train()
    value_model.train()
    new_logprobs, response_mask = forward_response_logprobs(
        policy, local.full_input_ids, local.full_attention_mask, local.response_starts
    )
    value_pred, _ = forward_values(value_model, local.full_input_ids, local.full_attention_mask, local.response_starts)
    policy_loss, policy_metrics = ppo_clipped_loss(
        new_logprobs,
        local.old_logprobs,
        advantages,
        response_mask,
        clip_epsilon=clip_epsilon,
    )
    critic_loss = value_function_loss(value_pred, targets, response_mask)
    entropy = entropy_bonus_from_logprobs(new_logprobs, response_mask)
    kl_value = sampled_token_kl(new_logprobs, local.ref_logprobs, response_mask)
    total_loss = policy_loss + value_coef * critic_loss + kl_loss_coef * kl_value - entropy_coef * entropy

    policy_optimizer.zero_grad(set_to_none=True)
    value_optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    policy_optimizer.step()
    value_optimizer.step()

    return {
        "loss_total": float(total_loss.detach().item()),
        "loss_policy": float(policy_loss.detach().item()),
        "loss_value": float(critic_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "kl": float(kl_value.detach().item()),
        "reward_mean": float((token_rewards * response_mask).sum(dim=1).mean().item()),
        "grad_norm_policy": gradient_norm(policy.parameters()),
        "grad_norm_value": gradient_norm(value_model.parameters()),
        **{key: float(value.item()) for key, value in policy_metrics.items()},
    }

from __future__ import annotations

from typing import Callable, Dict, Sequence

import torch

from alignment.kl import sampled_token_kl
from alignment.losses import grpo_clipped_loss, sampled_token_kl_penalty
from alignment.rollout import RolloutBatch, collect_rollout_batch, forward_response_logprobs
from runtime import gradient_norm


def compute_group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    advantages = torch.zeros_like(rewards)
    degenerate = 0
    for group_id in torch.unique(group_ids):
        indices = torch.nonzero(group_ids == group_id, as_tuple=False).squeeze(-1)
        group_rewards = rewards[indices]
        if torch.allclose(group_rewards, group_rewards[:1]):
            degenerate += 1
        advantages[indices] = group_rewards - group_rewards.mean()
    degenerate_fraction = degenerate / max(torch.unique(group_ids).numel(), 1)
    return advantages, degenerate_fraction


def broadcast_group_advantages(
    sequence_advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    broadcast = sequence_advantages.unsqueeze(1).expand_as(response_mask).to(torch.float32)
    broadcast = torch.where(response_mask.bool(), broadcast, torch.zeros_like(broadcast))
    valid = broadcast[response_mask.bool()]
    if valid.numel() > 1:
        mean = valid.mean()
        std = valid.std().clamp_min(1e-8)
        broadcast = torch.where(response_mask.bool(), (broadcast - mean) / std, broadcast)
    return broadcast


def collect_grpo_rollouts(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    reward_fn: Callable[[Sequence[str], list[str]], torch.Tensor],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    group_size: int,
    device: torch.device,
) -> RolloutBatch:
    expanded_prompts = []
    group_ids = []
    for group_idx, prompt in enumerate(prompts):
        for _ in range(group_size):
            expanded_prompts.append(prompt)
            group_ids.append(group_idx)
    return collect_rollout_batch(
        policy=policy,
        reference=reference,
        tokenizer=tokenizer,
        prompts=expanded_prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        device=device,
        reward_fn=reward_fn,
        group_ids=torch.tensor(group_ids, dtype=torch.long),
    )


def grpo_update_epoch(
    policy: torch.nn.Module,
    batch: RolloutBatch,
    optimizer: torch.optim.Optimizer,
    clip_epsilon: float,
    beta_kl: float,
    device: torch.device,
) -> Dict[str, float]:
    if batch.group_ids is None:
        raise ValueError("GRPO requires group ids.")

    local = batch.to(device)
    group_advantages, degenerate_fraction = compute_group_relative_advantages(local.rm_rewards, local.group_ids)
    token_advantages = broadcast_group_advantages(group_advantages, local.response_mask)
    lengths = local.response_lengths.unsqueeze(1).expand_as(local.response_mask)

    policy.train()
    new_logprobs, response_mask = forward_response_logprobs(
        policy, local.full_input_ids, local.full_attention_mask, local.response_starts
    )
    grpo_loss, metrics = grpo_clipped_loss(
        new_logprobs,
        local.old_logprobs,
        token_advantages,
        response_mask,
        clip_epsilon=clip_epsilon,
        lengths=lengths,
    )
    kl_reward = sampled_token_kl_penalty(new_logprobs, local.ref_logprobs, beta_kl)
    total_loss = grpo_loss + (kl_reward * response_mask).sum() / response_mask.sum().clamp_min(1)
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    kl_value = sampled_token_kl(new_logprobs, local.ref_logprobs, response_mask)

    return {
        "loss_total": float(total_loss.detach().item()),
        "loss_grpo": float(grpo_loss.detach().item()),
        "kl": float(kl_value.detach().item()),
        "reward_mean": float(local.rm_rewards.mean().item()),
        "degenerate_fraction": float(degenerate_fraction),
        "grad_norm": gradient_norm(policy.parameters()),
        **{key: float(value.item()) for key, value in metrics.items()},
    }

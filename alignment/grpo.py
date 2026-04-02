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


def prepare_grpo_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    sequence_advantages, degenerate_fraction = compute_group_relative_advantages(rewards, group_ids)
    token_advantages = broadcast_group_advantages(sequence_advantages, response_mask)
    return token_advantages, degenerate_fraction


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
    minibatch_size: int = 8,
    update_chunk_size: int = 8,
) -> Dict[str, float]:
    if batch.group_ids is None:
        raise ValueError("GRPO requires group ids.")

    local = batch.to(device)
    token_advantages, degenerate_fraction = prepare_grpo_advantages(local.rm_rewards, local.group_ids, local.response_mask)
    lengths = local.response_lengths.unsqueeze(1).expand_as(local.response_mask)

    policy.train()
    optimizer.zero_grad(set_to_none=True)

    total_valid_tokens = local.response_mask.sum().clamp_min(1).to(torch.float32)
    metrics_accum = {
        "loss_total": 0.0,
        "loss_grpo": 0.0,
        "kl": 0.0,
        "ratio_mean": 0.0,
        "clip_fraction": 0.0,
    }

    batch_size = local.full_input_ids.size(0)
    permutation = torch.randperm(batch_size, device=device)
    for mini_start in range(0, batch_size, minibatch_size):
        mini_indices = permutation[mini_start : mini_start + minibatch_size]
        mini_input_ids = local.full_input_ids.index_select(0, mini_indices)
        mini_attention_mask = local.full_attention_mask.index_select(0, mini_indices)
        mini_response_starts = local.response_starts.index_select(0, mini_indices)
        mini_old_logprobs = local.old_logprobs.index_select(0, mini_indices)
        mini_ref_logprobs = local.ref_logprobs.index_select(0, mini_indices)
        mini_advantages = token_advantages.index_select(0, mini_indices)
        mini_lengths = lengths.index_select(0, mini_indices)
        mini_response_mask = local.response_mask.index_select(0, mini_indices)

        mini_batch_size = mini_input_ids.size(0)
        for chunk_start in range(0, mini_batch_size, update_chunk_size):
            chunk_end = min(chunk_start + update_chunk_size, mini_batch_size)
            chunk_input_ids = mini_input_ids[chunk_start:chunk_end]
            chunk_attention_mask = mini_attention_mask[chunk_start:chunk_end]
            chunk_response_starts = mini_response_starts[chunk_start:chunk_end]
            chunk_old_logprobs = mini_old_logprobs[chunk_start:chunk_end]
            chunk_ref_logprobs = mini_ref_logprobs[chunk_start:chunk_end]
            chunk_advantages = mini_advantages[chunk_start:chunk_end]
            chunk_lengths = mini_lengths[chunk_start:chunk_end]

            new_logprobs, response_mask = forward_response_logprobs(
                policy, chunk_input_ids, chunk_attention_mask, chunk_response_starts
            )
            grpo_loss, metrics = grpo_clipped_loss(
                new_logprobs,
                chunk_old_logprobs,
                chunk_advantages,
                response_mask,
                clip_epsilon=clip_epsilon,
                lengths=chunk_lengths,
            )
            kl_reward = sampled_token_kl_penalty(new_logprobs, chunk_ref_logprobs, beta_kl)
            total_loss = grpo_loss + (kl_reward * response_mask).sum() / response_mask.sum().clamp_min(1)

            chunk_valid_tokens = response_mask.sum().to(torch.float32)
            weight = (chunk_valid_tokens / total_valid_tokens).to(total_loss.dtype)
            (total_loss * weight).backward()

            metrics_accum["loss_total"] += float((total_loss.detach() * weight).item())
            metrics_accum["loss_grpo"] += float((grpo_loss.detach() * weight).item())
            metrics_accum["kl"] += float(sampled_token_kl(new_logprobs, chunk_ref_logprobs, response_mask).item() * weight.item())
            metrics_accum["ratio_mean"] += float(metrics["ratio_mean"].item() * weight.item())
            metrics_accum["clip_fraction"] += float(metrics["clip_fraction"].item() * weight.item())

    optimizer.step()

    return {
        "loss_total": metrics_accum["loss_total"],
        "loss_grpo": metrics_accum["loss_grpo"],
        "kl": metrics_accum["kl"],
        "reward_mean": float(local.rm_rewards.mean().item()),
        "degenerate_fraction": float(degenerate_fraction),
        "grad_norm": gradient_norm(policy.parameters()),
        "ratio_mean": metrics_accum["ratio_mean"],
        "clip_fraction": metrics_accum["clip_fraction"],
        "degenerate_warning": float(degenerate_fraction > 0.30),
    }

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def bradley_terry_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    lambda_reg: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reward_gap = chosen_rewards - rejected_rewards
    pref_loss = -F.logsigmoid(reward_gap).mean()
    reg_loss = lambda_reg * ((chosen_rewards.square() + rejected_rewards.square()).mean())
    loss = pref_loss + reg_loss
    metrics = {
        "loss_pref": pref_loss.detach(),
        "loss_reg": reg_loss.detach(),
        "preference_accuracy": (reward_gap > 0).float().mean().detach(),
        "reward_gap_mean": reward_gap.mean().detach(),
        "reward_mean": torch.cat([chosen_rewards, rejected_rewards]).mean().detach(),
        "reward_std": torch.cat([chosen_rewards, rejected_rewards]).std().detach(),
    }
    return loss, metrics


def dpo_loss(
    delta_theta: torch.Tensor,
    delta_ref: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    z = beta * (delta_theta - delta_ref)
    loss = -F.logsigmoid(z).mean()
    metrics = {
        "implicit_margin": z.mean().detach(),
        "preference_accuracy": (delta_theta > 0).float().mean().detach(),
    }
    return loss, metrics


def ppo_clipped_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    clipped = clipped_ratio * advantages
    surrogate = torch.minimum(unclipped, clipped)
    loss = -masked_mean(surrogate, mask)
    metrics = {
        "ratio_mean": masked_mean(ratio, mask).detach(),
        "clip_fraction": masked_mean((torch.abs(ratio - 1.0) > clip_epsilon).float(), mask).detach(),
    }
    return loss, metrics


def value_function_loss(
    value_pred: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return masked_mean((value_pred - targets).square(), mask)


def entropy_bonus_from_logprobs(logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return -masked_mean(logprobs, mask)


def sampled_token_kl_penalty(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return -beta * (policy_logprobs - reference_logprobs)


def grpo_clipped_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_epsilon: float,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    objective = torch.minimum(unclipped, clipped)
    normalized = objective / lengths.clamp_min(1).to(objective.dtype)
    loss = -masked_mean(normalized, mask)
    metrics = {
        "ratio_mean": masked_mean(ratio, mask).detach(),
        "clip_fraction": masked_mean((torch.abs(ratio - 1.0) > clip_epsilon).float(), mask).detach(),
    }
    return loss, metrics

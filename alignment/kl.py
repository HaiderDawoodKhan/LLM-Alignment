from __future__ import annotations

import torch

from alignment.losses import masked_mean


def sampled_token_kl(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return masked_mean(policy_logprobs - reference_logprobs, mask)


def full_vocab_kl_from_logits(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
    policy_probs = policy_log_probs.exp()
    token_kl = (policy_probs * (policy_log_probs - reference_log_probs)).sum(dim=-1)
    return masked_mean(token_kl, mask)

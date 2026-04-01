from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader

from alignment.kl import sampled_token_kl
from alignment.losses import dpo_loss
from alignment.rollout import forward_response_logprobs
from runtime import gradient_norm


def sequence_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_idx: torch.Tensor,
) -> torch.Tensor:
    token_logprobs, response_mask = forward_response_logprobs(model, input_ids, attention_mask, response_start_idx)
    return (token_logprobs * response_mask).sum(dim=1)


def train_dpo_epoch(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: torch.device,
) -> Dict[str, float]:
    policy.train()
    reference.eval()
    totals: Dict[str, float] = {"loss": 0.0, "preference_accuracy": 0.0, "kl": 0.0, "grad_norm": 0.0, "steps": 0.0}
    for batch in dataloader:
        chosen_input_ids = batch["chosen_input_ids"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        chosen_response_starts = batch["chosen_response_starts"].to(device)
        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)
        rejected_response_starts = batch["rejected_response_starts"].to(device)

        chosen_logp = sequence_logprob(policy, chosen_input_ids, chosen_attention_mask, chosen_response_starts)
        rejected_logp = sequence_logprob(policy, rejected_input_ids, rejected_attention_mask, rejected_response_starts)
        with torch.no_grad():
            chosen_ref = sequence_logprob(reference, chosen_input_ids, chosen_attention_mask, chosen_response_starts)
            rejected_ref = sequence_logprob(reference, rejected_input_ids, rejected_attention_mask, rejected_response_starts)

        loss, metrics = dpo_loss(chosen_logp - rejected_logp, chosen_ref - rejected_ref, beta=beta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        totals["loss"] += float(loss.detach().item())
        totals["preference_accuracy"] += float(metrics["preference_accuracy"].item())
        totals["grad_norm"] += gradient_norm(policy.parameters())
        totals["steps"] += 1.0

        with torch.no_grad():
            chosen_token_logp, chosen_mask = forward_response_logprobs(
                policy, chosen_input_ids, chosen_attention_mask, chosen_response_starts
            )
            chosen_ref_token_logp, _ = forward_response_logprobs(
                reference, chosen_input_ids, chosen_attention_mask, chosen_response_starts
            )
            totals["kl"] += float(sampled_token_kl(chosen_token_logp, chosen_ref_token_logp, chosen_mask).item())

    steps = max(totals["steps"], 1.0)
    return {key: value / steps for key, value in totals.items() if key != "steps"}

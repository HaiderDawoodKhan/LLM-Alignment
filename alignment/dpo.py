from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from alignment.kl import sampled_token_kl
from alignment.losses import dpo_loss
from alignment.rollout import forward_response_logprobs, generate_batch
from model.lora import disable_adapters_for_reference
from runtime import gradient_norm
from model.utils import save_artifact


def sequence_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_idx: torch.Tensor,
) -> torch.Tensor:
    token_logprobs, response_mask = forward_response_logprobs(model, input_ids, attention_mask, response_start_idx)
    return (token_logprobs * response_mask).sum(dim=1)


def evaluate_dpo_policy(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    eval_dataloader: DataLoader,
    eval_prompts: Sequence[str],
    tokenizer,
    scorer,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[str, float]:
    policy.eval()
    reference.eval()
    reference_device = next(reference.parameters()).device
    preference_correct = 0.0
    preference_total = 0.0
    with torch.no_grad():
        for batch in eval_dataloader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            chosen_response_starts = batch["chosen_response_starts"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            rejected_response_starts = batch["rejected_response_starts"].to(device)
            chosen_logp = sequence_logprob(policy, chosen_input_ids, chosen_attention_mask, chosen_response_starts)
            rejected_logp = sequence_logprob(policy, rejected_input_ids, rejected_attention_mask, rejected_response_starts)
            preference_correct += float((chosen_logp > rejected_logp).float().sum().item())
            preference_total += float(chosen_logp.numel())

        _, _, _, _, responses = generate_batch(
            policy=policy,
            tokenizer=tokenizer,
            prompts=eval_prompts,
            max_length=max_seq_len,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            device=device,
        )
        rm_scores = scorer(eval_prompts, responses)

        prompt_input_ids, prompt_attention_mask, full_input_ids, full_attention_mask, _ = generate_batch(
            policy=policy,
            tokenizer=tokenizer,
            prompts=eval_prompts,
            max_length=max_seq_len,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            device=device,
        )
        response_starts = prompt_attention_mask.sum(dim=1) + (prompt_input_ids.size(1) - prompt_attention_mask.sum(dim=1))
        policy_logprobs, response_mask = forward_response_logprobs(policy, full_input_ids, full_attention_mask, response_starts)
        with disable_adapters_for_reference(reference):
            ref_logprobs, _ = forward_response_logprobs(
                reference,
                full_input_ids.to(reference_device),
                full_attention_mask.to(reference_device),
                response_starts.to(reference_device),
            )
        kl_value = sampled_token_kl(policy_logprobs, ref_logprobs.to(device), response_mask)

    return {
        "rm_score_mean": float(rm_scores.float().mean().item()),
        "kl_to_reference": float(kl_value.item()),
        "heldout_preference_accuracy": preference_correct / max(preference_total, 1.0),
    }


def train_dpo_epoch(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: torch.device,
    grad_accum_steps: int = 1,
    logger=None,
    log_every: int = 50,
    eval_every: int = 1000,
    step_offset: int = 0,
    checkpoint_dir: Optional[str | Path] = None,
    checkpoint_model: Optional[torch.nn.Module] = None,
    checkpoint_tokenizer=None,
    checkpoint_every: int = 250,
    evaluation_callback=None,
) -> Dict[str, float]:
    policy.train()
    reference.eval()
    reference_device = next(reference.parameters()).device
    totals: Dict[str, float] = {
        "loss": 0.0,
        "preference_accuracy": 0.0,
        "implicit_margin": 0.0,
        "kl": 0.0,
        "grad_norm": 0.0,
        "steps": 0.0,
    }
    running: Dict[str, float] = {
        "loss": 0.0,
        "preference_accuracy": 0.0,
        "implicit_margin": 0.0,
        "kl": 0.0,
        "grad_norm": 0.0,
        "steps": 0.0,
    }
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(dataloader, start=1):
        chosen_input_ids = batch["chosen_input_ids"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        chosen_response_starts = batch["chosen_response_starts"].to(device)
        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)
        rejected_response_starts = batch["rejected_response_starts"].to(device)

        chosen_logp = sequence_logprob(policy, chosen_input_ids, chosen_attention_mask, chosen_response_starts)
        rejected_logp = sequence_logprob(policy, rejected_input_ids, rejected_attention_mask, rejected_response_starts)
        with torch.no_grad():
            with disable_adapters_for_reference(reference):
                chosen_ref = sequence_logprob(
                    reference,
                    chosen_input_ids.to(reference_device),
                    chosen_attention_mask.to(reference_device),
                    chosen_response_starts.to(reference_device),
                ).to(device)
                rejected_ref = sequence_logprob(
                    reference,
                    rejected_input_ids.to(reference_device),
                    rejected_attention_mask.to(reference_device),
                    rejected_response_starts.to(reference_device),
                ).to(device)

        loss, metrics = dpo_loss(chosen_logp - rejected_logp, chosen_ref - rejected_ref, beta=beta)
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()
        if batch_idx % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        totals["loss"] += float(loss.detach().item())
        totals["preference_accuracy"] += float(metrics["preference_accuracy"].item())
        totals["implicit_margin"] += float(metrics["implicit_margin"].item())
        totals["grad_norm"] += gradient_norm(policy.parameters())
        totals["steps"] += 1.0
        running["loss"] += float(loss.detach().item())
        running["preference_accuracy"] += float(metrics["preference_accuracy"].item())
        running["implicit_margin"] += float(metrics["implicit_margin"].item())
        running["grad_norm"] += gradient_norm(policy.parameters())
        running["steps"] += 1.0

        with torch.no_grad():
            chosen_token_logp, chosen_mask = forward_response_logprobs(
                policy, chosen_input_ids, chosen_attention_mask, chosen_response_starts
            )
            with disable_adapters_for_reference(reference):
                chosen_ref_token_logp, _ = forward_response_logprobs(
                    reference,
                    chosen_input_ids.to(reference_device),
                    chosen_attention_mask.to(reference_device),
                    chosen_response_starts.to(reference_device),
                )
            kl_value = float(sampled_token_kl(chosen_token_logp, chosen_ref_token_logp.to(device), chosen_mask).item())
            totals["kl"] += kl_value
            running["kl"] += kl_value

        global_step = step_offset + batch_idx
        if logger is not None and batch_idx % log_every == 0:
            denom = max(running["steps"], 1.0)
            payload = {
                "loss": running["loss"] / denom,
                "preference_accuracy": running["preference_accuracy"] / denom,
                "implicit_margin": running["implicit_margin"] / denom,
                "kl": running["kl"] / denom,
                "grad_norm": running["grad_norm"] / denom,
            }
            logger.log_metrics(global_step, payload)
            print(
                f"[dpo step {global_step}] "
                f"loss={payload['loss']:.4f} "
                f"margin={payload['implicit_margin']:.4f} "
                f"pref_acc={payload['preference_accuracy']:.4f} "
                f"kl={payload['kl']:.4f}"
            )
            running = {
                "loss": 0.0,
                "preference_accuracy": 0.0,
                "implicit_margin": 0.0,
                "kl": 0.0,
                "grad_norm": 0.0,
                "steps": 0.0,
            }

        if logger is not None and evaluation_callback is not None and batch_idx % eval_every == 0:
            eval_metrics = evaluation_callback()
            logger.log_metrics(global_step, {f"eval_{key}": value for key, value in eval_metrics.items()})
            print(
                f"[dpo eval {global_step}] "
                f"rm_score={eval_metrics['rm_score_mean']:.4f} "
                f"kl={eval_metrics['kl_to_reference']:.4f} "
                f"heldout_pref_acc={eval_metrics['heldout_preference_accuracy']:.4f}"
            )

        if (
            checkpoint_dir is not None
            and checkpoint_model is not None
            and checkpoint_tokenizer is not None
            and batch_idx % checkpoint_every == 0
        ):
            save_artifact(
                checkpoint_model,
                checkpoint_tokenizer,
                checkpoint_dir,
                extra_metadata={"task": "dpo_policy", "checkpoint_step": global_step},
            )

    if len(dataloader) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    steps = max(totals["steps"], 1.0)
    return {key: value / steps for key, value in totals.items() if key != "steps"}

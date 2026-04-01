from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch


@dataclass
class RolloutBatch:
    prompts: List[str]
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    full_input_ids: torch.Tensor
    full_attention_mask: torch.Tensor
    response_starts: torch.Tensor
    response_mask: torch.Tensor
    response_lengths: torch.Tensor
    responses: List[str]
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    rewards: torch.Tensor
    rm_rewards: torch.Tensor
    values: torch.Tensor | None = None
    group_ids: torch.Tensor | None = None

    def to(self, device: torch.device) -> "RolloutBatch":
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            return value

        return RolloutBatch(**{field: move(getattr(self, field)) for field in self.__dataclass_fields__})


def build_response_mask(attention_mask: torch.Tensor, response_starts: torch.Tensor, shifted: bool = True) -> torch.Tensor:
    seq_len = attention_mask.size(1)
    positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0).expand_as(attention_mask)
    mask = (positions >= response_starts.unsqueeze(1)) & attention_mask.bool()
    return mask[:, 1:] if shifted else mask


def gather_shifted_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    return log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)


def forward_response_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    token_logprobs = gather_shifted_logprobs(outputs.logits, input_ids)
    response_mask = build_response_mask(attention_mask, response_starts, shifted=True)
    return token_logprobs, response_mask


def forward_values(
    value_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = value_model(input_ids=input_ids, attention_mask=attention_mask)[:, 1:]
    response_mask = build_response_mask(attention_mask, response_starts, shifted=True)
    return values, response_mask


def _build_full_attention(prompt_attention_mask: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
    max_prompt_len = prompt_attention_mask.size(1)
    total_len = sequences.size(1)
    output = torch.zeros_like(sequences)
    prompt_lengths = prompt_attention_mask.sum(dim=1)
    for idx, prompt_len in enumerate(prompt_lengths.tolist()):
        left_pad = max_prompt_len - prompt_len
        output[idx, left_pad:total_len] = 1
    return output


def generate_batch(
    policy: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    prompt_batch = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_batch = {k: v.to(device) for k, v in prompt_batch.items()}
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    sequences = policy.generate(**prompt_batch, **generation_kwargs)
    full_attention_mask = _build_full_attention(prompt_batch["attention_mask"], sequences)
    response_starts = prompt_batch["attention_mask"].sum(dim=1) + (prompt_batch["input_ids"].size(1) - prompt_batch["attention_mask"].sum(dim=1))
    response_mask = build_response_mask(full_attention_mask, response_starts, shifted=True)
    response_lengths = response_mask.sum(dim=1)

    responses = []
    for idx, start in enumerate(response_starts.tolist()):
        response_ids = sequences[idx, start:]
        responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))
    return (
        prompt_batch["input_ids"],
        prompt_batch["attention_mask"],
        sequences,
        full_attention_mask,
        responses,
    )


def collect_rollout_batch(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: torch.device,
    reward_fn: Callable[[Sequence[str], List[str]], torch.Tensor],
    value_model: torch.nn.Module | None = None,
    group_ids: torch.Tensor | None = None,
) -> RolloutBatch:
    prompt_input_ids, prompt_attention_mask, full_input_ids, full_attention_mask, responses = generate_batch(
        policy=policy,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        device=device,
    )
    response_starts = prompt_attention_mask.sum(dim=1) + (prompt_input_ids.size(1) - prompt_attention_mask.sum(dim=1))
    old_logprobs, response_mask = forward_response_logprobs(policy, full_input_ids, full_attention_mask, response_starts)
    with torch.no_grad():
        ref_logprobs, _ = forward_response_logprobs(reference, full_input_ids, full_attention_mask, response_starts)
        rewards = reward_fn(prompts, responses).to(device)
        values = None
        if value_model is not None:
            values, _ = forward_values(value_model, full_input_ids, full_attention_mask, response_starts)

    return RolloutBatch(
        prompts=list(prompts),
        prompt_input_ids=prompt_input_ids.cpu(),
        prompt_attention_mask=prompt_attention_mask.cpu(),
        full_input_ids=full_input_ids.cpu(),
        full_attention_mask=full_attention_mask.cpu(),
        response_starts=response_starts.cpu(),
        response_mask=response_mask.cpu(),
        response_lengths=response_mask.sum(dim=1).cpu(),
        responses=responses,
        old_logprobs=old_logprobs.detach().cpu(),
        ref_logprobs=ref_logprobs.detach().cpu(),
        rewards=rewards.detach().cpu(),
        rm_rewards=rewards.detach().cpu(),
        values=values.detach().cpu() if values is not None else None,
        group_ids=group_ids.cpu() if group_ids is not None else None,
    )

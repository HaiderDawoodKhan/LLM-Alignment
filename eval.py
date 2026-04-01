from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from alignment.kl import sampled_token_kl
from alignment.rollout import forward_response_logprobs, generate_batch
from config import AppConfig, default_config
from data.gsm8k import extract_answer, extract_gold_answer, format_gsm8k_prompt, load_gsm8k
from data.hh_rlhf import build_preference_datasets
from model.reward_model import score_sequences
from model.utils import get_torch_device
from train_helpers import default_device, load_policy_checkpoint, load_reward_checkpoint


def build_reward_scorer(reward_model, reward_tokenizer, device: torch.device, max_length: int):
    def score(prompts: Sequence[str], responses: Sequence[str]) -> torch.Tensor:
        pair_text = [prompt + response for prompt, response in zip(prompts, responses)]
        tokenized = reward_tokenizer(
            list(pair_text),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            return score_sequences(reward_model, tokenized["input_ids"], tokenized["attention_mask"]).detach().cpu()

    return score


def generate_responses(policy, tokenizer, prompts: Sequence[str], config: AppConfig, device: torch.device) -> List[str]:
    with torch.no_grad():
        _, _, _, _, responses = generate_batch(
            policy=policy,
            tokenizer=tokenizer,
            prompts=prompts,
            max_length=config.data.max_seq_len,
            max_new_tokens=config.data.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            device=device,
        )
    return responses


def compute_kl_to_reference(policy, reference, tokenizer, prompts: Sequence[str], config: AppConfig, device: torch.device) -> float:
    with torch.no_grad():
        prompt_input_ids, prompt_attention_mask, full_input_ids, full_attention_mask, _ = generate_batch(
            policy=policy,
            tokenizer=tokenizer,
            prompts=prompts,
            max_length=config.data.max_seq_len,
            max_new_tokens=config.data.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            device=device,
        )
        response_starts = prompt_attention_mask.sum(dim=1) + (prompt_input_ids.size(1) - prompt_attention_mask.sum(dim=1))
        policy_logprobs, response_mask = forward_response_logprobs(policy, full_input_ids, full_attention_mask, response_starts)
        ref_logprobs, _ = forward_response_logprobs(reference, full_input_ids, full_attention_mask, response_starts)
        return float(sampled_token_kl(policy_logprobs, ref_logprobs, response_mask).item())


def rm_win_rate_vs_sft(
    sft_model,
    candidate_model,
    policy_tokenizer,
    scorer,
    prompts: Sequence[str],
    config: AppConfig,
    device: torch.device,
) -> Dict[str, float]:
    sft_responses = generate_responses(sft_model, policy_tokenizer, prompts, config, device)
    candidate_responses = generate_responses(candidate_model, policy_tokenizer, prompts, config, device)
    sft_scores = scorer(prompts, sft_responses)
    candidate_scores = scorer(prompts, candidate_responses)
    wins = (candidate_scores > sft_scores).float()
    return {
        "rm_win_rate_vs_sft": float(wins.mean().item()),
        "candidate_score_mean": float(candidate_scores.mean().item()),
        "sft_score_mean": float(sft_scores.mean().item()),
    }


def build_sample_response_table(
    models: Dict[str, torch.nn.Module],
    tokenizer,
    scorer,
    prompts: Sequence[str],
    config: AppConfig,
    device: torch.device,
) -> List[dict]:
    table = []
    for prompt in prompts:
        row = {"prompt": prompt}
        for name, model in models.items():
            response = generate_responses(model, tokenizer, [prompt], config, device)[0]
            score = float(scorer([prompt], [response])[0].item())
            row[name] = {"response": response, "rm_score": score}
        table.append(row)
    return table


def evaluate_gsm8k_pass_at_one(policy, tokenizer, config: AppConfig, device: torch.device, max_examples: int = 200) -> Dict[str, float]:
    dataset = load_gsm8k(config.data)[config.data.eval_split]
    examples = dataset.select(range(min(max_examples, len(dataset))))
    prompts = [format_gsm8k_prompt(row["question"]) for row in examples]
    gold_answers = [extract_gold_answer(row["answer"]) for row in examples]
    responses = []
    with torch.no_grad():
        for idx in range(0, len(prompts), config.grpo.prompts_per_step):
            responses.extend(generate_responses(policy, tokenizer, prompts[idx : idx + config.grpo.prompts_per_step], config, device))
    parsed = [extract_answer(response) for response in responses]
    correct = [1.0 if pred is not None and pred == gold else 0.0 for pred, gold in zip(parsed, gold_answers)]
    parseable = [1.0 if pred is not None else 0.0 for pred in parsed]
    return {
        "gsm8k_pass_at_one": sum(correct) / max(len(correct), 1),
        "gsm8k_parseable_fraction": sum(parseable) / max(len(parseable), 1),
    }


def read_resource_summary(run_dir: str | Path) -> Dict[str, float]:
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    peak_vram = 0.0
    step_times = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            peak_vram = max(peak_vram, float(row.get("peak_allocated_gb", 0.0)))
            if "step_time_seconds" in row:
                step_times.append(float(row["step_time_seconds"]))
    return {
        "peak_vram_gb": peak_vram,
        "time_per_update_seconds": sum(step_times) / max(len(step_times), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate aligned models against SFT.")
    parser.add_argument("--models", nargs="+", required=True, help="Checkpoint names under checkpoints/")
    args = parser.parse_args()

    config = default_config(run_name="eval")
    device = default_device(config)
    datasets = build_preference_datasets(config.data)
    prompts = [datasets["prompt_eval"][idx]["prompt"] for idx in range(min(config.evaluation.num_prompts, len(datasets["prompt_eval"])))]

    reward_model, reward_tokenizer = load_reward_checkpoint(config.checkpoint_dir("reward_model"), config, trainable=False)
    reward_model.to(device)
    scorer = build_reward_scorer(reward_model, reward_tokenizer, device, config.data.max_seq_len)

    sft_model, policy_tokenizer = load_policy_checkpoint(config.checkpoint_dir("policy_sft"), config, trainable=False)
    sft_model.to(device)
    reference_model, _ = load_policy_checkpoint(config.checkpoint_dir("policy_ref"), config, trainable=False)
    reference_model.to(device)

    results = {}
    loaded_models = {"policy_sft": sft_model}
    for name in args.models:
        model, _ = load_policy_checkpoint(config.checkpoint_dir(name), config, trainable=False)
        model.to(device)
        loaded_models[name] = model
        if name == "policy_sft":
            continue
        metrics = rm_win_rate_vs_sft(sft_model, model, policy_tokenizer, scorer, prompts, config, device)
        metrics["kl_to_reference"] = compute_kl_to_reference(model, reference_model, policy_tokenizer, prompts, config, device)
        if name == "rlvr_policy":
            metrics.update(evaluate_gsm8k_pass_at_one(model, policy_tokenizer, config, device))
        metrics.update(read_resource_summary(config.log_dir(name)))
        results[name] = metrics

    sample_prompts = prompts[: config.evaluation.sample_prompts]
    sample_table = build_sample_response_table(loaded_models, policy_tokenizer, scorer, sample_prompts, config, device)
    print(json.dumps({"metrics": results, "sample_table": sample_table}, indent=2))


if __name__ == "__main__":
    main()

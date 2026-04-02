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
from model.lora import disable_adapters_for_reference
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


def score_responses(
    prompts: Sequence[str],
    responses: Sequence[str],
    scorer,
) -> torch.Tensor:
    return scorer(prompts, responses)


def generate_responses(
    policy,
    tokenizer,
    prompts: Sequence[str],
    config: AppConfig,
    device: torch.device,
    max_new_tokens: int | None = None,
) -> List[str]:
    with torch.no_grad():
        _, _, _, _, responses = generate_batch(
            policy=policy,
            tokenizer=tokenizer,
            prompts=prompts,
            max_length=config.data.max_seq_len,
            max_new_tokens=max_new_tokens or config.data.max_new_tokens,
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
        reference_device = next(reference.parameters()).device
        with disable_adapters_for_reference(reference):
            ref_logprobs, _ = forward_response_logprobs(
                reference,
                full_input_ids.to(reference_device),
                full_attention_mask.to(reference_device),
                response_starts.to(reference_device),
            )
        return float(sampled_token_kl(policy_logprobs, ref_logprobs.to(device), response_mask).item())


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


def rm_win_rate_from_cached_scores(
    sft_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
) -> Dict[str, float]:
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


def select_disagreement_indices(
    score_cache: Dict[str, torch.Tensor],
    num_rows: int,
) -> List[int]:
    if not score_cache:
        return list(range(num_rows))
    names = list(score_cache.keys())
    num_prompts = score_cache[names[0]].numel()
    ranked = []
    for idx in range(num_prompts):
        values = torch.tensor([float(score_cache[name][idx].item()) for name in names], dtype=torch.float32)
        spread = float((values.max() - values.min()).item())
        ranked.append((spread, idx))
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = [idx for _, idx in ranked[:num_rows]]
    if len(chosen) < num_rows:
        seen = set(chosen)
        for idx in range(num_prompts):
            if idx not in seen:
                chosen.append(idx)
            if len(chosen) >= num_rows:
                break
    return chosen[:num_rows]


def build_cached_sample_response_table(
    prompts: Sequence[str],
    response_cache: Dict[str, List[str]],
    score_cache: Dict[str, torch.Tensor],
    sample_indices: Sequence[int],
) -> List[dict]:
    table = []
    model_names = list(response_cache.keys())
    for idx in sample_indices:
        row = {"prompt": prompts[idx]}
        for name in model_names:
            row[name] = {
                "response": response_cache[name][idx],
                "rm_score": float(score_cache[name][idx].item()),
            }
        table.append(row)
    return table


def evaluate_gsm8k_pass_at_one(policy, tokenizer, config: AppConfig, device: torch.device, max_examples: int = 200) -> Dict[str, float]:
    dataset = load_gsm8k(config.data)[config.data.eval_split]
    examples = dataset.select(range(min(max_examples, len(dataset))))
    prompts = [format_gsm8k_prompt(row["question"], tokenizer=tokenizer, max_question_tokens=200) for row in examples]
    gold_answers = [extract_gold_answer(row["answer"]) for row in examples]
    responses = []
    with torch.no_grad():
        for idx in range(0, len(prompts), config.rlvr.prompts_per_step):
            responses.extend(
                generate_responses(
                    policy,
                    tokenizer,
                    prompts[idx : idx + config.rlvr.prompts_per_step],
                    config,
                    device,
                    max_new_tokens=config.data.rlvr_max_new_tokens,
                )
            )
    parsed = [extract_answer(response) for response in responses]
    correct = [1.0 if pred is not None and pred == gold else 0.0 for pred, gold in zip(parsed, gold_answers)]
    parseable = [1.0 if pred is not None else 0.0 for pred in parsed]
    response_lengths = [len(tokenizer([response], add_special_tokens=False)["input_ids"][0]) for response in responses]
    return {
        "gsm8k_pass_at_one": sum(correct) / max(len(correct), 1),
        "gsm8k_parseable_fraction": sum(parseable) / max(len(parseable), 1),
        "gsm8k_response_length_mean": sum(response_lengths) / max(len(response_lengths), 1),
    }


def build_gsm8k_sample_table(
    policy,
    tokenizer,
    config: AppConfig,
    device: torch.device,
    max_examples: int = 5,
) -> List[dict]:
    dataset = load_gsm8k(config.data)[config.data.eval_split]
    examples = dataset.select(range(min(max_examples, len(dataset))))
    table = []
    for row in examples:
        prompt = format_gsm8k_prompt(row["question"], tokenizer=tokenizer, max_question_tokens=200)
        response = generate_responses(
            policy,
            tokenizer,
            [prompt],
            config,
            device,
            max_new_tokens=config.data.rlvr_max_new_tokens,
        )[0]
        predicted_answer = extract_answer(response)
        gold_answer = extract_gold_answer(row["answer"])
        table.append(
            {
                "question": row["question"],
                "generated_solution": response,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "correct": bool(predicted_answer is not None and predicted_answer == gold_answer),
            }
        )
    return table


def read_resource_summary(run_dir: str | Path) -> Dict[str, float]:
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    peak_vram = 0.0
    step_times = []
    total_training_time = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            peak_vram = max(peak_vram, float(row.get("peak_allocated_gb", 0.0)))
            if "step_time_seconds" in row:
                step_times.append(float(row["step_time_seconds"]))
            if "wall_time_seconds" in row:
                total_training_time = float(row["wall_time_seconds"])
    return {
        "peak_vram_gb": peak_vram,
        "time_per_update_seconds": sum(step_times) / max(len(step_times), 1),
        "total_training_time_seconds": total_training_time if total_training_time is not None else 0.0,
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
    response_cache = {
        "policy_sft": generate_responses(sft_model, policy_tokenizer, prompts, config, device),
    }
    score_cache = {
        "policy_sft": score_responses(prompts, response_cache["policy_sft"], scorer),
    }
    for name in args.models:
        model, _ = load_policy_checkpoint(config.checkpoint_dir(name), config, trainable=False)
        model.to(device)
        loaded_models[name] = model
        response_cache[name] = generate_responses(model, policy_tokenizer, prompts, config, device)
        score_cache[name] = score_responses(prompts, response_cache[name], scorer)
        if name == "policy_sft":
            continue
        metrics = rm_win_rate_from_cached_scores(score_cache["policy_sft"], score_cache[name])
        metrics["kl_to_reference"] = compute_kl_to_reference(model, reference_model, policy_tokenizer, prompts, config, device)
        if name == "rlvr_policy":
            metrics.update(evaluate_gsm8k_pass_at_one(model, policy_tokenizer, config, device))
            metrics["gsm8k_sample_table"] = build_gsm8k_sample_table(model, policy_tokenizer, config, device)
        metrics.update(read_resource_summary(config.log_dir(name)))
        results[name] = metrics

    sample_indices = select_disagreement_indices(score_cache, config.evaluation.sample_prompts)
    sample_table = build_cached_sample_response_table(prompts, response_cache, score_cache, sample_indices)
    print(json.dumps({"metrics": results, "sample_table": sample_table}, indent=2))


if __name__ == "__main__":
    main()

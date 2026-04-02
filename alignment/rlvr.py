from __future__ import annotations

from typing import Iterable, List, Sequence

import torch

from data.gsm8k import extract_answer, extract_gold_answer


def compute_verifiable_rewards(predictions: Sequence[str], gold_answers: Sequence[str]) -> torch.Tensor:
    rewards = []
    for prediction, gold in zip(predictions, gold_answers):
        pred_answer = extract_answer(prediction)
        rewards.append(1.0 if pred_answer is not None and pred_answer == gold else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)


def gsm8k_reward_fn(gold_answers: Sequence[str]):
    def reward_fn(_prompts, responses: List[str]) -> torch.Tensor:
        return compute_verifiable_rewards(responses, gold_answers)

    return reward_fn


def _make_obviously_wrong_string(gold_answer: str | None, idx: int) -> str:
    if gold_answer is None:
        return f"I do not know the answer #{idx}."
    try:
        value = float(gold_answer)
    except (TypeError, ValueError):
        return f"The answer is definitely not {gold_answer}."
    wrong_value = value + idx + 1
    if wrong_value.is_integer():
        wrong_text = str(int(wrong_value))
    else:
        wrong_text = str(wrong_value)
    return f"The answer is {wrong_text}"


def verify_gsm8k_verifier(dataset, num_examples: int = 20) -> dict[str, object]:
    subset = dataset.select(range(min(num_examples, len(dataset))))
    gold_solutions = [row["answer"] for row in subset]
    gold_answers = [extract_gold_answer(solution) for solution in gold_solutions]
    gold_rewards = compute_verifiable_rewards(gold_solutions, gold_answers)

    wrong_strings = [
        _make_obviously_wrong_string(gold_answer, idx)
        for idx, gold_answer in enumerate(gold_answers)
    ]
    wrong_rewards = compute_verifiable_rewards(wrong_strings, gold_answers)

    return {
        "num_examples": len(gold_answers),
        "gold_reward_mean": float(gold_rewards.mean().item()) if len(gold_rewards) else 0.0,
        "gold_all_correct": bool(torch.all(gold_rewards == 1.0).item()) if len(gold_rewards) else False,
        "wrong_reward_mean": float(wrong_rewards.mean().item()) if len(wrong_rewards) else 0.0,
        "wrong_all_zero": bool(torch.all(wrong_rewards == 0.0).item()) if len(wrong_rewards) else False,
        "gold_answers": gold_answers,
        "wrong_strings": wrong_strings,
    }

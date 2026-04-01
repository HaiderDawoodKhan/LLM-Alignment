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

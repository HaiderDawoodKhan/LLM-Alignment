from __future__ import annotations

import math
import unittest

import torch

from alignment.dpo import sequence_logprob
from model.reward_model import score_sequences
from tests.helpers import FakeRewardModel, FixedLogitsModel


class RewardAndDPOTests(unittest.TestCase):
    def test_reward_scoring_uses_last_real_token(self) -> None:
        model = FakeRewardModel()
        input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        scores = score_sequences(model, input_ids, attention_mask)
        self.assertTrue(torch.equal(scores, torch.tensor([2.0, 5.0])))

    def test_sequence_logprob_excludes_prompt_and_padding(self) -> None:
        logits = torch.full((1, 5, 8), -100.0)
        logits[0, 0, 2] = 0.0
        logits[0, 1, 3] = 0.0
        logits[0, 2, 4] = 0.0
        logits[0, 3, 0] = 0.0
        model = FixedLogitsModel(logits)
        input_ids = torch.tensor([[1, 2, 3, 4, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
        response_start = torch.tensor([2])
        logp = sequence_logprob(model, input_ids, attention_mask, response_start)
        self.assertAlmostEqual(logp.item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from alignment.rollout import build_response_mask, forward_response_logprobs
from tests.helpers import FixedLogitsModel, TinyPolicy


class SmokeTests(unittest.TestCase):
    def test_forward_response_logprobs_smoke(self) -> None:
        logits = torch.zeros((1, 4, 16))
        logits[0, 0, 2] = 5.0
        logits[0, 1, 3] = 5.0
        logits[0, 2, 4] = 5.0
        model = FixedLogitsModel(logits)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.tensor([[1, 1, 1, 1]])
        response_starts = torch.tensor([2])
        logprobs, mask = forward_response_logprobs(model, input_ids, attention_mask, response_starts)
        self.assertEqual(logprobs.shape, (1, 3))
        self.assertTrue(torch.equal(mask, torch.tensor([[False, True, True]])))

    def test_tiny_policy_forward_with_labels(self) -> None:
        model = TinyPolicy()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 3, 4]]),
        }
        outputs = model(**batch)
        self.assertTrue(hasattr(outputs, "loss"))
        self.assertGreaterEqual(outputs.loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

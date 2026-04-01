from __future__ import annotations

import unittest

import torch

from alignment.grpo import broadcast_group_advantages, compute_group_relative_advantages
from alignment.ppo import compute_gae, ppo_clipping_reference_value, ppo_sanity_ratio_test


class PPOAndGRPOTests(unittest.TestCase):
    def test_compute_gae_targets_match_handworked_returns(self) -> None:
        rewards = torch.tensor([[1.0, 1.0, 1.0]])
        values = torch.zeros_like(rewards)
        mask = torch.tensor([[True, True, True]])
        _, targets = compute_gae(rewards, values, mask, gamma=1.0, gae_lambda=1.0)
        self.assertTrue(torch.allclose(targets, torch.tensor([[3.0, 2.0, 1.0]])))

    def test_ratio_test_detects_identity(self) -> None:
        old = torch.tensor([[0.1, 0.2]])
        self.assertTrue(ppo_sanity_ratio_test(old, old.clone()))

    def test_clipping_reference_value_matches_expected(self) -> None:
        self.assertEqual(ppo_clipping_reference_value(1.5, 1.0, 0.2), 1.2)

    def test_grpo_group_advantage_and_broadcast(self) -> None:
        rewards = torch.tensor([1.0, 3.0, 2.0, 2.0])
        group_ids = torch.tensor([0, 0, 1, 1])
        advantages, degenerate = compute_group_relative_advantages(rewards, group_ids)
        self.assertTrue(torch.allclose(advantages, torch.tensor([-1.0, 1.0, 0.0, 0.0])))
        self.assertAlmostEqual(degenerate, 0.5)

        response_mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)
        broadcast = broadcast_group_advantages(advantages, response_mask)
        self.assertEqual(broadcast.shape, response_mask.shape)
        self.assertEqual(float(broadcast[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()

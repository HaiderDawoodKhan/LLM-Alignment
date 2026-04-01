from __future__ import annotations

import unittest

from data.collators import DPOCollator, RMCollator, SFTCollator
from tests.helpers import DummyTokenizer


class CollatorTests(unittest.TestCase):
    def test_sft_masks_prompt_tokens(self) -> None:
        tokenizer = DummyTokenizer(padding_side="left")
        collator = SFTCollator(tokenizer=tokenizer, max_length=32)
        batch = collator([{"prompt": "AB", "response": "CD"}])
        labels = batch["labels"][0].tolist()
        self.assertEqual(labels[:2], [-100, -100])
        self.assertNotEqual(labels[-1], -100)

    def test_rm_uses_right_padding_and_last_indices(self) -> None:
        tokenizer = DummyTokenizer(padding_side="right")
        collator = RMCollator(tokenizer=tokenizer, max_length=32)
        batch = collator(
            [
                {"prompt": "A", "chosen": "B", "rejected": "C"},
                {"prompt": "AA", "chosen": "BB", "rejected": "CC"},
            ]
        )
        self.assertEqual(batch["chosen_attention_mask"][0, -1].item(), 0)
        self.assertEqual(batch["chosen_last_indices"].tolist(), [1, 3])

    def test_dpo_tracks_response_starts(self) -> None:
        tokenizer = DummyTokenizer(padding_side="left")
        collator = DPOCollator(tokenizer=tokenizer, max_length=32)
        batch = collator([{"prompt": "AB", "chosen": "CD", "rejected": "EF"}])
        self.assertEqual(batch["chosen_response_starts"].item(), 2)
        self.assertEqual(batch["rejected_response_starts"].item(), 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from data.parsing import split_prompt_response


class ParsingTests(unittest.TestCase):
    def test_split_prompt_response_uses_common_prefix(self) -> None:
        chosen = "Human: Hi\nAssistant: Helpful answer"
        rejected = "Human: Hi\nAssistant: Harmful answer"
        parsed = split_prompt_response(chosen, rejected)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["prompt"], "Human: Hi\nAssistant:")
        self.assertEqual(parsed["chosen"], "Helpful answer")
        self.assertEqual(parsed["rejected"], "Harmful answer")

    def test_split_prompt_response_rejects_missing_prompt(self) -> None:
        self.assertIsNone(split_prompt_response("abc", "xyz"))


if __name__ == "__main__":
    unittest.main()

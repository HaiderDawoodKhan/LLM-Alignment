from __future__ import annotations

import unittest

from data.gsm8k import extract_answer
from model.lora import require_peft
from model.utils import get_torch_device, require_hf_access


class GSM8KAndRuntimeTests(unittest.TestCase):
    def test_extract_answer_handles_multiple_formats(self) -> None:
        gold_cases = [
            "The answer is 42",
            "#### 42",
            "\\boxed{42}",
            "Final answer: -1,234",
            "answer is 3.5",
        ] * 4
        expected = ["42", "42", "42", "-1234", "3.5"] * 4
        for text, target in zip(gold_cases, expected):
            self.assertEqual(extract_answer(text), target)

    def test_extract_answer_rejects_non_numeric_cases(self) -> None:
        bad_cases = [
            "I do not know",
            "The answer is many",
            "#### none",
            "\\boxed{text}",
            "",
        ] * 4
        for text in bad_cases:
            self.assertIsNone(extract_answer(text))

    def test_runtime_guards(self) -> None:
        self.assertEqual(get_torch_device("auto").type, "cpu")
        error = require_hf_access(RuntimeError("nope"), "meta-llama/Llama-3.2-1B-Instruct")
        self.assertIn("meta-llama/Llama-3.2-1B-Instruct", str(error))
        with self.assertRaises(RuntimeError):
            require_peft()


if __name__ == "__main__":
    unittest.main()

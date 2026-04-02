from __future__ import annotations

import unittest

from alignment.rlvr import verify_gsm8k_verifier
from data.gsm8k import extract_answer, format_gsm8k_prompt
from model.lora import require_peft
from model.utils import get_torch_device, require_hf_access
from tests.helpers import DummyTokenizer


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

    def test_gsm8k_prompt_formatting_truncates_question_tokens(self) -> None:
        tokenizer = DummyTokenizer()
        question = "abcdefghij" * 50
        prompt = format_gsm8k_prompt(question, tokenizer=tokenizer, max_question_tokens=20)
        self.assertIn("Solve the following math problem step by step.", prompt)
        self.assertIn("At the end, write your final answer as a single number.", prompt)
        self.assertIn("Problem:", prompt)
        self.assertIn("Solution:", prompt)

    def test_rlvr_verifier_check_matches_gold_and_wrong_cases(self) -> None:
        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def select(self, indices):
                return FakeDataset([self.rows[idx] for idx in indices])

            def __iter__(self):
                return iter(self.rows)

        dataset = FakeDataset(
            [
                {"question": f"Question {idx}", "answer": f"Reasoning...\n#### {idx}"}
                for idx in range(20)
            ]
        )
        summary = verify_gsm8k_verifier(dataset, num_examples=20)
        self.assertTrue(summary["gold_all_correct"])
        self.assertTrue(summary["wrong_all_zero"])
        self.assertEqual(summary["num_examples"], 20)

    def test_runtime_guards(self) -> None:
        self.assertEqual(get_torch_device("auto").type, "cpu")
        error = require_hf_access(RuntimeError("nope"), "meta-llama/Llama-3.2-1B-Instruct")
        self.assertIn("meta-llama/Llama-3.2-1B-Instruct", str(error))
        with self.assertRaises(RuntimeError):
            require_peft()


if __name__ == "__main__":
    unittest.main()

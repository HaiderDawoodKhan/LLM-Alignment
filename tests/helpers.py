from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn


class DummyTokenizer:
    def __init__(self, padding_side: str = "left") -> None:
        self.padding_side = padding_side
        self.pad_token_id = 0
        self.eos_token_id = 99
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def _encode(self, text: str, max_length: int | None = None) -> List[int]:
        tokens = [ord(ch) % 50 + 1 for ch in text]
        if max_length is not None:
            tokens = tokens[:max_length]
        return tokens

    def __call__(
        self,
        texts: Sequence[str],
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True,
    ):
        encoded = [self._encode(text, max_length=max_length if truncation else None) for text in texts]
        if not padding:
            output = {"input_ids": encoded}
            if return_attention_mask:
                output["attention_mask"] = [[1] * len(ids) for ids in encoded]
            return output

        width = max(len(ids) for ids in encoded) if encoded else 0
        padded_ids = []
        masks = []
        for ids in encoded:
            pad_len = width - len(ids)
            if self.padding_side == "left":
                padded = [self.pad_token_id] * pad_len + ids
                mask = [0] * pad_len + [1] * len(ids)
            else:
                padded = ids + [self.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            padded_ids.append(padded)
            masks.append(mask)
        output = {"input_ids": torch.tensor(padded_ids, dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor(masks, dtype=torch.long)
        return output

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        chars = []
        for token in ids:
            if token in {self.pad_token_id, self.eos_token_id} and skip_special_tokens:
                continue
            chars.append(chr((token - 1) % 26 + 97))
        return "".join(chars)


class FixedLogitsModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        batch = input_ids.size(0)
        logits = self._logits
        if logits.size(0) != batch:
            logits = logits.expand(batch, -1, -1).clone()
        return type("Output", (), {"logits": logits})


class TinyPolicy(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels=None):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        output = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            output["loss"] = loss
        return type("Output", (), output)


class FakeBackbone(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, return_dict: bool = True):
        hidden = input_ids.unsqueeze(-1).float()
        return type("Output", (), {"last_hidden_state": hidden})


class FakeRewardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.score = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.score.weight.fill_(1.0)

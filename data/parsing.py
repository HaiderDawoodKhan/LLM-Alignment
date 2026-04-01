from __future__ import annotations

from typing import Dict, Optional


def longest_common_prefix(a: str, b: str) -> str:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return a[:index]


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def split_prompt_response(chosen_text: str, rejected_text: str) -> Optional[Dict[str, str]]:
    chosen_norm = normalize_text(chosen_text)
    rejected_norm = normalize_text(rejected_text)
    prompt = longest_common_prefix(chosen_norm, rejected_norm)
    cut = max(prompt.rfind("\n"), prompt.rfind(" "), prompt.rfind("\t"))
    if cut != -1 and cut < len(prompt):
        prompt = prompt[:cut]
    prompt = prompt.rstrip()
    if not prompt:
        return None

    chosen_resp = chosen_norm[len(prompt) :].strip()
    rejected_resp = rejected_norm[len(prompt) :].strip()
    if not chosen_resp or not rejected_resp:
        return None
    return {
        "prompt": prompt,
        "chosen": chosen_resp,
        "rejected": rejected_resp,
    }

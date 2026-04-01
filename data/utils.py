from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence


def chunked(items: Sequence, size: int) -> Iterator[Sequence]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def limit_rows(rows: Iterable[dict], limit: int | None) -> List[dict]:
    if limit is None:
        return list(rows)
    output = []
    for row in rows:
        output.append(row)
        if len(output) >= limit:
            break
    return output

from __future__ import annotations

import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Vocabulary:
    def __init__(self, max_size: int = 30000, min_freq: int = 2) -> None:
        self.max_size = max_size
        self.min_freq = min_freq
        self.token_to_id = {"<pad>": 0, "<unk>": 1}

    def fit(self, texts: list[str]) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        for token, count in counts.most_common(self.max_size - len(self.token_to_id)):
            if count < self.min_freq:
                break
            self.token_to_id[token] = len(self.token_to_id)
        return self

    def encode(self, text: str, max_tokens: int) -> list[int]:
        ids = [self.token_to_id.get(token, 1) for token in tokenize(text)[:max_tokens]]
        return ids or [1]

    def __len__(self) -> int:
        return len(self.token_to_id)

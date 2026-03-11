from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol, Sequence


class TokenEstimator(Protocol):
    def count(self, text: str) -> int:
        ...


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class ModelAdapter(Protocol):
    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        ...


class SimpleTokenEstimator:
    """Portable fallback estimator when provider tokenizers are unavailable."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        word_estimate = max(1, int(len(text.split()) * 1.35))
        char_estimate = max(1, math.ceil(len(text) / 4))
        return max(word_estimate, char_estimate)


class HashingEmbedder:
    """
    Stateless local embedder using signed hashing.

    This avoids heavy runtime dependencies while still giving stable vector
    similarity for memory retrieval.
    """

    _token_pattern = re.compile(r"[A-Za-z0-9_\\-]{3,}")

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be >= 32")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = [t.lower() for t in self._token_pattern.findall(text or "")]
        if not tokens:
            return vec

        counts = Counter(tokens)
        for token, freq in counts.items():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            hashed = int.from_bytes(digest, byteorder="big", signed=False)
            idx = hashed % self.dimensions
            sign = 1.0 if ((hashed >> 9) & 1) == 1 else -1.0
            weight = 1.0 + math.log1p(freq)
            vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


class EchoModelAdapter:
    """
    Minimal demo model adapter.

    It echoes the latest user message, which is enough to verify orchestration
    and context budgeting in tests/examples.
    """

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        latest_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                latest_user = msg.get("content", "")
                break
        if not latest_user:
            return "No user message was provided."
        return f"Echo response based on: {latest_user[:120]}"

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_ANCHOR_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\\-]{2,}")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "any",
    "are",
    "because",
    "been",
    "between",
    "could",
    "does",
    "from",
    "have",
    "into",
    "just",
    "like",
    "more",
    "much",
    "need",
    "other",
    "over",
    "some",
    "that",
    "them",
    "then",
    "there",
    "they",
    "this",
    "those",
    "time",
    "want",
    "with",
    "would",
    "your",
}


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    ax = list(a)
    bx = list(b)
    if len(ax) != len(bx):
        raise ValueError("vectors must have equal length")
    if not ax:
        return 0.0
    dot = sum(x * y for x, y in zip(ax, bx))
    norm_a = math.sqrt(sum(x * x for x in ax))
    norm_b = math.sqrt(sum(y * y for y in bx))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_anchors(text: str, max_anchors: int = 24) -> list[str]:
    tokens = [
        t.lower()
        for t in _ANCHOR_PATTERN.findall(text or "")
        if len(t) > 2 and not t.isdigit()
    ]
    filtered = [t for t in tokens if t not in _STOPWORDS]
    if not filtered:
        return []
    counts = Counter(filtered)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [anchor for anchor, _ in ranked[:max_anchors]]


def summarize_text(text: str, max_chars: int = 320) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean

    sentences = _SENTENCE_BREAK.split(clean)
    if not sentences:
        return clean[: max_chars - 3] + "..."

    selected: list[str] = []
    char_budget = max_chars
    for sentence in sentences:
        if not sentence:
            continue
        next_len = len(sentence) + (1 if selected else 0)
        if next_len > char_budget and selected:
            break
        selected.append(sentence)
        char_budget -= next_len
        if char_budget <= max_chars * 0.25:
            break

    if not selected:
        return clean[: max_chars - 3] + "..."

    summary = " ".join(selected).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


def merge_synopsis(existing: str, incoming: str, max_chars: int = 420) -> str:
    parts: list[str] = []
    for raw in [existing, incoming]:
        for sentence in _SENTENCE_BREAK.split(" ".join(raw.split())):
            sentence = sentence.strip()
            if sentence and sentence not in parts:
                parts.append(sentence)
    merged = " ".join(parts).strip()
    if len(merged) <= max_chars:
        return merged
    return merged[: max_chars - 3] + "..."


def mmr_rank(
    vectors: list[list[float]],
    base_scores: list[float],
    max_items: int,
    lambda_weight: float = 0.75,
) -> list[int]:
    if len(vectors) != len(base_scores):
        raise ValueError("vectors and base_scores must be same length")
    if not vectors or max_items <= 0:
        return []

    selected: list[int] = []
    remaining = set(range(len(vectors)))

    while remaining and len(selected) < max_items:
        best_idx = -1
        best_score = float("-inf")
        for idx in list(remaining):
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    cosine_similarity(vectors[idx], vectors[chosen]) for chosen in selected
                )
            score = lambda_weight * base_scores[idx] - (1.0 - lambda_weight) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected

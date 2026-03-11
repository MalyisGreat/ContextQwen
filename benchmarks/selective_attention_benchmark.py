from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

from memory_orb import MemoryOrbEngine, MemoryOrbEngineConfig, SimpleTokenEstimator

SEGMENT_ID_RE = re.compile(r"\bseg\d{3}\b")
TARGET_ALIASES: dict[str, list[str]] = {
    "oauth": ["token exchange", "identity handshake", "callback signature", "auth redirect flow"],
    "replication": ["replica lag", "wal shipping", "follower sync", "standby drift"],
    "ledger": ["account journal", "reconciliation log", "balance register", "posting stream"],
    "latency": ["tail response time", "p95 delay", "request lag", "slow path"],
    "checkpoint": ["state snapshot", "flush barrier", "recovery marker", "commit boundary"],
    "billing": ["invoice pipeline", "charge settlement", "usage metering", "payment posting"],
    "compliance": ["control evidence", "audit trail", "policy attestation", "risk register"],
    "rollback": ["revert release", "backout plan", "deployment reversal", "safe fallback"],
}


@dataclass(slots=True)
class Segment:
    seg_id: str
    text: str
    importance: float
    kind: str


@dataclass(slots=True)
class TrialResult:
    target_word: str
    memory_orb_recall: float
    memory_orb_high_recall: float
    memory_orb_noise_rate: float
    baseline_recall: float
    baseline_high_recall: float
    baseline_noise_rate: float
    quick_keyword_recall: float
    quick_keyword_high_recall: float
    quick_keyword_noise_rate: float
    quick_qwen_recall: float
    quick_qwen_high_recall: float
    quick_qwen_noise_rate: float
    memory_orb_capture_count: int
    baseline_capture_count: int
    quick_keyword_capture_count: int
    quick_qwen_capture_count: int


def _extract_segment_ids(text: str) -> set[str]:
    return set(SEGMENT_ID_RE.findall(text.lower()))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\\-]{3,}", text.lower())


def _format_noise(rng: random.Random, count: int) -> str:
    vocab = [
        "pipeline",
        "latency",
        "orchestration",
        "cache",
        "batch",
        "notebook",
        "timeline",
        "checklist",
        "owner",
        "rollback",
        "sprint",
        "queue",
    ]
    return " ".join(rng.choice(vocab) for _ in range(count))


def _build_segment_text(seg_id: str, kind: str, target_word: str, rng: random.Random, mode: str = "literal") -> str:
    aliases = TARGET_ALIASES.get(target_word, [target_word])
    alias = rng.choice(aliases)
    semantic_phrase = f"{alias} workflow"
    target_phrase = target_word if mode == "literal" else alias

    if kind == "high":
        core = (
            f"{seg_id} Action note: {target_phrase} requires immediate mitigation. "
            f"Run verification, canary release, and alert thresholds before production cutover."
        )
    elif kind == "medium":
        core = (
            f"{seg_id} Planning note: {semantic_phrase} appears in this workstream and needs follow-up owners "
            "for next sprint checkpoints."
        )
    elif kind == "decoy":
        core = (
            f"{seg_id} Historical note: {target_word} is mentioned in archived docs but not tied to the current "
            "execution plan."
        )
    else:
        core = (
            f"{seg_id} General project chatter about unrelated status updates, documentation cleanup, "
            "and routine standup summaries."
        )
    tail = _format_noise(rng, rng.randint(14, 26))
    return f"{core} {tail}"


def generate_long_writing(
    target_word: str,
    seed: int,
    segments: int = 150,
    high_count: int = 8,
    medium_count: int = 12,
    decoy_count: int = 10,
    mode: str = "literal",
) -> list[Segment]:
    if mode not in {"literal", "semantic"}:
        raise ValueError("mode must be one of: literal, semantic")
    if high_count + medium_count + decoy_count >= segments:
        raise ValueError("importance counts must be less than total segments")

    rng = random.Random(seed)
    indices = list(range(segments))
    rng.shuffle(indices)
    high_idx = set(indices[:high_count])
    medium_idx = set(indices[high_count : high_count + medium_count])
    decoy_idx = set(indices[high_count + medium_count : high_count + medium_count + decoy_count])

    rows: list[Segment] = []
    for idx in range(segments):
        seg_id = f"seg{idx:03d}"
        if idx in high_idx:
            kind = "high"
            importance = 1.0
        elif idx in medium_idx:
            kind = "medium"
            importance = 0.6
        elif idx in decoy_idx:
            kind = "decoy"
            importance = 0.2
        else:
            kind = "noise"
            importance = 0.0

        text = _build_segment_text(seg_id=seg_id, kind=kind, target_word=target_word, rng=rng, mode=mode)
        rows.append(Segment(seg_id=seg_id, text=text, importance=importance, kind=kind))

    return rows


def _evaluate_capture(captured_ids: Iterable[str], rows: list[Segment]) -> tuple[float, float, float]:
    id_to_row = {row.seg_id: row for row in rows}
    captured = {seg_id for seg_id in captured_ids if seg_id in id_to_row}

    total_importance = sum(row.importance for row in rows)
    captured_importance = sum(id_to_row[seg_id].importance for seg_id in captured)

    high_ids = {row.seg_id for row in rows if row.kind == "high"}
    high_hits = len(high_ids.intersection(captured))
    high_recall = high_hits / max(1, len(high_ids))

    noise_hits = sum(1 for seg_id in captured if id_to_row[seg_id].kind == "noise")
    noise_rate = noise_hits / max(1, len(captured))

    weighted_recall = captured_importance / max(1e-9, total_importance)
    return weighted_recall, high_recall, noise_rate


def _run_memory_orb(rows: list[Segment], target_word: str, dataset_mode: str) -> tuple[set[str], int]:
    alias_map = {}
    if dataset_mode == "semantic":
        alias_map = {target_word: TARGET_ALIASES.get(target_word, [])}
    cfg = MemoryOrbEngineConfig(
        context_max_tokens=560,
        working_max_tokens=220,
        working_target_tokens=140,
        memory_budget_ratio=0.48,
        max_retrieved_orbs=8,
        min_focus_orb_count=1,
        anchor_aliases=alias_map,
    )
    engine = MemoryOrbEngine(config=cfg)

    for row in rows:
        engine.add_turn("user", row.text, metadata={"seg_id": row.seg_id, "importance": row.importance, "kind": row.kind})
        engine.add_turn("assistant", "Acknowledged. Tracking this item in memory.")

    packet = engine.build_context(
        user_query=f"Focus on {target_word} and give execution steps.",
        system_prompt="You are an operations planner.",
    )
    packet_text = "\n".join(msg["content"] for msg in packet.messages)
    captured = _extract_segment_ids(packet_text)
    return captured, len(captured)


def _run_recent_window_baseline(rows: list[Segment], context_budget_tokens: int = 560) -> tuple[set[str], int]:
    estimator = SimpleTokenEstimator()
    window: deque[tuple[str, int]] = deque()
    total_tokens = 0

    def push(text: str) -> None:
        nonlocal total_tokens
        tok = estimator.count(text)
        window.append((text, tok))
        total_tokens += tok
        while window and total_tokens > context_budget_tokens:
            _, popped = window.popleft()
            total_tokens -= popped

    for row in rows:
        push(row.text)
        push("Acknowledged. Tracking this item in memory.")

    combined = "\n".join(text for text, _ in window)
    captured = _extract_segment_ids(combined)
    return captured, len(captured)


def _run_keyword_quick_search(rows: list[Segment], target_word: str, context_budget_tokens: int = 560) -> tuple[set[str], int]:
    estimator = SimpleTokenEstimator()
    target_l = target_word.lower()

    ranked: list[tuple[float, Segment]] = []
    for row in rows:
        tokens = _tokenize(row.text)
        target_hits = tokens.count(target_l)
        # Intentional lexical-only baseline.
        score = (3.0 * target_hits) + (0.01 * len(tokens))
        ranked.append((score, row))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[Segment] = []
    used = 0
    for _, row in ranked:
        tok = estimator.count(row.text)
        if used + tok > context_budget_tokens and selected:
            continue
        selected.append(row)
        used += tok
        if used >= context_budget_tokens:
            break

    captured = {row.seg_id for row in selected}
    return captured, len(captured)


def _request_ollama_relevance(model: str, query: str, text: str, timeout_s: int = 60) -> float:
    prompt = (
        "You are a strict relevance scorer.\n"
        "Task: rate how relevant the segment is to the query.\n"
        "Use this rubric:\n"
        "- 1.0 = directly actionable for the current query.\n"
        "- 0.6 = related planning context.\n"
        "- 0.2 = historical/archived mention, not current action.\n"
        "- 0.0 = unrelated.\n"
        "Return only one decimal number between 0 and 1.\n"
        f"Query: {query}\n"
        f"Segment: {text}\n"
        "Score:"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "num_ctx": 1024,
        },
        "keep_alive": "10m",
    }
    req = Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except URLError as err:
        raise RuntimeError(f"Ollama request failed: {err}") from err

    parsed = json.loads(body)
    response_text = str(parsed.get("response", "")).strip()
    if not response_text:
        return 0.0

    try:
        direct = float(response_text)
        return max(0.0, min(1.0, direct))
    except ValueError:
        pass

    match = re.search(r"-?\d+(?:\.\d+)?", response_text)
    if not match:
        return 0.0
    value = float(match.group(0))
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def _run_qwen_quick_search(
    rows: list[Segment],
    target_word: str,
    model: str,
    context_budget_tokens: int = 560,
    timeout_s: int = 60,
) -> tuple[set[str], int]:
    estimator = SimpleTokenEstimator()
    query = f"Focus on {target_word} and give execution steps."

    scored: list[tuple[float, Segment]] = []
    for row in rows:
        score = _request_ollama_relevance(model=model, query=query, text=row.text, timeout_s=timeout_s)
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[Segment] = []
    used = 0
    for _, row in scored:
        tok = estimator.count(row.text)
        if used + tok > context_budget_tokens and selected:
            continue
        selected.append(row)
        used += tok
        if used >= context_budget_tokens:
            break

    captured = {row.seg_id for row in selected}
    return captured, len(captured)


def run_benchmark(
    trials: int,
    seed: int,
    qwen_model: str = "qwen3:0.6b",
    qwen_segments: int = 80,
    qwen_timeout_s: int = 60,
    dataset_mode: str = "semantic",
) -> dict[str, object]:
    target_words = [
        "oauth",
        "replication",
        "ledger",
        "latency",
        "checkpoint",
        "billing",
        "compliance",
        "rollback",
    ]
    rng = random.Random(seed)
    results: list[TrialResult] = []
    sample_rows: list[Segment] = []

    for trial_idx in range(trials):
        target = target_words[trial_idx % len(target_words)]
        trial_seed = rng.randint(0, 10_000_000)
        rows = generate_long_writing(target_word=target, seed=trial_seed, mode=dataset_mode)
        qwen_rows = rows[: max(8, qwen_segments)]
        if trial_idx == 0:
            sample_rows = rows

        orb_captured, orb_count = _run_memory_orb(rows, target, dataset_mode=dataset_mode)
        base_captured, base_count = _run_recent_window_baseline(rows)
        keyword_captured, keyword_count = _run_keyword_quick_search(qwen_rows, target_word=target)
        qwen_captured, qwen_count = _run_qwen_quick_search(
            qwen_rows,
            target_word=target,
            model=qwen_model,
            timeout_s=qwen_timeout_s,
        )

        orb_recall, orb_high_recall, orb_noise = _evaluate_capture(orb_captured, rows)
        base_recall, base_high_recall, base_noise = _evaluate_capture(base_captured, rows)
        keyword_recall, keyword_high_recall, keyword_noise = _evaluate_capture(keyword_captured, rows)
        qwen_recall, qwen_high_recall, qwen_noise = _evaluate_capture(qwen_captured, rows)

        results.append(
            TrialResult(
                target_word=target,
                memory_orb_recall=orb_recall,
                memory_orb_high_recall=orb_high_recall,
                memory_orb_noise_rate=orb_noise,
                baseline_recall=base_recall,
                baseline_high_recall=base_high_recall,
                baseline_noise_rate=base_noise,
                quick_keyword_recall=keyword_recall,
                quick_keyword_high_recall=keyword_high_recall,
                quick_keyword_noise_rate=keyword_noise,
                quick_qwen_recall=qwen_recall,
                quick_qwen_high_recall=qwen_high_recall,
                quick_qwen_noise_rate=qwen_noise,
                memory_orb_capture_count=orb_count,
                baseline_capture_count=base_count,
                quick_keyword_capture_count=keyword_count,
                quick_qwen_capture_count=qwen_count,
            )
        )

    def mean(values: list[float]) -> float:
        return float(statistics.mean(values)) if values else 0.0

    summary = {
        "trials": trials,
        "dataset_mode": dataset_mode,
        "memory_orb": {
            "weighted_recall_mean": mean([r.memory_orb_recall for r in results]),
            "high_signal_recall_mean": mean([r.memory_orb_high_recall for r in results]),
            "noise_rate_mean": mean([r.memory_orb_noise_rate for r in results]),
            "capture_count_mean": mean([float(r.memory_orb_capture_count) for r in results]),
        },
        "baseline_recent_window": {
            "weighted_recall_mean": mean([r.baseline_recall for r in results]),
            "high_signal_recall_mean": mean([r.baseline_high_recall for r in results]),
            "noise_rate_mean": mean([r.baseline_noise_rate for r in results]),
            "capture_count_mean": mean([float(r.baseline_capture_count) for r in results]),
        },
        "quick_keyword_baseline": {
            "weighted_recall_mean": mean([r.quick_keyword_recall for r in results]),
            "high_signal_recall_mean": mean([r.quick_keyword_high_recall for r in results]),
            "noise_rate_mean": mean([r.quick_keyword_noise_rate for r in results]),
            "capture_count_mean": mean([float(r.quick_keyword_capture_count) for r in results]),
        },
        "quick_qwen_chunk_search": {
            "model": qwen_model,
            "weighted_recall_mean": mean([r.quick_qwen_recall for r in results]),
            "high_signal_recall_mean": mean([r.quick_qwen_high_recall for r in results]),
            "noise_rate_mean": mean([r.quick_qwen_noise_rate for r in results]),
            "capture_count_mean": mean([float(r.quick_qwen_capture_count) for r in results]),
        },
        "delta_memory_orb_minus_baseline": {
            "weighted_recall": mean([r.memory_orb_recall - r.baseline_recall for r in results]),
            "high_signal_recall": mean([r.memory_orb_high_recall - r.baseline_high_recall for r in results]),
            "noise_rate": mean([r.memory_orb_noise_rate - r.baseline_noise_rate for r in results]),
        },
        "delta_qwen_minus_keyword": {
            "weighted_recall": mean([r.quick_qwen_recall - r.quick_keyword_recall for r in results]),
            "high_signal_recall": mean([r.quick_qwen_high_recall - r.quick_keyword_high_recall for r in results]),
            "noise_rate": mean([r.quick_qwen_noise_rate - r.quick_keyword_noise_rate for r in results]),
        },
        "sample_trial": asdict(results[0]) if results else {},
        "sample_labels": [
            {
                "seg_id": row.seg_id,
                "kind": row.kind,
                "importance": row.importance,
                "text_preview": row.text[:140],
            }
            for row in sample_rows[:16]
        ],
    }
    return summary


def _first_trial_rows(seed: int, mode: str = "semantic") -> list[Segment]:
    target_words = [
        "oauth",
        "replication",
        "ledger",
        "latency",
        "checkpoint",
        "billing",
        "compliance",
        "rollback",
    ]
    rng = random.Random(seed)
    target = target_words[0]
    trial_seed = rng.randint(0, 10_000_000)
    return generate_long_writing(target_word=target, seed=trial_seed, mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark selective-attention retrieval against recent-window baseline."
    )
    parser.add_argument("--trials", type=int, default=24, help="Number of benchmark trials.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write JSON summary.",
    )
    parser.add_argument(
        "--labels-out",
        type=str,
        default="",
        help="Optional path to write full segment labels for the first trial.",
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default="qwen3:0.6b",
        help="Ollama model for chunk-by-chunk relevance scoring.",
    )
    parser.add_argument(
        "--qwen-segments",
        type=int,
        default=80,
        help="How many leading segments to score with Qwen per trial.",
    )
    parser.add_argument(
        "--qwen-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each Ollama generate call.",
    )
    parser.add_argument(
        "--dataset-mode",
        type=str,
        default="semantic",
        choices=["literal", "semantic"],
        help="Synthetic dataset mode: literal keyword-heavy or semantic paraphrase-heavy.",
    )
    args = parser.parse_args()

    summary = run_benchmark(
        trials=max(1, args.trials),
        seed=args.seed,
        qwen_model=args.qwen_model,
        qwen_segments=max(8, args.qwen_segments),
        qwen_timeout_s=max(5, args.qwen_timeout),
        dataset_mode=args.dataset_mode,
    )
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.labels_out:
        rows = _first_trial_rows(seed=args.seed, mode=args.dataset_mode)
        payload = [asdict(row) for row in rows]
        with open(args.labels_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

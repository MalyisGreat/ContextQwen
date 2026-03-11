from __future__ import annotations

import argparse
import json
import random
import re
import socket
import statistics
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request, urlopen

from memory_orb import MemoryOrbEngine, MemoryOrbEngineConfig


VALUE_POOL = [
    "us-west-3",
    "eu-central-2",
    "ap-south-1",
    "ca-central-1",
    "sa-east-1",
    "queue-red",
    "queue-gold",
    "tier-7",
    "tier-9",
    "v3.14.7",
    "v4.2.1",
    "ledger-alpha",
    "ledger-beta",
    "key-9f2a",
    "key-c7d1",
]

NOISE_WORDS = [
    "backlog",
    "status",
    "checklist",
    "coordination",
    "incident",
    "rollup",
    "change",
    "branch",
    "pipeline",
    "timeline",
    "handoff",
    "cleanup",
    "tracker",
]

NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class RecallTrial:
    trial_id: str
    full_document: str
    question: str
    answer: str
    target_record: str
    target_value: str


@dataclass(slots=True)
class RecallResult:
    trial_id: str
    expected: str
    memory_orb_answer_raw: str
    long_ctx_answer_raw: str
    memory_orb_answer: str
    long_ctx_answer: str
    memory_orb_exact: int
    long_ctx_exact: int


def _normalize(text: str) -> str:
    return NORMALIZE_RE.sub(" ", (text or "").lower()).strip()


def _exact(expected: str, predicted: str) -> int:
    exp = _normalize(expected)
    pred = _normalize(predicted)
    return 1 if exp and (exp == pred or exp in pred.split()) else 0


def _extract_value(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            candidate_values: list[str] = []
            if isinstance(parsed, dict):
                for key in ("value", "current_value", "answer", "result"):
                    raw = parsed.get(key)
                    if raw is not None:
                        candidate_values.append(str(raw).strip())
                if not candidate_values:
                    for raw in parsed.values():
                        if isinstance(raw, (str, int, float)):
                            candidate_values.append(str(raw).strip())
                            break

            for value in candidate_values:
                if not value:
                    continue
                lower_value = value.lower()
                token_match = re.search(
                    r"(us-west-3|eu-central-2|ap-south-1|ca-central-1|sa-east-1|queue-red|queue-gold|tier-7|tier-9|v3\.14\.7|v4\.2\.1|ledger-alpha|ledger-beta|key-9f2a|key-c7d1)",
                    lower_value,
                )
                if token_match:
                    return token_match.group(1)
                return lower_value
        except Exception:
            return ""
        return ""
    lower = text.lower()
    for value in VALUE_POOL:
        if value in lower:
            return value

    patterns = [
        r"\b[a-z]{2}-[a-z]+-\d+\b",
        r"\bqueue-[a-z]+\b",
        r"\btier-\d+\b",
        r"\bv\d+\.\d+\.\d+\b",
        r"\bledger-[a-z]+\b",
        r"\bkey-[a-z0-9]+\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            return match.group(0)
    return lower.strip().split()[0] if lower.strip() else ""


def _ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    num_ctx: int,
    timeout_s: int = 120,
    json_mode: bool = False,
    think: bool = False,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "num_ctx": num_ctx,
            "num_predict": 256,
        },
        "keep_alive": "10m",
    }
    if json_mode:
        payload["format"] = "json"
    req = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except HTTPError as err:
        detail = ""
        try:
            detail = err.read().decode("utf-8")
        except Exception:
            detail = str(err)
        raise RuntimeError(f"Ollama chat request failed (HTTP {err.code}): {detail}") from err
    except URLError as err:
        raise RuntimeError(f"Ollama chat request failed: {err}") from err
    except TimeoutError as err:
        raise RuntimeError(f"Ollama chat request timed out: {err}") from err
    except socket.timeout as err:
        raise RuntimeError(f"Ollama chat request socket timeout: {err}") from err
    parsed: dict[str, Any] = json.loads(body)
    message = parsed.get("message") or {}
    content = str(message.get("content", "")).strip()
    if content:
        return content
    return str(message.get("thinking", "")).strip()


def _noise_segment(rng: random.Random, seg_id: int) -> str:
    words = " ".join(rng.choice(NOISE_WORDS) for _ in range(rng.randint(18, 40)))
    return f"seg{seg_id:04d} NOTE {words}."


def make_trial(seed: int, trial_idx: int, records: int = 260, noise: int = 700) -> RecallTrial:
    rng = random.Random(seed + trial_idx * 101)
    seg_id = 0
    entries: list[str] = []

    record_ids = [f"REC-{trial_idx:02d}-{i:03d}" for i in range(records)]
    target_record = rng.choice(record_ids)
    target_value = rng.choice(VALUE_POOL)

    for rid in record_ids:
        value = target_value if rid == target_record else rng.choice(VALUE_POOL)
        entries.append(
            f"seg{seg_id:04d} FACT record={rid} priority={rng.choice(['low', 'med', 'high'])} current_value={value}."
        )
        seg_id += 1
        if rng.random() < 0.35:
            entries.append(_noise_segment(rng, seg_id))
            seg_id += 1

    for _ in range(noise):
        entries.append(_noise_segment(rng, seg_id))
        seg_id += 1

    # Add hard distractors late in the doc.
    for _ in range(5):
        wrong = rng.choice([v for v in VALUE_POOL if v != target_value])
        entries.append(
            f"seg{seg_id:04d} FACT archived record={target_record} old_value={wrong} not_current=true."
        )
        seg_id += 1

    rng.shuffle(entries)
    # Ensure one canonical target fact appears once near early-middle.
    canonical = f"seg{seg_id:04d} FACT record={target_record} current_value={target_value} source=canonical."
    insert_at = max(1, len(entries) // 3)
    entries.insert(insert_at, canonical)

    document = "\n".join(entries)
    question = (
        f"For record {target_record}, what is the current_value? "
        "Answer with the value only. Ignore archived or old values."
    )
    return RecallTrial(
        trial_id=f"trial-{trial_idx:03d}",
        full_document=document,
        question=question,
        answer=target_value,
        target_record=target_record,
        target_value=target_value,
    )


def _split_doc(doc: str, chunk_chars: int = 480) -> list[str]:
    chunks: list[str] = []
    current = []
    size = 0
    for line in doc.splitlines():
        ln = line.strip()
        if not ln:
            continue
        need = len(ln) + 1
        if current and size + need > chunk_chars:
            chunks.append("\n".join(current))
            current = [ln]
            size = len(ln)
        else:
            current.append(ln)
            size += need
    if current:
        chunks.append("\n".join(current))
    return chunks


def run_memory_orb_fact_recall(
    trial: RecallTrial,
    model: str = "qwen3:0.6b",
    context_max_tokens: int = 1800,
    timeout_s: int = 120,
    dwell_mode: str = "heuristic",
    dwell_ctx: int = 900,
) -> str:
    class _OllamaAdapter:
        def __init__(self, model_name: str, num_ctx: int, timeout: int) -> None:
            self.model_name = model_name
            self.num_ctx = num_ctx
            self.timeout = timeout

        def complete(self, messages: list[dict[str, str]]) -> str:
            return _ollama_chat(
                model=self.model_name,
                messages=messages,
                num_ctx=self.num_ctx,
                timeout_s=self.timeout,
                json_mode=True,
            )

        def complete_with_reasoning(self, messages: list[dict[str, str]], think: bool = True) -> str:
            return _ollama_chat(
                model=self.model_name,
                messages=messages,
                num_ctx=self.num_ctx,
                timeout_s=self.timeout,
                json_mode=False,
                think=think,
            )

    engine = MemoryOrbEngine(
        config=MemoryOrbEngineConfig(
            context_max_tokens=context_max_tokens,
            working_max_tokens=1000,
            working_target_tokens=760,
            memory_budget_ratio=0.5,
            max_retrieved_orbs=10,
            min_focus_orb_count=2,
            answer_dwell_mode=dwell_mode,
        )
    )
    for chunk in _split_doc(trial.full_document):
        engine.add_turn(
            "tool",
            chunk,
            metadata={
                "importance": 0.6,
                "source": "benchmark_document",
                "exclude_from_pulse_map": True,
            },
        )
        engine.add_turn(
            "assistant",
            "",
            metadata={
                "source": "benchmark_boundary",
                "exclude_from_pulse_map": True,
                "exclude_from_question_memory_pool": True,
            },
        )

    adapter = _OllamaAdapter(model_name=model, num_ctx=context_max_tokens, timeout=timeout_s)
    dwell_adapter = _OllamaAdapter(
        model_name=model,
        num_ctx=max(512, min(context_max_tokens, dwell_ctx)),
        timeout=timeout_s,
    )
    result = engine.answer_with_answer_document(
        model=adapter,
        question=trial.question,
        passes=4,
        per_pass_orbs=4,
        answer_doc_max_tokens=int(context_max_tokens * 0.68),
        system_prompt=(
            "You answer factual retrieval questions.\n"
            "Return JSON only: {\"value\":\"...\"}.\n"
            "Use exactly one value token in the value field.\n"
            "Do not add extra keys."
        ),
        dwell_model=dwell_adapter if dwell_mode == "reasoned" else None,
    )
    return result.answer


def run_long_context_fact_recall(
    trial: RecallTrial,
    model: str = "qwen3:4b",
    num_ctx: int = 262144,
    timeout_s: int = 120,
) -> str:
    prompt = (
        "You answer factual retrieval questions from a long document.\n"
        "Return JSON only: {\"value\":\"...\"}.\n"
        "Use exactly one value token in the value field.\n"
        "Ignore archived/old_value entries.\n\n"
        f"Document:\n{trial.full_document}\n\n"
        f"Question: {trial.question}"
    )
    messages = [
        {"role": "system", "content": "Return JSON only with one key: value."},
        {"role": "user", "content": prompt},
    ]
    ctx_candidates = [num_ctx]
    for candidate in [196608, 131072, 98304, 65536, 49152, 32768]:
        if candidate < num_ctx and candidate not in ctx_candidates:
            ctx_candidates.append(candidate)

    last_error = ""
    for ctx in ctx_candidates:
        try:
            return _ollama_chat(
                model=model,
                messages=messages,
                num_ctx=ctx,
                timeout_s=timeout_s,
                json_mode=True,
            )
        except RuntimeError as err:
            last_error = str(err)
            continue

    raise RuntimeError(f"All long-context attempts failed. last_error={last_error}")


def run_compare(
    trials: int,
    seed: int,
    memory_model: str,
    long_model: str,
    long_model_ctx: int,
    timeout_s: int,
    records: int,
    noise: int,
    memory_dwell_mode: str,
    reasoning_dwell_ctx: int,
) -> dict[str, Any]:
    rows: list[RecallResult] = []
    for i in range(trials):
        trial = make_trial(seed=seed, trial_idx=i, records=records, noise=noise)
        mem_raw = run_memory_orb_fact_recall(
            trial,
            model=memory_model,
            timeout_s=timeout_s,
            dwell_mode=memory_dwell_mode,
            dwell_ctx=reasoning_dwell_ctx,
        )
        long_raw = run_long_context_fact_recall(trial, model=long_model, num_ctx=long_model_ctx, timeout_s=timeout_s)
        mem_ans = _extract_value(mem_raw)
        long_ans = _extract_value(long_raw)
        rows.append(
            RecallResult(
                trial_id=trial.trial_id,
                expected=trial.answer,
                memory_orb_answer_raw=mem_raw,
                long_ctx_answer_raw=long_raw,
                memory_orb_answer=mem_ans,
                long_ctx_answer=long_ans,
                memory_orb_exact=_exact(trial.answer, mem_ans),
                long_ctx_exact=_exact(trial.answer, long_ans),
            )
        )

    mem_acc = statistics.mean(r.memory_orb_exact for r in rows) if rows else 0.0
    long_acc = statistics.mean(r.long_ctx_exact for r in rows) if rows else 0.0
    return {
        "trials": trials,
        "memory_orb_model": memory_model,
        "memory_orb_dwell_mode": memory_dwell_mode,
        "memory_orb_reasoning_ctx": reasoning_dwell_ctx if memory_dwell_mode == "reasoned" else None,
        "long_context_model": long_model,
        "long_context_window": long_model_ctx,
        "memory_orb_exact_match": mem_acc,
        "long_context_exact_match": long_acc,
        "delta_memory_orb_minus_long_ctx": mem_acc - long_acc,
        "records_per_trial": records,
        "noise_segments_per_trial": noise,
        "results": [asdict(r) for r in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare factual recall: Memory Orb vs long-context Qwen.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--memory-model", type=str, default="qwen3:0.6b")
    parser.add_argument("--long-model", type=str, default="qwen3:4b")
    parser.add_argument("--long-model-ctx", type=int, default=262144)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--records", type=int, default=260)
    parser.add_argument("--noise", type=int, default=700)
    parser.add_argument("--memory-dwell-mode", type=str, choices=["heuristic", "reasoned"], default="heuristic")
    parser.add_argument("--reasoning-dwell-ctx", type=int, default=900)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    summary = run_compare(
        trials=max(1, args.trials),
        seed=args.seed,
        memory_model=args.memory_model,
        long_model=args.long_model,
        long_model_ctx=max(8192, args.long_model_ctx),
        timeout_s=max(30, args.timeout),
        records=max(32, args.records),
        noise=max(64, args.noise),
        memory_dwell_mode=args.memory_dwell_mode,
        reasoning_dwell_ctx=max(256, args.reasoning_dwell_ctx),
    )
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

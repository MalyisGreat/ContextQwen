from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from benchmarks.api_backend import ChatBackendConfig
from benchmarks.api_backend import chat_completion
from memory_orb import MemoryOrbEngine
from memory_orb import MemoryOrbEngineConfig
from memory_orb.adapters import ModelAdapter


@dataclass(slots=True)
class MRCRCase:
    case_id: str
    n_needles: int
    desired_msg_index: int
    total_messages: int
    n_chars: int
    prompt_messages: list[dict[str, str]]
    answer: str
    random_string_to_prepend: str
    date_added: str


@dataclass(slots=True)
class MRCRResult:
    case_id: str
    n_needles: int
    desired_msg_index: int
    total_messages: int
    n_chars: int
    answer_chars: int
    direct_raw: str
    direct_exact: int
    direct_contains: int
    direct_prefix: int
    direct_error: str
    memory_raw: str
    memory_exact: int
    memory_contains: int
    memory_prefix: int
    memory_error: str
    direct_latency_s: float
    memory_latency_s: float
    orb_count: int


class _ChatModelAdapter(ModelAdapter):
    def __init__(self, backend: ChatBackendConfig, model_name: str, num_ctx: int, timeout_s: int, num_predict: int) -> None:
        self.backend = backend
        self.model_name = model_name
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.num_predict = num_predict

    def complete(self, messages: list[dict[str, str]]) -> str:
        return chat_completion(
            backend=self.backend,
            model=self.model_name,
            messages=messages,
            num_ctx=self.num_ctx,
            timeout_s=self.timeout_s,
            json_mode=False,
            response_schema=None,
            num_predict=self.num_predict,
        )


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int((done / float(total)) * width)
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _estimate_answer_tokens(answer: str) -> int:
    rough = max(96, int(math.ceil(len(answer or "") / 3.2)) + 48)
    return min(4096, rough)


def _select_cases(sample_size: int, seed: int, n_needles: int) -> list[MRCRCase]:
    dataset = load_dataset("openai/mrcr", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=max(64, sample_size * 8))
    selected: list[MRCRCase] = []
    for idx, row in enumerate(dataset):
        row_needles = int(row["n_needles"])
        if row_needles != n_needles:
            continue
        selected.append(
            MRCRCase(
                case_id=f"mrcr-{n_needles}-{idx}",
                n_needles=row_needles,
                desired_msg_index=int(row["desired_msg_index"]),
                total_messages=int(row["total_messages"]),
                n_chars=int(row["n_chars"]),
                prompt_messages=json.loads(row["prompt"]),
                answer=str(row["answer"]),
                random_string_to_prepend=str(row["random_string_to_prepend"]),
                date_added=str(row["date_added"]),
            )
        )
        if len(selected) >= sample_size:
            break
    return selected


def _evaluate_prediction(prediction: str, answer: str, prefix: str) -> tuple[int, int, int]:
    pred_norm = _normalize_text(prediction)
    answer_norm = _normalize_text(answer)
    exact = 1 if pred_norm == answer_norm and answer_norm else 0
    contains = 1 if answer_norm and answer_norm in pred_norm else 0
    prefix_hit = 1 if pred_norm.startswith(prefix) else 0
    return exact, contains, prefix_hit


def _run_direct_case(case: MRCRCase, model: str, direct_ctx: int, timeout_s: int, backend: ChatBackendConfig) -> str:
    return chat_completion(
        backend=backend,
        model=model,
        messages=case.prompt_messages,
        num_ctx=direct_ctx,
        timeout_s=timeout_s,
        json_mode=False,
        response_schema=None,
        num_predict=_estimate_answer_tokens(case.answer),
    )


def _build_memory_engine(memory_ctx: int, answer_dwell_mode: str) -> MemoryOrbEngine:
    return MemoryOrbEngine(
        config=MemoryOrbEngineConfig(
            context_max_tokens=memory_ctx,
            working_max_tokens=max(700, int(memory_ctx * 0.68)),
            working_target_tokens=max(520, int(memory_ctx * 0.52)),
            memory_budget_ratio=0.6,
            max_retrieved_orbs=16,
            min_focus_orb_count=2,
            answer_dwell_mode=answer_dwell_mode,
        )
    )


def _run_memory_case(
    case: MRCRCase,
    model: str,
    memory_ctx: int,
    timeout_s: int,
    backend: ChatBackendConfig,
    answer_dwell_mode: str,
) -> tuple[str, int]:
    if not case.prompt_messages:
        raise ValueError("MRCR case has no prompt messages")
    if case.prompt_messages[-1].get("role") != "user":
        raise ValueError("MRCR case does not end in a user query")

    engine = _build_memory_engine(memory_ctx=memory_ctx, answer_dwell_mode=answer_dwell_mode)
    final_user = case.prompt_messages[-1]["content"]
    for message in case.prompt_messages[:-1]:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if role == "system":
            role = "tool"
        engine.add_turn(
            role,
            content,
            metadata={"source": "mrcr_history"},
        )

    adapter = _ChatModelAdapter(
        backend=backend,
        model_name=model,
        num_ctx=memory_ctx,
        timeout_s=timeout_s,
        num_predict=_estimate_answer_tokens(case.answer),
    )
    answer, _packet = engine.chat(
        model=adapter,
        user_text=final_user,
        system_prompt=(
            "Answer using the prior conversation history only.\n"
            "Return only the exact requested text with no explanation, paraphrase, or extra words."
        ),
    )
    return answer, len(engine._orbs)


def run_mrcr_compare(
    sample_size: int,
    seed: int,
    n_needles: int,
    model: str,
    direct_ctx: int,
    memory_ctx: int,
    timeout_s: int,
    backend: ChatBackendConfig,
    answer_dwell_mode: str = "heuristic",
    show_progress: bool = True,
) -> dict[str, Any]:
    cases = _select_cases(sample_size=sample_size, seed=seed, n_needles=n_needles)
    results: list[MRCRResult] = []
    started = time.time()
    direct_latencies: list[float] = []
    memory_latencies: list[float] = []

    total_cases = len(cases)
    for idx, case in enumerate(cases, start=1):
        direct_raw = ""
        direct_error = ""
        direct_exact = 0
        direct_contains = 0
        direct_prefix = 0
        direct_latency_s = 0.0
        try:
            direct_started = time.perf_counter()
            direct_raw = _run_direct_case(case=case, model=model, direct_ctx=direct_ctx, timeout_s=timeout_s, backend=backend)
            direct_latency_s = time.perf_counter() - direct_started
            direct_exact, direct_contains, direct_prefix = _evaluate_prediction(
                prediction=direct_raw,
                answer=case.answer,
                prefix=case.random_string_to_prepend,
            )
        except Exception as err:
            direct_latency_s = time.perf_counter() - direct_started
            direct_error = str(err)
        direct_latencies.append(direct_latency_s)

        memory_raw = ""
        memory_error = ""
        memory_exact = 0
        memory_contains = 0
        memory_prefix = 0
        memory_latency_s = 0.0
        orb_count = 0
        try:
            memory_started = time.perf_counter()
            memory_raw, orb_count = _run_memory_case(
                case=case,
                model=model,
                memory_ctx=memory_ctx,
                timeout_s=timeout_s,
                backend=backend,
                answer_dwell_mode=answer_dwell_mode,
            )
            memory_latency_s = time.perf_counter() - memory_started
            memory_exact, memory_contains, memory_prefix = _evaluate_prediction(
                prediction=memory_raw,
                answer=case.answer,
                prefix=case.random_string_to_prepend,
            )
        except Exception as err:
            memory_latency_s = time.perf_counter() - memory_started
            memory_error = str(err)
        memory_latencies.append(memory_latency_s)

        results.append(
            MRCRResult(
                case_id=case.case_id,
                n_needles=case.n_needles,
                desired_msg_index=case.desired_msg_index,
                total_messages=case.total_messages,
                n_chars=case.n_chars,
                answer_chars=len(case.answer),
                direct_raw=direct_raw,
                direct_exact=direct_exact,
                direct_contains=direct_contains,
                direct_prefix=direct_prefix,
                direct_error=direct_error,
                memory_raw=memory_raw,
                memory_exact=memory_exact,
                memory_contains=memory_contains,
                memory_prefix=memory_prefix,
                memory_error=memory_error,
                direct_latency_s=round(direct_latency_s, 3),
                memory_latency_s=round(memory_latency_s, 3),
                orb_count=orb_count,
            )
        )

        if show_progress:
            bar = _progress_bar(idx, total_cases)
            direct_running = statistics.mean(row.direct_exact for row in results)
            memory_running = statistics.mean(row.memory_exact for row in results)
            print(
                f"{bar} {idx}/{total_cases} "
                f"case={case.case_id} "
                f"direct_exact={direct_exact} "
                f"memory_exact={memory_exact} "
                f"run_direct={direct_running:.3f} "
                f"run_memory={memory_running:.3f} "
                f"direct_t={statistics.mean(direct_latencies):.2f}s "
                f"memory_t={statistics.mean(memory_latencies):.2f}s",
                flush=True,
            )

    elapsed = time.time() - started
    return {
        "benchmark": "openai/mrcr",
        "sample_size": len(results),
        "seed": seed,
        "n_needles": n_needles,
        "model": model,
        "backend": backend.as_dict(),
        "direct_ctx": direct_ctx,
        "memory_ctx": memory_ctx,
        "answer_dwell_mode": answer_dwell_mode,
        "elapsed_seconds": round(elapsed, 2),
        "direct_exact_match": statistics.mean(row.direct_exact for row in results) if results else 0.0,
        "memory_exact_match": statistics.mean(row.memory_exact for row in results) if results else 0.0,
        "direct_contains_match": statistics.mean(row.direct_contains for row in results) if results else 0.0,
        "memory_contains_match": statistics.mean(row.memory_contains for row in results) if results else 0.0,
        "direct_prefix_match": statistics.mean(row.direct_prefix for row in results) if results else 0.0,
        "memory_prefix_match": statistics.mean(row.memory_prefix for row in results) if results else 0.0,
        "direct_mean_latency_s": round(statistics.mean(direct_latencies), 3) if direct_latencies else 0.0,
        "memory_mean_latency_s": round(statistics.mean(memory_latencies), 3) if memory_latencies else 0.0,
        "direct_error_count": sum(1 for row in results if row.direct_error),
        "memory_error_count": sum(1 for row in results if row.memory_error),
        "results": [asdict(row) for row in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct long-context inference vs Memory Orb on the OpenAI MRCR benchmark.")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--needles", type=str, default="8", help="Comma-separated MRCR variants to run: 2,4,8")
    parser.add_argument("--model", type=str, default="qwen3.5:0.8b")
    parser.add_argument("--direct-ctx", type=int, default=262144)
    parser.add_argument("--memory-ctx", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--backend-provider",
        type=str,
        choices=["ollama", "openai", "openai-compatible", "vllm"],
        default="ollama",
        help="Inference backend. Use openai/vllm for an OpenAI-compatible server such as vLLM.",
    )
    parser.add_argument("--api-base", type=str, default="", help="Base URL for an OpenAI-compatible server.")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key for the OpenAI-compatible server.")
    parser.add_argument("--answer-dwell-mode", type=str, choices=["heuristic", "reasoned"], default="heuristic")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    backend = ChatBackendConfig(
        provider=args.backend_provider,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    needle_values = [int(item.strip()) for item in args.needles.split(",") if item.strip()]
    runs = []
    for n_needles in needle_values:
        runs.append(
            run_mrcr_compare(
                sample_size=max(1, args.sample_size),
                seed=args.seed,
                n_needles=n_needles,
                model=args.model,
                direct_ctx=max(8192, args.direct_ctx),
                memory_ctx=max(800, args.memory_ctx),
                timeout_s=max(30, args.timeout),
                backend=backend,
                answer_dwell_mode=args.answer_dwell_mode,
                show_progress=not bool(args.no_progress),
            )
        )
    summary = {"benchmark": "openai/mrcr", "runs": runs}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()

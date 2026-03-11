from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.api_backend import ChatBackendConfig
from benchmarks.longbench_v2_compare import _extract_choice
from benchmarks.longbench_v2_compare import _progress_bar
from benchmarks.longbench_v2_compare import _run_memory_case
from benchmarks.longbench_v2_compare import _select_cases


@dataclass(slots=True)
class MemoryOnlyBenchResult:
    case_id: str
    length: str
    difficulty: str
    domain: str
    sub_domain: str
    answer: str
    memory_raw: str
    memory_pred: str
    memory_correct: int
    memory_error: str
    memory_latency_s: float
    context_chars: int
    chunk_count: int
    route_name: str = ""
    route_profile_shape: str = ""
    route_profile_confidence: float = 0.0
    deterministic_reader_used: int = 0
    reader_evidence_excerpt: str = ""


def run_memory_only(
    sample_size: int,
    seed: int,
    lengths: set[str],
    max_context_chars: int,
    model: str,
    memory_ctx: int,
    timeout_s: int,
    chunk_chars: int,
    memory_answer_mode: str,
    memory_dwell_mode: str,
    reasoning_dwell_ctx: int,
    backend: ChatBackendConfig,
    reasoning_num_predict: int = 192,
    enable_ollama_think: bool = False,
    difficulty_filter: set[str] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    cases = _select_cases(
        sample_size=sample_size,
        seed=seed,
        lengths=lengths,
        max_context_chars=max_context_chars,
        difficulty_filter=difficulty_filter,
    )
    results: list[MemoryOnlyBenchResult] = []
    memory_latencies: list[float] = []
    started = time.time()

    total_cases = len(cases)
    for idx, case in enumerate(cases, start=1):
        memory_raw = ""
        memory_pred = ""
        memory_correct = 0
        memory_error = ""
        chunk_count = 0
        memory_latency_s = 0.0
        route_name = ""
        route_profile_shape = ""
        route_profile_confidence = 0.0
        deterministic_reader_used = 0
        reader_evidence_excerpt = ""
        try:
            memory_started = time.perf_counter()
            memory_run = _run_memory_case(
                case=case,
                model=model,
                memory_ctx=memory_ctx,
                timeout_s=timeout_s,
                chunk_chars=chunk_chars,
                memory_answer_mode=memory_answer_mode,
                memory_dwell_mode=memory_dwell_mode,
                reasoning_dwell_ctx=reasoning_dwell_ctx,
                backend=backend,
                reasoning_num_predict=reasoning_num_predict,
                enable_ollama_think=enable_ollama_think,
            )
            memory_latency_s = time.perf_counter() - memory_started
            memory_raw = memory_run.answer_raw
            memory_pred = _extract_choice(memory_raw)
            memory_correct = 1 if memory_pred == case.answer else 0
            chunk_count = memory_run.chunk_count
            route_name = memory_run.route_name
            route_profile_shape = memory_run.route_profile_shape
            route_profile_confidence = memory_run.route_profile_confidence
            deterministic_reader_used = 1 if memory_run.deterministic_reader_used else 0
            reader_evidence_excerpt = memory_run.reader_evidence_excerpt
        except Exception as err:
            memory_latency_s = time.perf_counter() - memory_started
            memory_error = str(err)
        memory_latencies.append(memory_latency_s)
        results.append(
            MemoryOnlyBenchResult(
                case_id=case.case_id,
                length=case.length,
                difficulty=case.difficulty,
                domain=case.domain,
                sub_domain=case.sub_domain,
                answer=case.answer,
                memory_raw=memory_raw,
                memory_pred=memory_pred,
                memory_correct=memory_correct,
                memory_error=memory_error,
                memory_latency_s=round(memory_latency_s, 3),
                context_chars=len(case.context),
                chunk_count=chunk_count,
                route_name=route_name,
                route_profile_shape=route_profile_shape,
                route_profile_confidence=route_profile_confidence,
                deterministic_reader_used=deterministic_reader_used,
                reader_evidence_excerpt=reader_evidence_excerpt,
            )
        )
        if show_progress:
            running = statistics.mean(row.memory_correct for row in results)
            status = "ok" if memory_correct else ("err" if memory_error else "x")
            bar = _progress_bar(idx, total_cases)
            print(
                f"{bar} {idx}/{total_cases} "
                f"case={case.case_id[:8]} "
                f"mem={memory_pred or '-'}({status}) "
                f"run_mem={running:.3f} "
                f"t_mem={statistics.mean(memory_latencies):.2f}s",
                flush=True,
            )

    elapsed = time.time() - started
    return {
        "benchmark": "THUDM/LongBench-v2",
        "task_format": "multiple-choice (A/B/C/D)",
        "sample_size": len(results),
        "requested_sample_size": sample_size,
        "seed": seed,
        "length_filter": sorted(lengths),
        "difficulty_filter": sorted(difficulty_filter) if difficulty_filter else [],
        "max_context_chars": max_context_chars,
        "model": model,
        "backend": backend.as_dict(),
        "memory_ctx": memory_ctx,
        "memory_answer_mode": memory_answer_mode,
        "memory_dwell_mode": memory_dwell_mode,
        "reasoning_dwell_ctx": reasoning_dwell_ctx if memory_answer_mode == "reasoned-chat" or memory_dwell_mode == "reasoned" else None,
        "reasoning_num_predict": reasoning_num_predict if memory_answer_mode == "reasoned-chat" or memory_dwell_mode == "reasoned" else None,
        "enable_ollama_think": bool(enable_ollama_think),
        "chunk_chars": chunk_chars,
        "elapsed_seconds": round(elapsed, 2),
        "memory_accuracy": statistics.mean(row.memory_correct for row in results) if results else 0.0,
        "memory_mean_latency_s": round(statistics.mean(memory_latencies), 3) if memory_latencies else 0.0,
        "memory_error_count": sum(1 for row in results if row.memory_error),
        "results": [asdict(row) for row in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-only LongBench-v2 runner with configurable reasoning budget.")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths", type=str, default="medium", help="Comma-separated: short,medium,long")
    parser.add_argument("--difficulty", type=str, default="", help="Optional comma-separated filter: easy,hard")
    parser.add_argument("--max-context-chars", type=int, default=220000)
    parser.add_argument("--model", type=str, default="qwen3.5:0.8b")
    parser.add_argument("--memory-ctx", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--chunk-chars", type=int, default=1400)
    parser.add_argument(
        "--backend-provider",
        type=str,
        choices=["ollama", "openai", "openai-compatible", "vllm"],
        default="ollama",
        help="Inference backend. Use openai/vllm for an OpenAI-compatible server such as vLLM.",
    )
    parser.add_argument("--api-base", type=str, default="", help="Base URL for an OpenAI-compatible server.")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key for the OpenAI-compatible server.")
    parser.add_argument("--memory-answer-mode", type=str, choices=["chat", "answer-doc", "reasoned-chat"], default="reasoned-chat")
    parser.add_argument("--memory-dwell-mode", type=str, choices=["heuristic", "reasoned"], default="reasoned")
    parser.add_argument("--reasoning-dwell-ctx", type=int, default=900)
    parser.add_argument("--reasoning-num-predict", type=int, default=192)
    parser.add_argument(
        "--reasoning-predict-multiplier",
        type=float,
        default=1.0,
        help="Multiplies --reasoning-num-predict. Example: 5.0 gives 5x the default reasoning output budget.",
    )
    parser.add_argument("--enable-ollama-think", action="store_true", help="Allow Ollama think mode for supported models on reasoning subcalls.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-sample progress lines.")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    lengths = {item.strip() for item in args.lengths.split(",") if item.strip()}
    difficulty_filter = {item.strip() for item in args.difficulty.split(",") if item.strip()} or None
    base_predict = max(32, args.reasoning_num_predict)
    reasoning_num_predict = max(32, int(round(base_predict * max(0.1, args.reasoning_predict_multiplier))))
    backend = ChatBackendConfig(
        provider=args.backend_provider,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    summary = run_memory_only(
        sample_size=max(1, args.sample_size),
        seed=args.seed,
        lengths=lengths,
        max_context_chars=max(20000, args.max_context_chars),
        model=args.model,
        memory_ctx=max(800, args.memory_ctx),
        timeout_s=max(30, args.timeout),
        chunk_chars=max(500, args.chunk_chars),
        memory_answer_mode=args.memory_answer_mode,
        memory_dwell_mode=args.memory_dwell_mode,
        reasoning_dwell_ctx=max(256, args.reasoning_dwell_ctx),
        backend=backend,
        reasoning_num_predict=reasoning_num_predict,
        enable_ollama_think=bool(args.enable_ollama_think),
        difficulty_filter=difficulty_filter,
        show_progress=not bool(args.no_progress),
    )
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()

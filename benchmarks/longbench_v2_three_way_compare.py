from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from benchmarks.longbench_v2_compare import BenchCase
from benchmarks.longbench_v2_compare import _extract_choice
from benchmarks.longbench_v2_compare import _progress_bar
from benchmarks.longbench_v2_compare import _run_direct_case
from benchmarks.longbench_v2_compare import _run_memory_case
from benchmarks.longbench_v2_compare import _select_cases


@dataclass(slots=True)
class ThreeWayBenchResult:
    global_index: int
    case_id: str
    length: str
    difficulty: str
    domain: str
    sub_domain: str
    answer: str
    direct_raw: str
    direct_pred: str
    direct_correct: int
    direct_error: str
    direct_latency_s: float
    heuristic_raw: str
    heuristic_pred: str
    heuristic_correct: int
    heuristic_error: str
    heuristic_latency_s: float
    reasoned_raw: str
    reasoned_pred: str
    reasoned_correct: int
    reasoned_error: str
    reasoned_latency_s: float
    context_chars: int
    heuristic_chunk_count: int
    reasoned_chunk_count: int
    heuristic_route_name: str = ""
    heuristic_route_profile_shape: str = ""
    heuristic_route_profile_confidence: float = 0.0
    heuristic_deterministic_reader_used: int = 0
    heuristic_reader_evidence_excerpt: str = ""
    reasoned_route_name: str = ""
    reasoned_route_profile_shape: str = ""
    reasoned_route_profile_confidence: float = 0.0
    reasoned_deterministic_reader_used: int = 0
    reasoned_reader_evidence_excerpt: str = ""


def run_three_way_compare(
    sample_size: int,
    seed: int,
    lengths: set[str],
    max_context_chars: int,
    model: str,
    direct_ctx: int,
    memory_ctx: int,
    timeout_s: int,
    chunk_chars: int,
    reasoning_dwell_ctx: int,
    difficulty_filter: set[str] | None = None,
    show_progress: bool = True,
    start_index: int = 1,
    end_index: int | None = None,
    checkpoint_path: str = "",
) -> dict[str, Any]:
    selected_cases: list[BenchCase] = _select_cases(
        sample_size=sample_size,
        seed=seed,
        lengths=lengths,
        max_context_chars=max_context_chars,
        difficulty_filter=difficulty_filter,
    )
    if start_index < 1:
        raise ValueError("start_index must be >= 1")
    if end_index is None:
        end_index = len(selected_cases)
    if end_index < start_index:
        raise ValueError("end_index must be >= start_index")
    indexed_cases = list(enumerate(selected_cases, start=1))[start_index - 1 : end_index]
    results: list[ThreeWayBenchResult] = []
    processed_case_ids: set[str] = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        for row in checkpoint.get("results", []):
            results.append(ThreeWayBenchResult(**row))
            processed_case_ids.add(row["case_id"])
    started = time.time()

    direct_latencies: list[float] = [row.direct_latency_s for row in results]
    heuristic_latencies: list[float] = [row.heuristic_latency_s for row in results]
    reasoned_latencies: list[float] = [row.reasoned_latency_s for row in results]

    total_cases = len(indexed_cases)
    completed = len(results)
    for global_index, case in indexed_cases:
        if case.case_id in processed_case_ids:
            continue
        direct_raw = ""
        direct_pred = ""
        direct_correct = 0
        direct_error = ""
        direct_latency_s = 0.0
        heuristic_raw = ""
        heuristic_pred = ""
        heuristic_correct = 0
        heuristic_error = ""
        heuristic_latency_s = 0.0
        reasoned_raw = ""
        reasoned_pred = ""
        reasoned_correct = 0
        reasoned_error = ""
        reasoned_latency_s = 0.0
        heuristic_chunk_count = 0
        reasoned_chunk_count = 0

        direct_started = time.perf_counter()
        try:
            direct_raw = _run_direct_case(case=case, model=model, direct_ctx=direct_ctx, timeout_s=timeout_s)
            direct_pred = _extract_choice(direct_raw)
            direct_correct = 1 if direct_pred == case.answer else 0
        except Exception as err:
            direct_error = str(err)
        finally:
            direct_latency_s = time.perf_counter() - direct_started
            direct_latencies.append(direct_latency_s)

        heuristic_started = time.perf_counter()
        heuristic_route_name = ""
        heuristic_route_profile_shape = ""
        heuristic_route_profile_confidence = 0.0
        heuristic_deterministic_reader_used = 0
        heuristic_reader_evidence_excerpt = ""
        try:
            heuristic_run = _run_memory_case(
                case=case,
                model=model,
                memory_ctx=memory_ctx,
                timeout_s=timeout_s,
                chunk_chars=chunk_chars,
                memory_answer_mode="chat",
                memory_dwell_mode="heuristic",
                reasoning_dwell_ctx=reasoning_dwell_ctx,
            )
            heuristic_raw = heuristic_run.answer_raw
            heuristic_chunk_count = heuristic_run.chunk_count
            heuristic_route_name = heuristic_run.route_name
            heuristic_route_profile_shape = heuristic_run.route_profile_shape
            heuristic_route_profile_confidence = heuristic_run.route_profile_confidence
            heuristic_deterministic_reader_used = 1 if heuristic_run.deterministic_reader_used else 0
            heuristic_reader_evidence_excerpt = heuristic_run.reader_evidence_excerpt
            heuristic_pred = _extract_choice(heuristic_raw)
            heuristic_correct = 1 if heuristic_pred == case.answer else 0
        except Exception as err:
            heuristic_error = str(err)
        finally:
            heuristic_latency_s = time.perf_counter() - heuristic_started
            heuristic_latencies.append(heuristic_latency_s)

        reasoned_started = time.perf_counter()
        reasoned_route_name = ""
        reasoned_route_profile_shape = ""
        reasoned_route_profile_confidence = 0.0
        reasoned_deterministic_reader_used = 0
        reasoned_reader_evidence_excerpt = ""
        try:
            reasoned_run = _run_memory_case(
                case=case,
                model=model,
                memory_ctx=memory_ctx,
                timeout_s=timeout_s,
                chunk_chars=chunk_chars,
                memory_answer_mode="reasoned-chat",
                memory_dwell_mode="reasoned",
                reasoning_dwell_ctx=reasoning_dwell_ctx,
            )
            reasoned_raw = reasoned_run.answer_raw
            reasoned_chunk_count = reasoned_run.chunk_count
            reasoned_route_name = reasoned_run.route_name
            reasoned_route_profile_shape = reasoned_run.route_profile_shape
            reasoned_route_profile_confidence = reasoned_run.route_profile_confidence
            reasoned_deterministic_reader_used = 1 if reasoned_run.deterministic_reader_used else 0
            reasoned_reader_evidence_excerpt = reasoned_run.reader_evidence_excerpt
            reasoned_pred = _extract_choice(reasoned_raw)
            reasoned_correct = 1 if reasoned_pred == case.answer else 0
        except Exception as err:
            reasoned_error = str(err)
        finally:
            reasoned_latency_s = time.perf_counter() - reasoned_started
            reasoned_latencies.append(reasoned_latency_s)

        results.append(
            ThreeWayBenchResult(
                global_index=global_index,
                case_id=case.case_id,
                length=case.length,
                difficulty=case.difficulty,
                domain=case.domain,
                sub_domain=case.sub_domain,
                answer=case.answer,
                direct_raw=direct_raw,
                direct_pred=direct_pred,
                direct_correct=direct_correct,
                direct_error=direct_error,
                direct_latency_s=round(direct_latency_s, 3),
                heuristic_raw=heuristic_raw,
                heuristic_pred=heuristic_pred,
                heuristic_correct=heuristic_correct,
                heuristic_error=heuristic_error,
                heuristic_latency_s=round(heuristic_latency_s, 3),
                reasoned_raw=reasoned_raw,
                reasoned_pred=reasoned_pred,
                reasoned_correct=reasoned_correct,
                reasoned_error=reasoned_error,
                reasoned_latency_s=round(reasoned_latency_s, 3),
                context_chars=len(case.context),
                heuristic_chunk_count=heuristic_chunk_count,
                reasoned_chunk_count=reasoned_chunk_count,
                heuristic_route_name=heuristic_route_name,
                heuristic_route_profile_shape=heuristic_route_profile_shape,
                heuristic_route_profile_confidence=heuristic_route_profile_confidence,
                heuristic_deterministic_reader_used=heuristic_deterministic_reader_used,
                heuristic_reader_evidence_excerpt=heuristic_reader_evidence_excerpt,
                reasoned_route_name=reasoned_route_name,
                reasoned_route_profile_shape=reasoned_route_profile_shape,
                reasoned_route_profile_confidence=reasoned_route_profile_confidence,
                reasoned_deterministic_reader_used=reasoned_deterministic_reader_used,
                reasoned_reader_evidence_excerpt=reasoned_reader_evidence_excerpt,
            )
        )
        completed += 1
        if checkpoint_path:
            _write_result_payload(
                checkpoint_path,
                _result_payload(
                    sample_size=sample_size,
                    seed=seed,
                    lengths=lengths,
                    difficulty_filter=difficulty_filter,
                    max_context_chars=max_context_chars,
                    model=model,
                    direct_ctx=direct_ctx,
                    memory_ctx=memory_ctx,
                    reasoning_dwell_ctx=reasoning_dwell_ctx,
                    chunk_chars=chunk_chars,
                    elapsed=time.time() - started,
                    results=results,
                    direct_latencies=direct_latencies,
                    heuristic_latencies=heuristic_latencies,
                    reasoned_latencies=reasoned_latencies,
                    start_index=start_index,
                    end_index=end_index,
                    total_selected_cases=len(selected_cases),
                ),
            )

        if show_progress:
            direct_running = statistics.mean(row.direct_correct for row in results)
            heuristic_running = statistics.mean(row.heuristic_correct for row in results)
            reasoned_running = statistics.mean(row.reasoned_correct for row in results)
            bar = _progress_bar(completed, total_cases)
            direct_status = "ok" if direct_correct else ("err" if direct_error else "x")
            heuristic_status = "ok" if heuristic_correct else ("err" if heuristic_error else "x")
            reasoned_status = "ok" if reasoned_correct else ("err" if reasoned_error else "x")
            print(
                f"{bar} {completed}/{total_cases} "
                f"case={case.case_id[:8]} "
                f"direct={direct_pred or '-'}({direct_status}) "
                f"heur={heuristic_pred or '-'}({heuristic_status}) "
                f"new={reasoned_pred or '-'}({reasoned_status}) "
                f"run_direct={direct_running:.3f} "
                f"run_heur={heuristic_running:.3f} "
                f"run_new={reasoned_running:.3f} "
                f"t_direct={statistics.mean(direct_latencies):.2f}s "
                f"t_heur={statistics.mean(heuristic_latencies):.2f}s "
                f"t_new={statistics.mean(reasoned_latencies):.2f}s",
                flush=True,
            )

    elapsed = time.time() - started
    return _result_payload(
        sample_size=sample_size,
        seed=seed,
        lengths=lengths,
        difficulty_filter=difficulty_filter,
        max_context_chars=max_context_chars,
        model=model,
        direct_ctx=direct_ctx,
        memory_ctx=memory_ctx,
        reasoning_dwell_ctx=reasoning_dwell_ctx,
        chunk_chars=chunk_chars,
        elapsed=elapsed,
        results=results,
        direct_latencies=direct_latencies,
        heuristic_latencies=heuristic_latencies,
        reasoned_latencies=reasoned_latencies,
        start_index=start_index,
        end_index=end_index,
        total_selected_cases=len(selected_cases),
    )


def _result_payload(
    sample_size: int,
    seed: int,
    lengths: set[str],
    difficulty_filter: set[str] | None,
    max_context_chars: int,
    model: str,
    direct_ctx: int,
    memory_ctx: int,
    reasoning_dwell_ctx: int,
    chunk_chars: int,
    elapsed: float,
    results: list[ThreeWayBenchResult],
    direct_latencies: list[float],
    heuristic_latencies: list[float],
    reasoned_latencies: list[float],
    start_index: int,
    end_index: int,
    total_selected_cases: int,
) -> dict[str, Any]:
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
        "direct_ctx": direct_ctx,
        "memory_ctx": memory_ctx,
        "reasoned_memory_answer_mode": "reasoned-chat",
        "reasoning_dwell_ctx": reasoning_dwell_ctx,
        "chunk_chars": chunk_chars,
        "start_index": start_index,
        "end_index": end_index,
        "total_selected_cases": total_selected_cases,
        "elapsed_seconds": round(elapsed, 2),
        "direct_accuracy": statistics.mean(row.direct_correct for row in results) if results else 0.0,
        "heuristic_accuracy": statistics.mean(row.heuristic_correct for row in results) if results else 0.0,
        "reasoned_accuracy": statistics.mean(row.reasoned_correct for row in results) if results else 0.0,
        "heuristic_minus_direct": (
            statistics.mean(row.heuristic_correct for row in results)
            - statistics.mean(row.direct_correct for row in results)
        )
        if results
        else 0.0,
        "reasoned_minus_direct": (
            statistics.mean(row.reasoned_correct for row in results)
            - statistics.mean(row.direct_correct for row in results)
        )
        if results
        else 0.0,
        "reasoned_minus_heuristic": (
            statistics.mean(row.reasoned_correct for row in results)
            - statistics.mean(row.heuristic_correct for row in results)
        )
        if results
        else 0.0,
        "direct_mean_latency_s": round(statistics.mean(direct_latencies), 3) if direct_latencies else 0.0,
        "heuristic_mean_latency_s": round(statistics.mean(heuristic_latencies), 3) if heuristic_latencies else 0.0,
        "reasoned_mean_latency_s": round(statistics.mean(reasoned_latencies), 3) if reasoned_latencies else 0.0,
        "direct_error_count": sum(1 for row in results if row.direct_error),
        "heuristic_error_count": sum(1 for row in results if row.heuristic_error),
        "reasoned_error_count": sum(1 for row in results if row.reasoned_error),
        "results": [asdict(row) for row in results],
    }


def _write_result_payload(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way LongBench compare: direct vs heuristic Memory Orb vs reasoned Memory Orb.")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths", type=str, default="medium", help="Comma-separated: short,medium,long")
    parser.add_argument("--difficulty", type=str, default="", help="Optional comma-separated filter: easy,hard")
    parser.add_argument("--max-context-chars", type=int, default=220000)
    parser.add_argument("--model", type=str, default="qwen3.5:0.8b")
    parser.add_argument("--direct-ctx", type=int, default=262144)
    parser.add_argument("--memory-ctx", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--chunk-chars", type=int, default=1400)
    parser.add_argument("--reasoning-dwell-ctx", type=int, default=900)
    parser.add_argument("--start-index", type=int, default=1, help="1-based start index within the deterministic sampled case list.")
    parser.add_argument("--end-index", type=int, default=0, help="1-based inclusive end index within the deterministic sampled case list. 0 means full range.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-sample progress lines.")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    lengths = {item.strip() for item in args.lengths.split(",") if item.strip()}
    difficulty_filter = {item.strip() for item in args.difficulty.split(",") if item.strip()} or None
    summary = run_three_way_compare(
        sample_size=max(1, args.sample_size),
        seed=args.seed,
        lengths=lengths,
        max_context_chars=max(20000, args.max_context_chars),
        model=args.model,
        direct_ctx=max(8192, args.direct_ctx),
        memory_ctx=max(800, args.memory_ctx),
        timeout_s=max(30, args.timeout),
        chunk_chars=max(500, args.chunk_chars),
        reasoning_dwell_ctx=max(256, args.reasoning_dwell_ctx),
        difficulty_filter=difficulty_filter,
        show_progress=not bool(args.no_progress),
        start_index=max(1, args.start_index),
        end_index=max(1, args.end_index) if args.end_index else None,
        checkpoint_path=args.json_out,
    )
    print(json.dumps(summary, indent=2))
    if args.json_out:
        _write_result_payload(args.json_out, summary)


if __name__ == "__main__":
    main()

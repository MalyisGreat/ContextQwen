from __future__ import annotations

import json
from dataclasses import asdict

from benchmarks import longbench_v2_three_way_compare as three_way
from benchmarks.longbench_v2_compare import BenchCase
from benchmarks.longbench_v2_compare import _MemoryRunOutcome


def _make_case(case_id: str, answer: str = "A") -> BenchCase:
    return BenchCase(
        case_id=case_id,
        length="medium",
        difficulty="easy",
        domain="Single-Document QA",
        sub_domain="Detective",
        question="Which answer is correct?",
        context=f"context for {case_id}",
        choice_a="a",
        choice_b="b",
        choice_c="c",
        choice_d="d",
        answer=answer,
    )


def test_run_three_way_compare_respects_subrange_and_checkpoint(tmp_path, monkeypatch):
    cases = [_make_case("c1"), _make_case("c2"), _make_case("c3")]

    monkeypatch.setattr(three_way, "_select_cases", lambda **_: cases)
    monkeypatch.setattr(three_way, "_run_direct_case", lambda **_: '{"answer":"A"}')
    monkeypatch.setattr(
        three_way,
        "_run_memory_case",
        lambda **kwargs: _MemoryRunOutcome(
            answer_raw='{"answer":"A"}',
            chunk_count=2,
            route_name="orb/chat",
            route_profile_shape="prose",
            route_profile_confidence=0.2,
            deterministic_reader_used=False,
            reader_evidence_excerpt="",
        ),
    )

    checkpoint_path = tmp_path / "chunk.json"
    checkpoint_payload = {
        "results": [
            {
                "global_index": 2,
                "case_id": "c2",
                "length": "medium",
                "difficulty": "easy",
                "domain": "Single-Document QA",
                "sub_domain": "Detective",
                "answer": "A",
                "direct_raw": '{"answer":"A"}',
                "direct_pred": "A",
                "direct_correct": 1,
                "direct_error": "",
                "direct_latency_s": 1.0,
                "heuristic_raw": '{"answer":"A"}',
                "heuristic_pred": "A",
                "heuristic_correct": 1,
                "heuristic_error": "",
                "heuristic_latency_s": 2.0,
                "reasoned_raw": '{"answer":"A"}',
                "reasoned_pred": "A",
                "reasoned_correct": 1,
                "reasoned_error": "",
                "reasoned_latency_s": 3.0,
                "context_chars": 10,
                "heuristic_chunk_count": 2,
                "reasoned_chunk_count": 2,
                "heuristic_route_name": "orb/chat",
                "heuristic_route_profile_shape": "prose",
                "heuristic_route_profile_confidence": 0.2,
                "heuristic_deterministic_reader_used": 0,
                "heuristic_reader_evidence_excerpt": "",
                "reasoned_route_name": "orb/reasoned-chat",
                "reasoned_route_profile_shape": "prose",
                "reasoned_route_profile_confidence": 0.2,
                "reasoned_deterministic_reader_used": 0,
                "reasoned_reader_evidence_excerpt": "",
            }
        ]
    }
    checkpoint_path.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

    summary = three_way.run_three_way_compare(
        sample_size=3,
        seed=42,
        lengths={"medium"},
        max_context_chars=1000,
        model="qwen3.5:0.8b",
        direct_ctx=1024,
        memory_ctx=900,
        timeout_s=30,
        chunk_chars=500,
        reasoning_dwell_ctx=512,
        show_progress=False,
        start_index=2,
        end_index=3,
        checkpoint_path=str(checkpoint_path),
    )

    assert summary["sample_size"] == 2
    assert summary["start_index"] == 2
    assert summary["end_index"] == 3
    assert [row["case_id"] for row in summary["results"]] == ["c2", "c3"]
    assert checkpoint_path.exists()


def test_result_payload_keeps_route_audit_fields():
    row = three_way.ThreeWayBenchResult(
        global_index=1,
        case_id="c1",
        length="medium",
        difficulty="easy",
        domain="Single-Document QA",
        sub_domain="Manual",
        answer="A",
        direct_raw='{"answer":"A"}',
        direct_pred="A",
        direct_correct=1,
        direct_error="",
        direct_latency_s=1.0,
        heuristic_raw='{"answer":"B"}',
        heuristic_pred="B",
        heuristic_correct=0,
        heuristic_error="",
        heuristic_latency_s=2.0,
        reasoned_raw='{"answer":"A"}',
        reasoned_pred="A",
        reasoned_correct=1,
        reasoned_error="",
        reasoned_latency_s=3.0,
        context_chars=100,
        heuristic_chunk_count=2,
        reasoned_chunk_count=2,
        heuristic_route_name="orb/chat",
        heuristic_route_profile_shape="prose",
        heuristic_route_profile_confidence=0.3,
        heuristic_deterministic_reader_used=0,
        heuristic_reader_evidence_excerpt="",
        reasoned_route_name="manual/claim-matrix",
        reasoned_route_profile_shape="manual",
        reasoned_route_profile_confidence=0.92,
        reasoned_deterministic_reader_used=0,
        reasoned_reader_evidence_excerpt="A: overall=supported",
    )

    payload = three_way._result_payload(
        sample_size=1,
        seed=42,
        lengths={"medium"},
        difficulty_filter=None,
        max_context_chars=1000,
        model="qwen3.5:0.8b",
        direct_ctx=1024,
        memory_ctx=900,
        reasoning_dwell_ctx=512,
        chunk_chars=500,
        elapsed=1.0,
        results=[row],
        direct_latencies=[1.0],
        heuristic_latencies=[2.0],
        reasoned_latencies=[3.0],
        start_index=1,
        end_index=1,
        total_selected_cases=1,
    )

    result_row = payload["results"][0]
    assert result_row["reasoned_route_name"] == "manual/claim-matrix"
    assert result_row["reasoned_route_profile_shape"] == "manual"
    assert result_row["reasoned_route_profile_confidence"] == 0.92

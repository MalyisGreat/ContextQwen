from __future__ import annotations

from benchmarks import longbench_v2_memory_only as memory_only
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


def test_run_memory_only_passes_reasoning_overrides(monkeypatch):
    cases = [_make_case("c1"), _make_case("c2", answer="B")]
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(memory_only, "_select_cases", lambda **_: cases)

    def fake_run_memory_case(**kwargs):
        captured.append(kwargs)
        answer = '{"answer":"A"}' if kwargs["case"].case_id == "c1" else '{"answer":"B"}'
        return _MemoryRunOutcome(
            answer_raw=answer,
            chunk_count=3,
            route_name="orb/reasoned-chat",
            route_profile_shape="prose",
            route_profile_confidence=0.4,
            deterministic_reader_used=False,
            reader_evidence_excerpt="",
        )

    monkeypatch.setattr(memory_only, "_run_memory_case", fake_run_memory_case)

    summary = memory_only.run_memory_only(
        sample_size=2,
        seed=42,
        lengths={"medium"},
        max_context_chars=1000,
        model="qwen3.5:0.8b",
        memory_ctx=2200,
        timeout_s=30,
        chunk_chars=500,
        memory_answer_mode="reasoned-chat",
        memory_dwell_mode="reasoned",
        reasoning_dwell_ctx=900,
        reasoning_num_predict=960,
        enable_ollama_think=True,
        show_progress=False,
    )

    assert len(captured) == 2
    assert all(call["reasoning_num_predict"] == 960 for call in captured)
    assert all(call["enable_ollama_think"] is True for call in captured)
    assert summary["memory_accuracy"] == 1.0
    assert summary["reasoning_num_predict"] == 960
    assert summary["enable_ollama_think"] is True

from __future__ import annotations

from benchmarks.api_backend import ChatBackendConfig
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
        backend=ChatBackendConfig(provider="ollama"),
        reasoning_num_predict=960,
        enable_ollama_think=True,
        show_progress=False,
    )

    assert len(captured) == 2
    assert all(call["reasoning_num_predict"] == 960 for call in captured)
    assert all(call["enable_ollama_think"] is True for call in captured)
    assert all(call["backend"].normalized_provider() == "ollama" for call in captured)
    assert summary["memory_accuracy"] == 1.0
    assert summary["reasoning_num_predict"] == 960
    assert summary["enable_ollama_think"] is True


def test_run_memory_only_reports_permutation_audit(monkeypatch):
    cases = [_make_case("c1", answer="B")]

    monkeypatch.setattr(memory_only, "_select_cases", lambda **_: cases)
    monkeypatch.setattr(
        memory_only,
        "_permute_case_labels",
        lambda case, seed_token: (
            BenchCase(
                case_id=case.case_id,
                length=case.length,
                difficulty=case.difficulty,
                domain=case.domain,
                sub_domain=case.sub_domain,
                question=case.question,
                context=case.context,
                choice_a=case.choice_b,
                choice_b=case.choice_c,
                choice_c=case.choice_d,
                choice_d=case.choice_a,
                answer="A",
            ),
            {"A": "B", "B": "C", "C": "D", "D": "A"},
        ),
    )

    def fake_run_memory_case(**kwargs):
        case = kwargs["case"]
        return _MemoryRunOutcome(
            answer_raw=f'{{"answer":"{case.answer}"}}',
            chunk_count=2,
            route_name="orb/reasoned-chat",
            route_profile_shape="prose",
            route_profile_confidence=0.6,
            deterministic_reader_used=False,
            reader_evidence_excerpt="",
        )

    monkeypatch.setattr(memory_only, "_run_memory_case", fake_run_memory_case)

    summary = memory_only.run_memory_only(
        sample_size=1,
        seed=42,
        lengths={"medium"},
        max_context_chars=1000,
        model="Qwen/Qwen3.5-0.8B",
        memory_ctx=2200,
        timeout_s=30,
        chunk_chars=500,
        memory_answer_mode="reasoned-chat",
        memory_dwell_mode="reasoned",
        reasoning_dwell_ctx=900,
        backend=ChatBackendConfig(provider="openai", api_base="http://127.0.0.1:8000/v1"),
        permutation_audit=True,
        show_progress=False,
    )

    assert summary["permutation_audit_enabled"] is True
    assert summary["memory_permuted_same_letter_rate"] == 0.0
    assert summary["memory_permuted_mapped_accuracy"] == 1.0

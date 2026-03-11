from __future__ import annotations

from benchmarks.linked_longform_benchmark import _extract_answer
from benchmarks.linked_longform_benchmark import _is_exact_match
from benchmarks.linked_longform_benchmark import build_linked_cases
from benchmarks.linked_longform_benchmark import dataset_summary


def test_longform_dataset_has_10_linked_cases():
    cases = build_linked_cases()
    assert len(cases) == 10

    for case in cases:
        word_count = len(case.context.split())
        assert word_count >= 850
        lower_context = case.context.lower()
        assert case.answer.lower() in lower_context
        assert case.legacy_code.lower() in lower_context
        assert case.pilot_code.lower() in lower_context
        assert case.answer != case.pilot_code
        assert case.answer != case.legacy_code
        assert case.question


def test_dataset_summary_reports_nontrivial_lengths():
    cases = build_linked_cases()
    summary = dataset_summary(cases)

    assert summary["case_count"] == 10
    assert summary["avg_words_per_passage"] >= 850
    assert summary["min_words_per_passage"] >= 850
    assert summary["max_words_per_passage"] >= summary["min_words_per_passage"]


def test_answer_extraction_accepts_json_and_plain_text():
    assert _extract_answer('{"answer":"HBR-LOCK-731"}') == "HBR-LOCK-731"
    assert _extract_answer('{"final_code":"MNT-FLOW-284"}') == "MNT-FLOW-284"
    assert _extract_answer("The final approved control is ARC-NODE-553.") == "ARC-NODE-553"


def test_exact_match_accepts_aliases():
    expected = "DELTA-ROUTE-909"
    aliases = ["delta-route-909", "deltaroute909"]
    assert _is_exact_match(expected, aliases, "DELTA-ROUTE-909") == 1
    assert _is_exact_match(expected, aliases, "deltaroute909") == 1
    assert _is_exact_match(expected, aliases, "DELTA-ROUTE-846") == 0

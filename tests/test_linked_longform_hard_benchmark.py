from __future__ import annotations

from benchmarks.linked_longform_hard_benchmark import _extract_hard_answer
from benchmarks.linked_longform_hard_benchmark import _score_case_fields
from benchmarks.linked_longform_hard_benchmark import build_hard_cases
from benchmarks.linked_longform_hard_benchmark import hard_dataset_summary


def test_hard_dataset_has_long_multi_fact_cases():
    cases = build_hard_cases()
    assert len(cases) == 10

    for case in cases:
        words = len(case.context.split())
        assert words >= 1000

        lower_ctx = case.context.lower()
        assert case.expected["final_code"].lower() in lower_ctx
        assert case.expected["approver"].lower() in lower_ctx
        assert case.expected["final_date"].lower() in lower_ctx
        assert case.expected["constraint"].lower() in lower_ctx

        assert case.decoy_code.lower() in lower_ctx
        assert case.decoy_approver.lower() in lower_ctx
        assert case.decoy_date.lower() in lower_ctx


def test_hard_dataset_summary_lengths():
    summary = hard_dataset_summary(build_hard_cases())
    assert summary["case_count"] == 10
    assert summary["avg_words_per_passage"] >= 1000
    assert summary["min_words_per_passage"] >= 1000


def test_extract_hard_answer_from_json():
    raw = (
        '{"final_code":"DELTA-ROUTE-909","approver":"Noah Finley",'
        '"final_date":"July 11, 2025","constraint":"peaker unit spin-up latency under heat alerts"}'
    )
    parsed = _extract_hard_answer(raw)
    assert parsed["final_code"] == "DELTA-ROUTE-909"
    assert parsed["approver"] == "Noah Finley"
    assert parsed["final_date"] == "July 11, 2025"
    assert "peaker unit spin-up latency" in parsed["constraint"]


def test_score_case_fields_uses_multi_field_match():
    expected = {
        "final_code": "DELTA-ROUTE-909",
        "approver": "Noah Finley",
        "final_date": "July 11, 2025",
        "constraint": "peaker unit spin-up latency under heat alerts",
    }
    predicted_good = {
        "final_code": "DELTA-ROUTE-909",
        "approver": "Noah Finley",
        "final_date": "July 11, 2025",
        "constraint": "the peaker unit spin-up latency under heat alerts was unresolved",
    }
    predicted_bad = {
        "final_code": "DELTA-ROUTE-846",
        "approver": "Noah Finley",
        "final_date": "July 11, 2025",
        "constraint": "different phrase entirely",
    }

    good_scores = _score_case_fields(expected, predicted_good)
    bad_scores = _score_case_fields(expected, predicted_bad)

    assert good_scores["all_fields_match"] == 1
    assert bad_scores["final_code_exact"] == 0
    assert bad_scores["all_fields_match"] == 0

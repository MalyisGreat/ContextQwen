from __future__ import annotations

from benchmarks.api_backend import ChatBackendConfig
from benchmarks import mrcr_compare


def test_evaluate_prediction_tracks_exact_contains_and_prefix():
    exact, contains, prefix = mrcr_compare._evaluate_prediction(
        prediction="ABC123 hello world",
        answer="ABC123 hello world",
        prefix="ABC123",
    )

    assert (exact, contains, prefix) == (1, 1, 1)


def test_run_mrcr_compare_smoke(monkeypatch):
    cases = [
        mrcr_compare.MRCRCase(
            case_id="mrcr-8-0",
            n_needles=8,
            desired_msg_index=11,
            total_messages=12,
            n_chars=1280,
            prompt_messages=[
                {"role": "user", "content": "write a poem about clocks"},
                {"role": "assistant", "content": "time text"},
                {"role": "user", "content": "Prepend ABC123 to the 1st poem about clocks. Do not include any other text in your response."},
            ],
            answer="ABC123time text",
            random_string_to_prepend="ABC123",
            date_added="2025-04-12",
        )
    ]

    monkeypatch.setattr(mrcr_compare, "_select_cases", lambda **_: cases)
    monkeypatch.setattr(
        mrcr_compare,
        "_run_direct_case",
        lambda **kwargs: "ABC123time text",
    )
    monkeypatch.setattr(
        mrcr_compare,
        "_run_memory_case",
        lambda **kwargs: ("ABC123time text", 2),
    )

    summary = mrcr_compare.run_mrcr_compare(
        sample_size=1,
        seed=42,
        n_needles=8,
        model="qwen3.5:0.8b",
        direct_ctx=262144,
        memory_ctx=2200,
        timeout_s=30,
        backend=ChatBackendConfig(provider="ollama"),
        show_progress=False,
    )

    assert summary["direct_exact_match"] == 1.0
    assert summary["memory_exact_match"] == 1.0
    assert summary["direct_prefix_match"] == 1.0
    assert summary["memory_prefix_match"] == 1.0

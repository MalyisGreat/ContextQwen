from __future__ import annotations

import json

from benchmarks.api_backend import ChatBackendConfig
from benchmarks.api_backend import _normalize_messages_for_chat_template
from benchmarks.api_backend import chat_completion
import benchmarks.api_backend as api_backend


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload
        self.status = 200

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_openai_backend_normalizes_base_url():
    backend = ChatBackendConfig(provider="vllm", api_base="http://127.0.0.1:8000")

    assert backend.normalized_provider() == "openai"
    assert backend.normalized_api_base() == "http://127.0.0.1:8000/v1"


def test_chat_completion_openai_parses_message_text(monkeypatch):
    def fake_urlopen(request, timeout=0):
        assert request.full_url == "http://127.0.0.1:8000/v1/chat/completions"
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "result text",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(api_backend, "urlopen", fake_urlopen)

    result = chat_completion(
        backend=ChatBackendConfig(provider="openai", api_base="http://127.0.0.1:8000/v1", api_key="EMPTY"),
        model="Qwen/Qwen3.5-0.8B",
        messages=[{"role": "user", "content": "hello"}],
        num_ctx=4096,
        timeout_s=30,
        json_mode=False,
        num_predict=32,
    )

    assert result == "result text"


def test_normalize_messages_moves_system_messages_to_front():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "system", "content": "follow policy"},
        {"role": "user", "content": "question"},
    ]

    normalized = _normalize_messages_for_chat_template(messages)

    assert [item["role"] for item in normalized] == ["system", "user", "assistant", "user"]
    assert normalized[0]["content"] == "follow policy"

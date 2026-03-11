from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen


@dataclass(frozen=True, slots=True)
class ChatBackendConfig:
    provider: str = "ollama"
    api_base: str = ""
    api_key: str = ""
    keep_alive: str = ""

    def normalized_provider(self) -> str:
        value = (self.provider or "ollama").strip().lower()
        if value in {"openai-compatible", "openai_compatible", "vllm"}:
            return "openai"
        return value

    def normalized_api_base(self) -> str:
        base = (self.api_base or "").strip().rstrip("/")
        if not base:
            return "http://127.0.0.1:8000/v1"
        if base.endswith("/chat/completions"):
            return base[: -len("/chat/completions")]
        if base.endswith("/v1"):
            return base
        return base + "/v1"

    def effective_keep_alive(self) -> str:
        if self.keep_alive.strip():
            return self.keep_alive.strip()
        return os.environ.get("MEMORY_ORB_OLLAMA_KEEP_ALIVE", "0s")

    def as_dict(self) -> dict[str, str]:
        return {
            "provider": self.normalized_provider(),
            "api_base": self.normalized_api_base() if self.normalized_provider() == "openai" else "",
            "keep_alive": self.effective_keep_alive() if self.normalized_provider() == "ollama" else "",
        }


def chat_completion(
    backend: ChatBackendConfig,
    model: str,
    messages: list[dict[str, str]],
    num_ctx: int,
    timeout_s: int = 180,
    json_mode: bool = True,
    response_schema: dict[str, Any] | None = None,
    think: bool = False,
    num_predict: int = 64,
) -> str:
    provider = backend.normalized_provider()
    if provider == "ollama":
        return _ollama_chat(
            backend=backend,
            model=model,
            messages=messages,
            num_ctx=num_ctx,
            timeout_s=timeout_s,
            json_mode=json_mode,
            response_schema=response_schema,
            think=think,
            num_predict=num_predict,
        )
    if provider == "openai":
        return _openai_chat(
            backend=backend,
            model=model,
            messages=messages,
            timeout_s=timeout_s,
            num_predict=num_predict,
        )
    raise ValueError(f"Unsupported backend provider: {backend.provider}")


def _ollama_chat(
    backend: ChatBackendConfig,
    model: str,
    messages: list[dict[str, str]],
    num_ctx: int,
    timeout_s: int = 180,
    json_mode: bool = True,
    response_schema: dict[str, Any] | None = None,
    think: bool = False,
    num_predict: int = 64,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "keep_alive": backend.effective_keep_alive(),
    }
    if response_schema is not None:
        payload["format"] = response_schema
    elif json_mode:
        payload["format"] = "json"

    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except HTTPError as err:
        detail = ""
        try:
            detail = err.read().decode("utf-8")
        except Exception:
            detail = str(err)
        raise RuntimeError(f"Ollama chat failed (HTTP {err.code}): {detail}") from err
    except URLError as err:
        raise RuntimeError(f"Ollama chat failed: {err}") from err
    except TimeoutError as err:
        raise RuntimeError(f"Ollama chat timed out: {err}") from err
    except socket.timeout as err:
        raise RuntimeError(f"Ollama socket timeout: {err}") from err

    parsed: dict[str, Any] = json.loads(body)
    message = parsed.get("message") or {}
    content = str(message.get("content", "")).strip()
    if content:
        return content
    return str(message.get("thinking", "")).strip()


def _openai_chat(
    backend: ChatBackendConfig,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: int = 180,
    num_predict: int = 64,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "top_p": 0.1,
        "max_tokens": num_predict,
    }
    request = Request(
        backend.normalized_api_base() + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {backend.api_key or 'EMPTY'}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except HTTPError as err:
        detail = ""
        try:
            detail = err.read().decode("utf-8")
        except Exception:
            detail = str(err)
        raise RuntimeError(f"OpenAI-compatible chat failed (HTTP {err.code}): {detail}") from err
    except URLError as err:
        raise RuntimeError(f"OpenAI-compatible chat failed: {err}") from err
    except TimeoutError as err:
        raise RuntimeError(f"OpenAI-compatible chat timed out: {err}") from err
    except socket.timeout as err:
        raise RuntimeError(f"OpenAI-compatible socket timeout: {err}") from err

    parsed: dict[str, Any] = json.loads(body)
    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compatible chat returned no choices: {body[:400]}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return str(content).strip()

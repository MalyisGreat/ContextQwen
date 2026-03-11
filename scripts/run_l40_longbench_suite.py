from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen


def _run_and_tee(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return process.wait()


def _wait_for_ollama(timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    request = Request("http://127.0.0.1:11434/api/tags", method="GET")
    last_error = ""
    while time.time() < deadline:
        try:
            with urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return
        except URLError as err:
            last_error = str(err)
        except Exception as err:  # pragma: no cover - operational guard
            last_error = str(err)
        time.sleep(1.0)
    raise RuntimeError(f"Ollama did not become ready within {timeout_s}s: {last_error}")


def _warm_model(model: str, keep_alive: str, timeout_s: int = 120) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": 'Reply with JSON only: {"answer":"OK"}'}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "top_p": 0.1, "num_ctx": 512, "num_predict": 16},
        "keep_alive": keep_alive,
    }
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        if response.status != 200:
            raise RuntimeError(f"Warmup failed with HTTP {response.status}")
        response.read()


def _start_ollama_server(workdir: Path, env: dict[str, str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["ollama", "serve"],
        cwd=str(workdir),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an L40-oriented LongBench suite with model pull, warmup, compare, and logs.")
    parser.add_argument("--model", type=str, default="qwen3.5:0.8b")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths", type=str, default="medium")
    parser.add_argument("--difficulty", type=str, default="")
    parser.add_argument("--max-context-chars", type=int, default=400000)
    parser.add_argument("--direct-ctx", type=int, default=262144)
    parser.add_argument("--memory-ctx", type=int, default=2200)
    parser.add_argument("--reasoning-dwell-ctx", type=int, default=900)
    parser.add_argument("--reasoning-num-predict", type=int, default=192)
    parser.add_argument("--reasoning-predict-multiplier", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--chunk-chars", type=int, default=1400)
    parser.add_argument("--runs-root", type=str, default="runs/l40-longbench")
    parser.add_argument("--ollama-keep-alive", type=str, default="30m")
    parser.add_argument("--ollama-max-loaded-models", type=int, default=1)
    parser.add_argument("--ollama-num-parallel", type=int, default=1)
    parser.add_argument("--ollama-max-queue", type=int, default=512)
    parser.add_argument("--ollama-kv-cache-type", type=str, default="q8_0")
    parser.add_argument("--cuda-visible-devices", type=str, default="")
    parser.add_argument("--extra-model", action="append", default=[], help="Additional Ollama models to pull before running.")
    parser.add_argument("--enable-ollama-think", action="store_true")
    parser.add_argument("--skip-server-start", action="store_true", help="Assume an Ollama server is already running.")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-memory-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = repo_root / args.runs_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OLLAMA_FLASH_ATTENTION"] = env.get("OLLAMA_FLASH_ATTENTION", "1")
    env["OLLAMA_MAX_LOADED_MODELS"] = str(max(1, args.ollama_max_loaded_models))
    env["OLLAMA_NUM_PARALLEL"] = str(max(1, args.ollama_num_parallel))
    env["OLLAMA_MAX_QUEUE"] = str(max(1, args.ollama_max_queue))
    env["OLLAMA_KV_CACHE_TYPE"] = args.ollama_kv_cache_type
    env["OLLAMA_KEEP_ALIVE"] = args.ollama_keep_alive
    env["MEMORY_ORB_OLLAMA_KEEP_ALIVE"] = args.ollama_keep_alive
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    pulled_models = [args.model] + [item for item in args.extra_model if item]
    deduped_models: list[str] = []
    seen: set[str] = set()
    for model in pulled_models:
        if model not in seen:
            deduped_models.append(model)
            seen.add(model)

    manifest = {
        "model": args.model,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "lengths": args.lengths,
        "difficulty": args.difficulty,
        "max_context_chars": args.max_context_chars,
        "direct_ctx": args.direct_ctx,
        "memory_ctx": args.memory_ctx,
        "reasoning_dwell_ctx": args.reasoning_dwell_ctx,
        "reasoning_num_predict": args.reasoning_num_predict,
        "reasoning_predict_multiplier": args.reasoning_predict_multiplier,
        "enable_ollama_think": bool(args.enable_ollama_think),
        "ollama_keep_alive": args.ollama_keep_alive,
        "ollama_max_loaded_models": args.ollama_max_loaded_models,
        "ollama_num_parallel": args.ollama_num_parallel,
        "ollama_max_queue": args.ollama_max_queue,
        "ollama_kv_cache_type": args.ollama_kv_cache_type,
        "cuda_visible_devices": args.cuda_visible_devices,
        "models_to_pull": deduped_models,
    }
    _write_manifest(run_dir / "manifest.json", manifest)

    server_process: subprocess.Popen[str] | None = None
    try:
        if not args.skip_server_start:
            server_process = _start_ollama_server(repo_root, env, run_dir / "ollama-server.log")
            _wait_for_ollama(timeout_s=90)

        if not args.skip_pull:
            for model in deduped_models:
                code = _run_and_tee(["ollama", "pull", model], repo_root, env, run_dir / f"pull-{model.replace(':', '_')}.log")
                if code != 0:
                    raise RuntimeError(f"Failed to pull model {model}")

        _warm_model(args.model, keep_alive=args.ollama_keep_alive, timeout_s=max(60, args.timeout))

        compare_json = run_dir / "compare-direct-vs-reasoned.json"
        compare_cmd = [
            sys.executable,
            "-m",
            "benchmarks.longbench_v2_compare",
            "--sample-size",
            str(max(1, args.sample_size)),
            "--seed",
            str(args.seed),
            "--lengths",
            args.lengths,
            "--max-context-chars",
            str(max(20000, args.max_context_chars)),
            "--memory-model",
            args.model,
            "--long-model",
            args.model,
            "--direct-ctx",
            str(max(8192, args.direct_ctx)),
            "--memory-ctx",
            str(max(800, args.memory_ctx)),
            "--timeout",
            str(max(30, args.timeout)),
            "--chunk-chars",
            str(max(500, args.chunk_chars)),
            "--memory-answer-mode",
            "reasoned-chat",
            "--memory-dwell-mode",
            "reasoned",
            "--reasoning-dwell-ctx",
            str(max(256, args.reasoning_dwell_ctx)),
            "--reasoning-num-predict",
            str(max(32, args.reasoning_num_predict)),
            "--json-out",
            str(compare_json),
        ]
        if args.difficulty:
            compare_cmd.extend(["--difficulty", args.difficulty])
        if args.enable_ollama_think:
            compare_cmd.append("--enable-ollama-think")
        if args.no_progress:
            compare_cmd.append("--no-progress")

        memory_json = run_dir / "memory-only-scaled.json"
        memory_cmd = [
            sys.executable,
            "-m",
            "benchmarks.longbench_v2_memory_only",
            "--sample-size",
            str(max(1, args.sample_size)),
            "--seed",
            str(args.seed),
            "--lengths",
            args.lengths,
            "--max-context-chars",
            str(max(20000, args.max_context_chars)),
            "--model",
            args.model,
            "--memory-ctx",
            str(max(800, args.memory_ctx)),
            "--timeout",
            str(max(30, args.timeout)),
            "--chunk-chars",
            str(max(500, args.chunk_chars)),
            "--memory-answer-mode",
            "reasoned-chat",
            "--memory-dwell-mode",
            "reasoned",
            "--reasoning-dwell-ctx",
            str(max(256, args.reasoning_dwell_ctx)),
            "--reasoning-num-predict",
            str(max(32, args.reasoning_num_predict)),
            "--reasoning-predict-multiplier",
            str(max(0.1, args.reasoning_predict_multiplier)),
            "--json-out",
            str(memory_json),
        ]
        if args.difficulty:
            memory_cmd.extend(["--difficulty", args.difficulty])
        if args.enable_ollama_think:
            memory_cmd.append("--enable-ollama-think")
        if args.no_progress:
            memory_cmd.append("--no-progress")

        results: dict[str, Any] = {"run_dir": str(run_dir)}
        if not args.skip_compare:
            code = _run_and_tee(compare_cmd, repo_root, env, run_dir / "compare-direct-vs-reasoned.log")
            if code != 0:
                raise RuntimeError("Direct vs reasoned compare run failed")
            results["compare_json"] = str(compare_json)
        if not args.skip_memory_only:
            code = _run_and_tee(memory_cmd, repo_root, env, run_dir / "memory-only-scaled.log")
            if code != 0:
                raise RuntimeError("Memory-only scaled run failed")
            results["memory_only_json"] = str(memory_json)
        _write_manifest(run_dir / "run-results.json", results)
        print(json.dumps(results, indent=2))
    finally:
        if server_process is not None:
            _stop_process(server_process)


if __name__ == "__main__":
    main()

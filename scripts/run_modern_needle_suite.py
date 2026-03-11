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


def _run_checked(command: list[str], cwd: Path, env: dict[str, str], log_path: Path, label: str) -> None:
    code = _run_and_tee(command, cwd, env, log_path)
    if code != 0:
        raise RuntimeError(f"{label} failed")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _wait_for_url(url: str, timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    request = Request(url, method="GET")
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
    raise RuntimeError(f"Server did not become ready within {timeout_s}s: {last_error}")


def _start_process(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(command, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_mrcr(repo_root: Path, run_dir: Path, env: dict[str, str], args: argparse.Namespace) -> Path:
    output = run_dir / "mrcr-compare.json"
    command = [
        sys.executable,
        "-m",
        "benchmarks.mrcr_compare",
        "--sample-size",
        str(max(1, args.mrcr_sample_size)),
        "--seed",
        str(args.seed),
        "--needles",
        args.mrcr_needles,
        "--model",
        args.model,
        "--direct-ctx",
        str(max(8192, args.direct_ctx)),
        "--memory-ctx",
        str(max(800, args.memory_ctx)),
        "--timeout",
        str(max(30, args.timeout)),
        "--backend-provider",
        args.backend_provider,
        "--api-base",
        args.api_base,
        "--api-key",
        args.api_key,
        "--json-out",
        str(output),
    ]
    code = _run_and_tee(command, repo_root, env, run_dir / "mrcr-compare.log")
    if code != 0:
        raise RuntimeError("MRCR run failed")
    return output


def _run_ruler(repo_root: Path, run_dir: Path, env: dict[str, str], args: argparse.Namespace) -> Path:
    external_root = repo_root / args.external_root / "RULER"
    scripts_root = external_root / "scripts"
    result_root = run_dir / "ruler"
    data_root = result_root / "data"
    pred_root = result_root / "pred"
    server = _start_process(
        [
            sys.executable,
            "pred/serve_vllm.py",
            "--model",
            args.hf_model_id,
            "--tensor-parallel-size",
            str(max(1, args.vllm_tensor_parallel_size)),
            "--dtype",
            "bfloat16",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.ruler_port),
            "--disable-custom-all-reduce",
        ],
        scripts_root,
        env,
        run_dir / "ruler-server.log",
    )
    try:
        _wait_for_url(f"http://127.0.0.1:{args.ruler_port}/health", timeout_s=180)
        for seq_length in [int(item.strip()) for item in args.ruler_seq_lengths.split(",") if item.strip()]:
            for task in [item.strip() for item in args.ruler_tasks.split(",") if item.strip()]:
                _run_checked(
                    [
                        sys.executable,
                        "data/prepare.py",
                        "--save_dir",
                        str(data_root / str(seq_length)),
                        "--benchmark",
                        "synthetic",
                        "--task",
                        task,
                        "--tokenizer_path",
                        args.hf_model_id,
                        "--tokenizer_type",
                        "hf",
                        "--max_seq_length",
                        str(seq_length),
                        "--model_template_type",
                        "base",
                        "--num_samples",
                        str(max(1, args.ruler_sample_size)),
                    ],
                    scripts_root,
                    env,
                    run_dir / f"ruler-prepare-{task}-{seq_length}.log",
                    f"RULER prepare failed for {task} at {seq_length}",
                )
                _run_checked(
                    [
                        sys.executable,
                        "pred/call_api.py",
                        "--data_dir",
                        str(data_root / str(seq_length)),
                        "--save_dir",
                        str(pred_root / str(seq_length)),
                        "--benchmark",
                        "synthetic",
                        "--task",
                        task,
                        "--server_type",
                        "vllm",
                        "--server_host",
                        "127.0.0.1",
                        "--server_port",
                        str(args.ruler_port),
                        "--model_name_or_path",
                        args.hf_model_id,
                        "--temperature",
                        "0",
                        "--top_k",
                        "32",
                        "--top_p",
                        "1.0",
                        "--batch_size",
                        "1",
                    ],
                    scripts_root,
                    env,
                    run_dir / f"ruler-predict-{task}-{seq_length}.log",
                    f"RULER predict failed for {task} at {seq_length}",
                )
            _run_checked(
                [
                    sys.executable,
                    "eval/evaluate.py",
                    "--data_dir",
                    str(pred_root / str(seq_length)),
                    "--benchmark",
                    "synthetic",
                ],
                scripts_root,
                env,
                run_dir / f"ruler-eval-{seq_length}.log",
                f"RULER eval failed at {seq_length}",
            )
    finally:
        _stop_process(server)
    manifest = {
        "benchmark": "RULER",
        "tasks": [item.strip() for item in args.ruler_tasks.split(",") if item.strip()],
        "seq_lengths": [int(item.strip()) for item in args.ruler_seq_lengths.split(",") if item.strip()],
        "result_root": str(result_root),
    }
    output = run_dir / "ruler-summary.json"
    _write_json(output, manifest)
    return output


def _run_nolima(repo_root: Path, run_dir: Path, env: dict[str, str], args: argparse.Namespace) -> Path:
    external_root = repo_root / args.external_root / "NoLiMa"
    evaluation_root = external_root / "evaluation"
    server = _start_process(
        [
            "vllm",
            "serve",
            args.hf_model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.nolima_port),
            "--api-key",
            args.api_key,
            "--tensor-parallel-size",
            str(max(1, args.vllm_tensor_parallel_size)),
            "--gpu-memory-utilization",
            str(args.vllm_gpu_memory_utilization),
            "--max-model-len",
            str(max(4096, args.nolima_max_model_len)),
            "--disable-log-stats",
        ],
        repo_root,
        env,
        run_dir / "nolima-server.log",
    )
    try:
        _wait_for_url(f"http://127.0.0.1:{args.nolima_port}/v1/models", timeout_s=180)
        outputs: list[str] = []
        for context_length in [int(item.strip()) for item in args.nolima_context_lengths.split(",") if item.strip()]:
            context_tag = f"{context_length//1000}K" if context_length >= 1000 else str(context_length)
            model_config_path = run_dir / f"nolima-model-{context_tag}.json"
            run_config_path = run_dir / f"nolima-run-{context_tag}.yaml"
            results_dir = run_dir / "nolima-results" / context_tag
            model_config = {
                "model": args.hf_model_id,
                "api_key": args.api_key,
                "api_url": f"http://127.0.0.1:{args.nolima_port}/v1",
                "api_provider": "vllm",
                "max_tokens": max(192, args.nolima_max_tokens),
                "temperature": 0.0,
                "top_p": 1.0,
                "timeout": max(60, args.timeout),
                "max_retries": 3,
            }
            run_config = "\n".join(
                [
                    f'model_name: "{model_config_path.stem}"',
                    f'model_configs_dir: "{run_dir.as_posix()}/"',
                    f'needle_set_path: "{(external_root / "data" / "needlesets" / args.nolima_needle_set).as_posix()}"',
                    f'haystack_dir: "{(external_root / "data" / args.nolima_haystack_dir).as_posix()}"',
                    f'parent_results_dir: "{results_dir.as_posix()}/"',
                    f"context_length: {context_length}",
                    "document_depth_percent_min: 0",
                    "document_depth_percent_max: 100",
                    f"document_depth_percent_intervals: {max(2, args.nolima_depth_intervals)}",
                    "metric: contains",
                    "use_default_system_prompt: true",
                ]
            )
            model_config_path.write_text(json.dumps(model_config, indent=2) + "\n", encoding="utf-8")
            run_config_path.write_text(run_config + "\n", encoding="utf-8")
            code = _run_and_tee(
                [
                    sys.executable,
                    "run_tests.py",
                    "--config",
                    str(run_config_path),
                ],
                evaluation_root,
                env,
                run_dir / f"nolima-{context_tag}.log",
            )
            if code != 0:
                raise RuntimeError(f"NoLiMa run failed for context length {context_length}")
            outputs.append(str(results_dir))
    finally:
        _stop_process(server)
    output = run_dir / "nolima-summary.json"
    _write_json(
        output,
        {
            "benchmark": "NoLiMa",
            "context_lengths": [int(item.strip()) for item in args.nolima_context_lengths.split(",") if item.strip()],
            "results_roots": outputs,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a modern needle benchmark suite spanning MRCR, RULER, and NoLiMa.")
    parser.add_argument("--model", type=str, default="qwen3.5:0.8b", help="Model name for internal MRCR runs.")
    parser.add_argument("--hf-model-id", type=str, default="", help="Hugging Face model id for vLLM-backed official benchmarks.")
    parser.add_argument("--sample-size", type=int, default=12, help="Default sample size for internal runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--direct-ctx", type=int, default=262144)
    parser.add_argument("--memory-ctx", type=int, default=2200)
    parser.add_argument("--backend-provider", type=str, choices=["ollama", "openai", "openai-compatible", "vllm"], default="ollama")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--external-root", type=str, default="external_benchmarks")
    parser.add_argument("--runs-root", type=str, default="runs/modern-needle")
    parser.add_argument("--run-mrcr", action="store_true")
    parser.add_argument("--run-ruler", action="store_true")
    parser.add_argument("--run-nolima", action="store_true")
    parser.add_argument("--mrcr-sample-size", type=int, default=12)
    parser.add_argument("--mrcr-needles", type=str, default="8")
    parser.add_argument("--ruler-sample-size", type=int, default=20)
    parser.add_argument("--ruler-tasks", type=str, default="niah_single_1,niah_single_2,niah_multikey_2")
    parser.add_argument("--ruler-seq-lengths", type=str, default="4096,8192,16384")
    parser.add_argument("--ruler-port", type=int, default=5000)
    parser.add_argument("--nolima-context-lengths", type=str, default="4000,8000,16000,32000")
    parser.add_argument("--nolima-needle-set", type=str, default="needle_set.json")
    parser.add_argument("--nolima-haystack-dir", type=str, default="haystack/rand_shuffle")
    parser.add_argument("--nolima-depth-intervals", type=int, default=26)
    parser.add_argument("--nolima-max-model-len", type=int, default=40000)
    parser.add_argument("--nolima-max-tokens", type=int, default=384)
    parser.add_argument("--nolima-port", type=int, default=8000)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.95)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / args.runs_root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    shim_path = str((repo_root / "third_party_shims").resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = shim_path if not existing_pythonpath else shim_path + os.pathsep + existing_pythonpath

    manifest = {
        "model": args.model,
        "hf_model_id": args.hf_model_id,
        "backend_provider": args.backend_provider,
        "api_base": args.api_base,
        "run_mrcr": bool(args.run_mrcr),
        "run_ruler": bool(args.run_ruler),
        "run_nolima": bool(args.run_nolima),
    }
    _write_json(run_dir / "manifest.json", manifest)

    outputs: dict[str, str] = {}
    if args.run_mrcr:
        outputs["mrcr"] = str(_run_mrcr(repo_root, run_dir, env, args))
    if args.run_ruler:
        if not args.hf_model_id:
            raise ValueError("--hf-model-id is required for --run-ruler")
        outputs["ruler"] = str(_run_ruler(repo_root, run_dir, env, args))
    if args.run_nolima:
        if not args.hf_model_id:
            raise ValueError("--hf-model-id is required for --run-nolima")
        outputs["nolima"] = str(_run_nolima(repo_root, run_dir, env, args))

    _write_json(run_dir / "run-results.json", outputs)
    print(json.dumps({"run_dir": str(run_dir), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def _download_hf(model_id: str, local_dir: str, token: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=local_dir or None,
        token=token or None,
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark models for either Ollama or Hugging Face/vLLM.")
    parser.add_argument("--ollama-model", action="append", default=[], help="Ollama model tag to pull, for example qwen3.5:0.8b")
    parser.add_argument("--hf-model", action="append", default=[], help="Hugging Face model id to snapshot download, for example Qwen/Qwen3.5-0.8B")
    parser.add_argument("--hf-local-dir", type=str, default="", help="Optional destination directory for Hugging Face snapshots.")
    parser.add_argument("--hf-token", type=str, default="", help="Optional Hugging Face token for gated models.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    downloaded_hf: dict[str, str] = {}

    for model in args.ollama_model:
        if model.strip():
            _run(["ollama", "pull", model.strip()], repo_root)

    if args.hf_model:
        try:
            from huggingface_hub import snapshot_download  # noqa: F401
        except Exception as err:  # pragma: no cover - dependency guard
            raise RuntimeError("huggingface_hub is required for --hf-model downloads") from err
        for model in args.hf_model:
            if model.strip():
                downloaded_hf[model.strip()] = _download_hf(model.strip(), args.hf_local_dir, args.hf_token)

    if downloaded_hf:
        for model_id, path in downloaded_hf.items():
            print(f"{model_id} -> {path}")
    else:
        print("No Hugging Face models requested.")


if __name__ == "__main__":
    main()

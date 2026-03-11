from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen


RULER_REPO = "https://github.com/NVIDIA/RULER.git"
NOLIMA_REPO = "https://github.com/adobe-research/NoLiMa.git"
NOLIMA_URLS = [
    ("data/needlesets/needle_set.json", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set.json"),
    ("data/needlesets/needle_set_MC.json", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set_MC.json"),
    ("data/needlesets/needle_set_ONLYDirect.json", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set_ONLYDirect.json"),
    ("data/needlesets/needle_set_hard.json", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set_hard.json"),
    ("data/needlesets/needle_set_w_CoT.json", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set_w_CoT.json"),
    ("data/needlesets/needle_set_w_Distractor.json", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/needlesets/needle_set_w_Distractor.json"),
    ("data/haystack/rand_shuffle/rand_book_1.txt", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_1.txt"),
    ("data/haystack/rand_shuffle/rand_book_2.txt", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_2.txt"),
    ("data/haystack/rand_shuffle/rand_book_3.txt", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_3.txt"),
    ("data/haystack/rand_shuffle/rand_book_4.txt", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_4.txt"),
    ("data/haystack/rand_shuffle/rand_book_5.txt", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle/rand_book_5.txt"),
    ("data/haystack/rand_shuffle_long/rand_book_1.txt", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_1.txt"),
    ("data/haystack/rand_shuffle_long/rand_book_2.txt", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_2.txt"),
    ("data/haystack/rand_shuffle_long/rand_book_3.txt", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_3.txt"),
    ("data/haystack/rand_shuffle_long/rand_book_4.txt", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_4.txt"),
    ("data/haystack/rand_shuffle_long/rand_book_5.txt", "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/rand_shuffle_long/rand_book_5.txt"),
]


def _run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def _git_clone_or_update(repo_url: str, dest: Path) -> str:
    if dest.exists():
        _run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin"], dest.parent)
        _run(["git", "-C", str(dest), "pull", "--ff-only"], dest.parent)
    else:
        _run(["git", "clone", "--depth", "1", repo_url, str(dest)], dest.parent)
    commit = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    return commit


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with urlopen(url, timeout=120) as response:
        dest.write_bytes(response.read())


def _setup_ruler_data(ruler_root: Path, include_qa: bool) -> None:
    json_root = ruler_root / "scripts" / "data" / "synthetic" / "json"
    _run([sys.executable, "download_paulgraham_essay.py"], json_root)
    if include_qa:
        _download(
            "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
            json_root / "squad.json",
        )
        _download(
            "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            json_root / "hotpotqa.json",
        )


def _setup_nolima_data(nolima_root: Path) -> None:
    for rel_path, url in NOLIMA_URLS:
        _download(url, nolima_root / rel_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone and prepare modern needle benchmarks (RULER and NoLiMa).")
    parser.add_argument("--external-root", type=str, default="external_benchmarks")
    parser.add_argument("--skip-ruler", action="store_true")
    parser.add_argument("--skip-nolima", action="store_true")
    parser.add_argument("--skip-ruler-data", action="store_true")
    parser.add_argument("--skip-nolima-data", action="store_true")
    parser.add_argument("--include-ruler-qa", action="store_true", help="Also download the optional QA datasets used by non-NIAH RULER tasks.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    external_root = repo_root / args.external_root
    external_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {"external_root": str(external_root)}

    if not args.skip_ruler:
        ruler_root = external_root / "RULER"
        manifest["ruler_commit"] = _git_clone_or_update(RULER_REPO, ruler_root)
        if not args.skip_ruler_data:
            _setup_ruler_data(ruler_root, include_qa=bool(args.include_ruler_qa))
        manifest["ruler_root"] = str(ruler_root)

    if not args.skip_nolima:
        nolima_root = external_root / "NoLiMa"
        manifest["nolima_commit"] = _git_clone_or_update(NOLIMA_REPO, nolima_root)
        if not args.skip_nolima_data:
            _setup_nolima_data(nolima_root)
        manifest["nolima_root"] = str(nolima_root)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

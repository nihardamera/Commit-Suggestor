"""Publish the model and the demo Space to the Hugging Face Hub.

    uv run python scripts/publish_hf.py --user HF_USER              # dry run: lists every file
    uv run python scripts/publish_hf.py --user HF_USER --yes        # publish (public)

Model repo HF_USER/commit-suggestor:
    README.md                          docs/MODEL_CARD.md (with the Hub's metadata header)
    mlx/                               the MLX 4-bit model, for Apple Silicon
    commit-suggestor-q4_k_m.gguf/.json the GGUF build, for llama.cpp
Space HF_USER/commit-suggestor-demo (static, built by scripts/static_space.py):
    index.html, README.md

Hugging Face hosts static Spaces for free but needs PRO for Gradio ones, so the
public demo shows stored test-set subjects; space/app.py is the live version,
run locally. The CLI downloads the model at the revision this run creates,
which is printed at the end.

    --part model | space     publish only one of them
The token is read from the macOS Keychain (service "hf-token") or HF_TOKEN.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CARD_HEADER = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: mlx
pipeline_tag: text-generation
language: [en]
tags: [git, commit-messages, code, lora, mlx, gguf]
---

"""


def token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    return subprocess.run(
        ["security", "find-generic-password", "-s", "hf-token", "-w"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def model_files(mlx: Path, gguf: Path) -> dict[str, Path]:
    # mlx_lm convert writes a stub README.md; the model card is the only one.
    files = {
        f"mlx/{p.name}": p for p in sorted(mlx.iterdir()) if p.is_file() and p.name != "README.md"
    }
    files[gguf.name] = gguf
    files[gguf.with_suffix(".json").name] = gguf.with_suffix(".json")
    return files


def space_files() -> dict[str, Path]:
    site = ROOT / "data" / "static-space"
    build = [sys.executable, str(ROOT / "scripts" / "static_space.py"), str(site)]
    subprocess.run(build, check=True)
    return {name: site / name for name in ("index.html", "README.md")}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--user", required=True, help="Hugging Face user name")
    parser.add_argument("--mlx", type=Path, default=ROOT / "data/models/dpo-mlx")
    parser.add_argument(
        "--gguf",
        type=Path,
        default=ROOT / "data/kaggle-output/gguf/full/commit-suggestor-q4_k_m.gguf",
    )
    parser.add_argument("--part", choices=["model", "space", "both"], default="both")
    parser.add_argument("--yes", action="store_true", help="publish; without it, a dry run")
    args = parser.parse_args()

    model_repo = f"{args.user}/commit-suggestor"
    space_repo = f"{args.user}/commit-suggestor-demo"
    card = ROOT / "docs" / "MODEL_CARD.md"
    models = model_files(args.mlx, args.gguf) if args.part != "space" else {}
    spaces = space_files() if args.part != "model" else {}
    needed = [card, *models.values()] if models else []
    missing = [str(p) for p in [*needed, *spaces.values()] if not p.exists()]
    if missing:
        sys.exit("missing: " + ", ".join(missing))
    if models:
        print(f"model {model_repo}: README.md + {len(models)} files")
    for name, path in models.items():
        print(f"  {name:40} {path.stat().st_size / 1e6:8.1f} MB")
    if spaces:
        print(f"space {space_repo}: {', '.join(spaces)}")
    if not args.yes:
        print("dry run: nothing was uploaded. Re-run with --yes to publish (public).")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token())
    if models:
        api.create_repo(model_repo, repo_type="model", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            (stage / "README.md").write_text(CARD_HEADER + card.read_text())
            for name, path in models.items():
                (stage / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, stage / name)
            commit = api.upload_folder(
                repo_id=model_repo, folder_path=stage, commit_message="Publish Commit Suggestor v2"
            )
        print(f"model: https://huggingface.co/{model_repo} at revision {commit.oid}")
        print(f'set HF_REVISION = "{commit.oid}" in commit_suggestor/cli.py')
    if spaces:
        api.create_repo(space_repo, repo_type="space", space_sdk="static", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            for name, path in spaces.items():
                shutil.copy(path, stage / name)
            api.upload_folder(
                repo_id=space_repo,
                repo_type="space",
                folder_path=stage,
                commit_message="Publish the demo",
            )
        print(f"space: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()

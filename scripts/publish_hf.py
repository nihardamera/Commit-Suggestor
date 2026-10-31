"""Publish the model and the demo Space to the Hugging Face Hub.

    uv run python scripts/publish_hf.py --user HF_USER              # dry run: lists every file
    uv run python scripts/publish_hf.py --user HF_USER --yes        # publish (public)

Model repo HF_USER/commit-suggestor:
    README.md                          docs/MODEL_CARD.md (with the Hub's metadata header)
    mlx/                               the MLX 4-bit model, for Apple Silicon
    commit-suggestor-q4_k_m.gguf/.json the GGUF build, for llama.cpp
Space HF_USER/commit-suggestor-demo:
    app.py, README.md, requirements.txt (space/), and commit_suggestor/{prompt,prepare}.py

The Space and the CLI download the model at the revision this run creates,
which is printed at the end and set on the Space as MODEL_REVISION.
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
    files = {f"mlx/{p.name}": p for p in sorted(mlx.iterdir()) if p.is_file()}
    files[gguf.name] = gguf
    files[gguf.with_suffix(".json").name] = gguf.with_suffix(".json")
    return files


def space_files() -> dict[str, Path]:
    files = {name: ROOT / "space" / name for name in ("app.py", "README.md", "requirements.txt")}
    for name in ("__init__.py", "prompt.py", "prepare.py"):
        files[f"commit_suggestor/{name}"] = ROOT / "commit_suggestor" / name
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--user", required=True, help="Hugging Face user name")
    parser.add_argument("--mlx", type=Path, default=ROOT / "data/models/dpo-mlx")
    parser.add_argument(
        "--gguf",
        type=Path,
        default=ROOT / "data/kaggle-output/gguf/full/commit-suggestor-q4_k_m.gguf",
    )
    parser.add_argument("--yes", action="store_true", help="publish; without it, a dry run")
    args = parser.parse_args()

    model_repo = f"{args.user}/commit-suggestor"
    space_repo = f"{args.user}/commit-suggestor-demo"
    card = ROOT / "docs" / "MODEL_CARD.md"
    models, spaces = model_files(args.mlx, args.gguf), space_files()
    missing = [str(p) for p in [card, *models.values(), *spaces.values()] if not p.exists()]
    if missing:
        sys.exit("missing: " + ", ".join(missing))
    print(f"model {model_repo}: README.md + {len(models)} files")
    for name, path in models.items():
        print(f"  {name:40} {path.stat().st_size / 1e6:8.1f} MB")
    print(f"space {space_repo}: {', '.join(spaces)}")
    if not args.yes:
        print("dry run: nothing was uploaded. Re-run with --yes to publish (public).")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token())
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
    revision = commit.oid
    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(space_repo, "MODEL_REPO", model_repo)
    api.add_space_variable(space_repo, "MODEL_REVISION", revision)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name, path in spaces.items():
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, stage / name)
        api.upload_folder(
            repo_id=space_repo,
            repo_type="space",
            folder_path=stage,
            commit_message="Publish the demo",
        )
    print(f"model: https://huggingface.co/{model_repo} at revision {revision}")
    print(f"space: https://huggingface.co/spaces/{space_repo}")
    print(f'set HF_REVISION = "{revision}" in commit_suggestor/cli.py')


if __name__ == "__main__":
    main()

"""Upload the v2 data to Kaggle, run notebooks/sft_v2.py on a GPU, fetch the results.

    uv run python scripts/kaggle_job.py dataset            # create or update the private dataset
    uv run python scripts/kaggle_job.py push --mode smoke  # 20 steps, to catch errors cheaply
    uv run python scripts/kaggle_job.py status
    uv run python scripts/kaggle_job.py output --mode smoke

The Kaggle token is read from the macOS Keychain (service "kaggle-api-token")
or from KAGGLE_API_TOKEN, and passed to the CLI in its environment only.
The dataset and the job are private.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "nihardamera"
DATASET = f"{USER}/commit-suggestor-v2-data"
KERNEL = f"{USER}/commit-suggestor-sft-v2"
STAGE = ROOT / "data" / "kaggle"
KAGGLE = ["uvx", "--quiet", "kaggle@2.2.5"]


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    token = (
        os.environ.get("KAGGLE_API_TOKEN")
        or subprocess.run(
            ["security", "find-generic-password", "-s", "kaggle-api-token", "-w"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    result = subprocess.run(
        [*KAGGLE, *args],
        env={**os.environ, "KAGGLE_API_TOKEN": token},
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.returncode and check:
        sys.exit(result.stderr.strip() or f"kaggle {args[0]} failed")
    return result


def dataset(_: argparse.Namespace) -> None:
    stage = STAGE / "dataset"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for name in ("train.jsonl", "valid.jsonl", "valid.meta.jsonl"):
        shutil.copy(ROOT / "data" / "v2" / name, stage / name)
    for name in ("test.jsonl", "test_context.jsonl"):
        shutil.copy(ROOT / "eval" / name, stage / name)
    # The training job builds generation prompts with the repository's own code.
    shutil.copy(ROOT / "commit_suggestor" / "prompt.py", stage / "prompt.py")
    (stage / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Commit Suggestor v2 data",
                "id": DATASET,
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
    )
    exists = kaggle("datasets", "status", DATASET, check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(stage), "-m", "Rebuilt v2 data")
    else:
        kaggle("datasets", "create", "-p", str(stage))  # private unless --public


def push(args: argparse.Namespace) -> None:
    stage = STAGE / "kernel"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    source = (ROOT / "notebooks" / "sft_v2.py").read_text()
    marker = 'MODE = "smoke"'
    if marker not in source:
        sys.exit("notebooks/sft_v2.py no longer sets MODE as expected")
    (stage / "sft_v2.py").write_text(source.replace(marker, f'MODE = "{args.mode}"', 1))
    (stage / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": KERNEL,
                "title": "commit-suggestor-sft-v2",
                "code_file": "sft_v2.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [DATASET],
            },
            indent=2,
        )
    )
    kaggle("kernels", "push", "-p", str(stage))


def status(_: argparse.Namespace) -> None:
    kaggle("kernels", "status", KERNEL)


def output(args: argparse.Namespace) -> None:
    target = ROOT / "data" / "kaggle-output" / args.mode
    target.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", KERNEL, "-p", str(target), "-o")
    print(f"downloaded to {target.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dataset").set_defaults(run=dataset)
    p = sub.add_parser("push")
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.set_defaults(run=push)
    sub.add_parser("status").set_defaults(run=status)
    o = sub.add_parser("output")
    o.add_argument("--mode", choices=["smoke", "full"], required=True)
    o.set_defaults(run=output)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

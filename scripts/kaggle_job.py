"""Upload the v2 data to Kaggle, run the GPU jobs in notebooks/, fetch the results.

    uv run python scripts/kaggle_job.py dataset [--which pairs]         # private datasets
    uv run python scripts/kaggle_job.py push --job sft --mode smoke     # a short run first
    uv run python scripts/kaggle_job.py status --job sft
    uv run python scripts/kaggle_job.py output --job sft --mode full

Jobs: sft (notebooks/sft_v2.py), sample (dpo_sample.py: candidates for DPO
pairs), dpo (dpo_train.py) and gguf (gguf.py, CPU: the llama.cpp build).
`--job` defaults to sft.

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
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
USER = "nihardamera"
DATASET = f"{USER}/commit-suggestor-v2-data"
PAIRS_DATASET = f"{USER}/commit-suggestor-dpo-pairs"
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


def upload(slug: str, title: str, files: dict[str, Path]) -> None:
    """Create the private dataset `slug`, or add a new version of it."""
    stage = STAGE / slug.split("/", 1)[1]
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for name, source in files.items():
        shutil.copy(source, stage / name)
    (stage / "dataset-metadata.json").write_text(
        json.dumps({"title": title, "id": slug, "licenses": [{"name": "other"}]}, indent=2)
    )
    if kaggle("datasets", "status", slug, check=False).returncode == 0:
        kaggle("datasets", "version", "-p", str(stage), "-m", "Rebuilt")
    else:
        kaggle("datasets", "create", "-p", str(stage))  # private unless --public


def dataset(args: argparse.Namespace) -> None:
    # The jobs import the repository's own code, so it ships with the data: a
    # Kaggle script job uploads only the script.
    if args.which == "v2":
        upload(
            DATASET,
            "Commit Suggestor v2 data",
            {
                **{
                    n: ROOT / "data" / "v2" / n
                    for n in ("train.jsonl", "train.meta.jsonl", "valid.jsonl", "valid.meta.jsonl")
                },
                **{n: ROOT / "eval" / n for n in ("test.jsonl", "test_context.jsonl")},
                "prompt.py": ROOT / "commit_suggestor" / "prompt.py",
            },
        )
    else:
        upload(
            PAIRS_DATASET,
            "Commit Suggestor DPO pairs",
            {
                "pairs.jsonl": ROOT / "data" / "dpo" / "pairs.jsonl",
                "dpo.py": ROOT / "commit_suggestor" / "dpo.py",
            },
        )


#: Each Kaggle job: its script, its kernel, the datasets and earlier jobs it reads.
JOBS: dict[str, dict[str, Any]] = {
    "sft": {
        "code": "sft_v2.py",
        "kernel": f"{USER}/commit-suggestor-sft-v2",
        "datasets": [DATASET],
        "after": [],
    },
    "sample": {
        "code": "dpo_sample.py",
        "kernel": f"{USER}/commit-suggestor-dpo-sample",
        "datasets": [DATASET],
        "after": [f"{USER}/commit-suggestor-sft-v2"],
    },
    "dpo": {
        "code": "dpo_train.py",
        "kernel": f"{USER}/commit-suggestor-dpo-train",
        "datasets": [DATASET, PAIRS_DATASET],
        "after": [f"{USER}/commit-suggestor-sft-v2"],
    },
    # CPU only: converts the shipped model to GGUF and checks it with llama.cpp.
    "gguf": {
        "code": "gguf.py",
        "kernel": f"{USER}/commit-suggestor-gguf",
        "datasets": [DATASET],
        "after": [f"{USER}/commit-suggestor-sft-v2"],  # + the DPO job for a DPO --source
        "gpu": False,
    },
}


def push(args: argparse.Namespace) -> None:
    job = JOBS[args.job]
    code = job["code"]
    stage = STAGE / f"kernel-{args.job}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    source = (ROOT / "notebooks" / code).read_text()
    marker = 'MODE = "smoke"'
    if marker not in source:
        sys.exit(f"notebooks/{code} no longer sets MODE as expected")
    source = source.replace(marker, f'MODE = "{args.mode}"', 1)
    if args.source:
        if 'SOURCE = "sft"' not in source:
            sys.exit(f"notebooks/{code} has no SOURCE to set")
        source = source.replace('SOURCE = "sft"', f'SOURCE = "{args.source}"', 1)
    (stage / code).write_text(source)
    (stage / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": job["kernel"],
                "title": job["kernel"].split("/", 1)[1],
                "code_file": code,
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": job.get("gpu", True),
                "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": job["datasets"],
                "kernel_sources": job["after"]
                + ([JOBS["dpo"]["kernel"]] if args.source not in (None, "sft") else []),
            },
            indent=2,
        )
    )
    kaggle("kernels", "push", "-p", str(stage))


def status(args: argparse.Namespace) -> None:
    kaggle("kernels", "status", str(JOBS[args.job]["kernel"]))


def output(args: argparse.Namespace) -> None:
    target = ROOT / "data" / "kaggle-output" / args.job / args.mode
    target.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", str(JOBS[args.job]["kernel"]), "-p", str(target), "-o")
    print(f"downloaded to {target.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--which", choices=["v2", "pairs"], default="v2")
    d.set_defaults(run=dataset)
    p = sub.add_parser("push")
    p.add_argument("--job", choices=sorted(JOBS), default="sft")
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--source", help='gguf job only: "sft" or a DPO adapter, e.g. beta-0.1')
    p.set_defaults(run=push)
    st = sub.add_parser("status")
    st.add_argument("--job", choices=sorted(JOBS), default="sft")
    st.set_defaults(run=status)
    o = sub.add_parser("output")
    o.add_argument("--job", choices=sorted(JOBS), default="sft")
    o.add_argument("--mode", choices=["smoke", "full"], required=True)
    o.set_defaults(run=output)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

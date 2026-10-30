"""Upload the v2 data to Kaggle, run the GPU jobs in notebooks/, fetch the results.

    uv run python scripts/kaggle_job.py dataset                         # private dataset
    uv run python scripts/kaggle_job.py push --job sft --mode smoke     # a short run first
    uv run python scripts/kaggle_job.py status --job sft
    uv run python scripts/kaggle_job.py output --job sft --mode full

Jobs: sft (notebooks/sft_v2.py), sample (dpo_sample.py: candidates for DPO
pairs) and dpo (dpo_train.py). `--job` defaults to sft.

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


#: Each Kaggle job: its script, its kernel, and the outputs of earlier jobs it reads.
JOBS: dict[str, dict[str, object]] = {
    "sft": {"code": "sft_v2.py", "kernel": f"{USER}/commit-suggestor-sft-v2", "after": []},
    "sample": {
        "code": "dpo_sample.py",
        "kernel": f"{USER}/commit-suggestor-dpo-sample",
        "after": [f"{USER}/commit-suggestor-sft-v2"],
    },
    "dpo": {
        "code": "dpo_train.py",
        "kernel": f"{USER}/commit-suggestor-dpo-train",
        "after": [f"{USER}/commit-suggestor-sft-v2"],
    },
}


def push(args: argparse.Namespace) -> None:
    job = JOBS[args.job]
    code = str(job["code"])
    stage = STAGE / f"kernel-{args.job}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    source = (ROOT / "notebooks" / code).read_text()
    marker = 'MODE = "smoke"'
    if marker not in source:
        sys.exit(f"notebooks/{code} no longer sets MODE as expected")
    (stage / code).write_text(source.replace(marker, f'MODE = "{args.mode}"', 1))
    for extra in args.extra_file or []:
        shutil.copy(extra, stage / Path(extra).name)
    kernel = str(job["kernel"])
    (stage / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel,
                "title": kernel.split("/", 1)[1],
                "code_file": code,
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [DATASET, *args.dataset],
                "kernel_sources": job["after"],
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
    sub.add_parser("dataset").set_defaults(run=dataset)
    p = sub.add_parser("push")
    p.add_argument("--job", choices=sorted(JOBS), default="sft")
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="another private dataset the job reads, user/slug",
    )
    p.add_argument("--extra-file", action="append", help="a file to ship next to the script")
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

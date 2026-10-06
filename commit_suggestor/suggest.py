"""Suggest a commit subject line for a diff.

    python -m commit_suggestor.suggest           # the staged changes (git diff --cached)
    git diff HEAD~1 | python -m commit_suggestor.suggest -   # any diff on stdin

The diff is cleaned with the same function the training data went through, and
the prompt is the one the model was trained on, so what it sees in use matches
what it learned from.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from commit_suggestor.prepare import clean_diff
from commit_suggestor.prompt import build_messages, clean_reply

BASE_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
DEFAULT_ADAPTER = Path(__file__).resolve().parent.parent / "adapters"


def staged_diff() -> str:
    result = subprocess.run(["git", "diff", "--cached", "--no-color", "-M"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(result.stderr.strip() or "not a git repository")
    return result.stdout


def suggest(diff: str, adapter: Path | None = DEFAULT_ADAPTER) -> str:
    # A command-line tool should print the suggestion, not a download bar.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    cleaned, files = clean_diff(diff)
    if not files:
        cleaned = diff  # e.g. a diff without `diff --git` headers; use it as given
    model, tokenizer = load(BASE_MODEL,
                            adapter_path=str(adapter) if adapter else None)
    prompt = tokenizer.apply_chat_template(build_messages(cleaned),
                                           add_generation_prompt=True, tokenize=False)
    reply = generate(model, tokenizer, prompt=prompt, max_tokens=40,
                     sampler=make_sampler(temp=0.0), verbose=False)
    return clean_reply(reply)


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest a commit subject line.")
    parser.add_argument("source", nargs="?", help="'-' to read a diff from stdin")
    parser.add_argument("--adapter", type=Path,
                        default=Path(os.environ.get("COMMIT_SUGGESTOR_ADAPTER", DEFAULT_ADAPTER)))
    parser.add_argument("--base", action="store_true",
                        help="use the base model without the fine-tuned adapter")
    args = parser.parse_args()

    diff = sys.stdin.read() if args.source == "-" else staged_diff()
    if not diff.strip():
        sys.exit("nothing to describe: no staged changes (git add something first)")
    adapter = None if args.base else args.adapter
    if adapter and not (adapter / "adapters.safetensors").exists():
        sys.exit(f"no adapter at {adapter}; run scripts/get_adapter.sh, or pass --base")
    print(suggest(diff, adapter))


if __name__ == "__main__":
    main()

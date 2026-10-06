"""Filter mined commits into training, validation and test sets.

Most commits are poor training examples, and the filters are where the
quality of the model is actually decided:

- Bots, merges, reverts, releases and version bumps are dropped. Their
  subjects are mechanical or say nothing about the diff.
- Pull-request numbers like "(#1234)" are stripped from subjects. They cannot
  be inferred from a diff, so leaving them in teaches the model to invent them.
- Commits touching more than four files, or with very large diffs, are dropped.
  The subject of a sprawling change rarely describes most of it, and the diff
  would not fit in the model's context anyway.
- Lock files, minified files and binary changes are removed from the diff text.

Output is the chat format `mlx_lm.lora` expects, built with the same prompt
the CLI uses at inference time.

Usage:
    python -m commit_suggestor.prepare --raw data/raw --out data/mlx
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from commit_suggestor.prompt import MAX_DIFF_CHARS, build_messages

MIN_SUBJECT, MAX_SUBJECT = 12, 72
MIN_DIFF, MAX_FILES = 40, 4
#: Test examples are capped so evaluation runs in minutes, sampled across the
#: test repositories rather than taken from the most recent commits.
MAX_TEST, MAX_VALID = 450, 300

BOT = re.compile(r"bot\b|\[bot\]|dependabot|renovate|pre-commit-ci|github-actions", re.I)
REJECT_SUBJECT = re.compile(
    r"^(merge|revert|bump|release|prepare|version|v?\d+\.\d+|wip\b|fixup!|squash!|amend!)"
    r"|pre-commit autoupdate|\[pre-commit\.ci\]|changelog$|^update (history|changes)$",
    re.I)
TRAILING_PR = re.compile(r"\s*\((?:#|gh-)\d+\)\s*$|\s+#\d+\s*$")
LEADING_TAG = re.compile(r"^\s*\[(?:skip ci|ci skip|skip-ci|svn r\d+)\]\s*", re.I)
SKIP_FILE = re.compile(
    r"(\.lock|poetry\.lock|package-lock\.json|yarn\.lock|\.min\.(js|css)|\.svg|\.png|\.jpg|\.ico)$")


def clean_subject(subject: str) -> str:
    subject = LEADING_TAG.sub("", subject).strip()
    subject = TRAILING_PR.sub("", subject).strip()
    return subject.rstrip(".").strip()


def split_files(diff: str) -> list[str]:
    """One chunk per `diff --git` section."""
    parts = re.split(r"(?m)^(?=diff --git )", diff)
    return [p for p in parts if p.startswith("diff --git ")]


def clean_diff(diff: str) -> tuple[str, int]:
    """Drop noise sections and lines; return the text and the file count."""
    kept: list[str] = []
    for section in split_files(diff):
        header = section.splitlines()[0]
        if SKIP_FILE.search(header) or "\nBinary files " in section:
            continue
        lines = [ln for ln in section.splitlines()
                 if not ln.startswith(("index ", "similarity index", "dissimilarity index"))]
        kept.append("\n".join(lines))
    return "\n".join(kept).strip(), len(kept)


def keep(row: dict) -> tuple[bool, str]:
    """Whether a mined commit becomes an example, and why not if it doesn't."""
    if BOT.search(row["author"]) or BOT.search(row["email"]):
        return False, "bot author"
    subject = clean_subject(row["subject"])
    if REJECT_SUBJECT.search(subject):
        return False, "merge/revert/release/bump"
    if not MIN_SUBJECT <= len(subject) <= MAX_SUBJECT:
        return False, "subject length"
    diff, files = clean_diff(row["diff"])
    if files == 0:
        return False, "no text diff"
    if files > MAX_FILES:
        return False, "too many files"
    if not MIN_DIFF <= len(diff) <= MAX_DIFF_CHARS:
        return False, "diff size"
    row["subject_clean"], row["diff_clean"] = subject, diff
    return True, "kept"


def to_example(row: dict) -> dict:
    return {"messages": build_messages(row["diff_clean"])
            + [{"role": "assistant", "content": row["subject_clean"]}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/mlx"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reasons: Counter[str] = Counter()
    per_repo: Counter[str] = Counter()
    seen: set[str] = set()
    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}

    for path in sorted(args.raw.glob("*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            ok, why = keep(row)
            reasons[why] += 1
            if not ok:
                continue
            fingerprint = hashlib.sha1(row["diff_clean"].encode()).hexdigest()
            if fingerprint in seen:
                reasons["duplicate diff"] += 1
                reasons["kept"] -= 1
                continue
            seen.add(fingerprint)
            per_repo[row["repo"]] += 1
            splits[row["split"]].append(row)

    rng = random.Random(args.seed)
    for name in splits:
        rng.shuffle(splits[name])
    splits["test"] = splits["test"][:MAX_TEST]
    splits["valid"] = splits["valid"][:MAX_VALID]

    for name, rows in splits.items():
        with (args.out / f"{name}.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(to_example(row)) + "\n")
        # Keep the provenance for evaluation and for inspecting examples.
        with (args.out / f"{name}.meta.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps({"repo": row["repo"], "sha": row["sha"],
                                     "subject": row["subject_clean"],
                                     "diff": row["diff_clean"]}) + "\n")

    total = sum(reasons.values())
    print(f"mined {total} commits")
    for why, n in reasons.most_common():
        print(f"  {why:28} {n:6}  ({n / total:.0%})")
    print({name: len(rows) for name, rows in splits.items()})
    for repo, n in sorted(per_repo.items(), key=lambda kv: -kv[1]):
        print(f"  {repo:34} {n}")


if __name__ == "__main__":
    main()

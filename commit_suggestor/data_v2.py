"""Build the v2 training data: v1's filters, three more, and repo-style context.

On top of v1's filters (commit_suggestor.prepare.keep) v2 drops:
  - generic subjects ("Fix bug", "Update docs", "Update foo.py"), which teach
    the model to say little (commit_suggestor.metrics.is_generic),
  - subjects not in the imperative mood ("Added x", "Fixes y"), since the
    prompt asks for the imperative,
  - near-duplicate diffs: MinHash over the changed lines, within the training
    data and, as a contamination check, against the validation and test diffs,
and caps each training project's share so a few large projects don't dominate.

Subject case and style are left alone: following a project's own style is
what the repo-style context is for. Each example gets the subjects of the five
commits before it (commit_suggestor.context); in training half the examples
drop the context (fixed per commit), so the model also works without history.

The test set is not rebuilt here. It stays the 450 examples in eval/test.jsonl.

    uv run python -m commit_suggestor.data_v2 --raw data/raw --out data/v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from commit_suggestor.context import N_RECENT, clean_context
from commit_suggestor.metrics import is_generic, is_imperative
from commit_suggestor.mine import REPOS
from commit_suggestor.prepare import keep
from commit_suggestor.prompt import PROMPT_VERSION, build_messages

EVAL = Path(__file__).resolve().parent.parent / "eval"
PER_PROJECT_CAP = 4000
MAX_VALID = 600
CONTEXT_DROPOUT = 0.5
NEAR_DUP_JACCARD = 0.8
#: Below this many shingles a diff changes too little text for similarity to
#: mean anything (two diffs that only add blank lines both have none). Such
#: diffs get the exact-duplicate check only.
MIN_SHINGLES = 5
SEED = 0

# MinHash: 64 hash functions in 16 bands of 4. Two diffs whose shingle sets have
# Jaccard similarity s share a band with probability 1 - (1 - s^4)^16: 0.99 at
# s = 0.8, 0.07 at s = 0.4. Candidates are then checked with the exact Jaccard.
N_HASHES, BANDS = 64, 16
ROWS = N_HASHES // BANDS
# Each shingle is hashed once; the 64 permutations are then (a * x + b) mod p
# for a Mersenne prime p, which numpy computes for all shingles at once.
_PRIME = (1 << 31) - 1
_rng = np.random.default_rng(12345)
_A = _rng.integers(1, _PRIME, size=N_HASHES, dtype=np.int64)
_B = _rng.integers(0, _PRIME, size=N_HASHES, dtype=np.int64)
_EMPTY = tuple([_PRIME] * N_HASHES)


def shingles(diff: str, n: int = 5) -> set[str]:
    """Word 5-grams of the changed lines: what the diff actually does."""
    changed = " ".join(
        line[1:]
        for line in diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )
    words = re.findall(r"\S+", changed)
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _base_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little") % _PRIME


def minhash(items: set[str]) -> tuple[int, ...]:
    if not items:
        return _EMPTY
    x = np.fromiter((_base_hash(s) for s in items), dtype=np.int64, count=len(items))
    # a < 2^31 and x < 2^31, so a * x + b stays below 2^63: no int64 overflow.
    permuted = (_A[:, None] * x[None, :] + _B[:, None]) % _PRIME
    return tuple(int(v) for v in permuted.min(axis=1))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


class NearDuplicateIndex:
    """Finds earlier diffs whose changed lines are at least `threshold` similar."""

    def __init__(self, threshold: float = NEAR_DUP_JACCARD) -> None:
        self.threshold = threshold
        self.buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        self.sets: list[set[str]] = []

    def find(self, items: set[str]) -> int | None:
        """Index of a near-duplicate already added, or None."""
        signature = minhash(items)
        seen: set[int] = set()
        for band in range(BANDS):
            for j in self.buckets.get((band, signature[band * ROWS : (band + 1) * ROWS]), []):
                if j not in seen:
                    seen.add(j)
                    if jaccard(items, self.sets[j]) >= self.threshold:
                        return j
        return None

    def add(self, items: set[str]) -> int:
        signature = minhash(items)
        index = len(self.sets)
        self.sets.append(items)
        for band in range(BANDS):
            self.buckets[(band, signature[band * ROWS : (band + 1) * ROWS])].append(index)
        return index


def with_context(rows: list[dict[str, Any]]) -> None:
    """Attach the subjects of the commits mined after each row (older, newest first).

    `rows` is one repository in `git log` order, so the commits before row i are
    rows i+1, i+2, ...
    """
    subjects = [r["subject"] for r in rows]
    for i, row in enumerate(rows):
        row["context"] = clean_context(subjects[i + 1 : i + 1 + 3 * N_RECENT])


def keeps_context(sha: str) -> bool:
    """Fixed per commit, so a rebuild makes the same choice."""
    digest = hashlib.sha256(f"{SEED}-context-{sha}".encode()).digest()
    return digest[0] / 255 >= CONTEXT_DROPOUT


def to_example(row: dict[str, Any], context: bool) -> dict[str, Any]:
    messages = build_messages(row["diff_clean"], row["context"] if context else None)
    return {"messages": [*messages, {"role": "assistant", "content": row["subject_clean"]}]}


def quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "p10": ordered[len(ordered) // 10],
        "median": statistics.median(ordered),
        "p90": ordered[(9 * len(ordered)) // 10],
        "mean": round(statistics.mean(ordered), 1),
    }


def build(raw: Path, out: Path) -> dict[str, Any]:
    reasons: dict[str, Counter[str]] = {"train": Counter(), "valid": Counter()}
    kept: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    mined_per_repo: Counter[str] = Counter()

    # The protected diffs: the test set, then validation, so training loses an
    # example rather than an evaluation keeping a copy of it.
    index = NearDuplicateIndex()
    origin: list[str] = []
    exact: dict[str, str] = {}
    for example in (json.loads(line) for line in (EVAL / "test.jsonl").open()):
        exact[hashlib.sha1(example["diff"].encode()).hexdigest()] = "test"
        items = shingles(example["diff"])
        if len(items) >= MIN_SHINGLES:
            index.add(items)
            origin.append("test")

    for split in ("valid", "train"):
        for slug in REPOS[split]:
            path = raw / f"{slug.replace('/', '__')}.jsonl"
            rows = [json.loads(line) for line in path.open()]
            mined_per_repo[slug] = len(rows)
            with_context(rows)
            for row in rows:
                ok, why = keep(row)
                if ok and not is_imperative(row["subject_clean"]):
                    ok, why = False, "not imperative"
                if ok and is_generic(row["subject_clean"]):
                    ok, why = False, "generic subject"
                if ok:
                    digest = hashlib.sha1(row["diff_clean"].encode()).hexdigest()
                    if digest in exact:
                        ok, why = False, f"duplicate of a {exact[digest]} diff"
                checkable = False
                if ok:
                    items = shingles(row["diff_clean"])
                    checkable = len(items) >= MIN_SHINGLES
                    match = index.find(items) if checkable else None
                    if match is not None:
                        ok, why = False, f"near-duplicate of a {origin[match]} diff"
                if not ok:
                    reasons[split][why] += 1
                    continue
                exact[digest] = split
                if checkable:
                    index.add(items)
                    origin.append(split)
                else:
                    reasons[split]["kept: too little text for the near-duplicate check"] += 1
                reasons[split]["kept"] += 1
                kept[split].append({**row, "repo": slug})

    rng = random.Random(SEED)
    capped: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    per_project: dict[str, Counter[str]] = {"train": Counter(), "valid": Counter()}
    for split in ("train", "valid"):
        by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in kept[split]:
            by_repo[row["repo"]].append(row)
        for _repo, rows in sorted(by_repo.items()):
            rng.shuffle(rows)
            if split == "train" and len(rows) > PER_PROJECT_CAP:
                reasons[split]["over the per-project cap"] += len(rows) - PER_PROJECT_CAP
                reasons[split]["kept"] -= len(rows) - PER_PROJECT_CAP
                rows = rows[:PER_PROJECT_CAP]
            capped[split].extend(rows)
        rng.shuffle(capped[split])
    if len(capped["valid"]) > MAX_VALID:
        reasons["valid"]["over the validation cap"] += len(capped["valid"]) - MAX_VALID
        reasons["valid"]["kept"] -= len(capped["valid"]) - MAX_VALID
        capped["valid"] = capped["valid"][:MAX_VALID]
    for split, rows in capped.items():  # counted after every cap
        per_project[split].update(row["repo"] for row in rows)

    out.mkdir(parents=True, exist_ok=True)
    for split, rows in capped.items():
        with (out / f"{split}.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(to_example(row, keeps_context(row["sha"]))) + "\n")
        with (out / f"{split}.meta.jsonl").open("w") as fh:
            for row in rows:
                fh.write(
                    json.dumps(
                        {
                            "repo": row["repo"],
                            "sha": row["sha"],
                            "date": row["date"],
                            "subject": row["subject_clean"],
                            "context": row["context"],
                            "diff": row["diff_clean"],
                            "with_context": keeps_context(row["sha"]),
                        }
                    )
                    + "\n"
                )

    def summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows = list(rows)
        return {
            "examples": len(rows),
            "subject_words": quantiles([len(r["subject_clean"].split()) for r in rows]),
            "diff_chars": quantiles([len(r["diff_clean"]) for r in rows]),
            "with_context_pct": round(
                100 * sum(keeps_context(r["sha"]) for r in rows) / max(len(rows), 1), 1
            ),
        }

    return {
        "prompt_version": PROMPT_VERSION,
        "settings": {
            "per_project_cap": PER_PROJECT_CAP,
            "min_shingles": MIN_SHINGLES,
            "max_valid": MAX_VALID,
            "context_dropout": CONTEXT_DROPOUT,
            "near_dup_jaccard": NEAR_DUP_JACCARD,
            "context_subjects": N_RECENT,
            "seed": SEED,
        },
        "mined_per_repo": dict(mined_per_repo),
        "filter_reasons": {s: dict(c.most_common()) for s, c in reasons.items()},
        "kept_per_repo": {s: dict(c.most_common()) for s, c in per_project.items()},
        "train": summary(capped["train"]),
        "valid": summary(capped["valid"]),
    }


def card(stats: dict[str, Any]) -> str:
    """docs/DATA_CARD.md, written from the build's own counts."""
    reasons, settings = stats["filter_reasons"], stats["settings"]
    mined = sum(stats["mined_per_repo"].values())
    lines = [
        "# Data card: Commit Suggestor v2 training data",
        "",
        "Generated by `uv run python -m commit_suggestor.data_v2` with the counts in "
        "`docs/data_v2_stats.json`. The data itself is not distributed: it is built from "
        "public repositories under their own open-source licences.",
        "",
        "## Source",
        "",
        f"- {len(stats['mined_per_repo'])} training and validation projects (Python), newest "
        f"4,000 commits (shallow clone depth) each: {mined:,} non-merge commits mined with "
        "`git log -p`. The 3 test projects are not used here.",
        "- Split by project: train 15, validation 2 (attrs, typer). The test set is the "
        "fixed 450 examples in `eval/test.jsonl` from django-rest-framework, pydantic and "
        "sphinx.",
        "- v1, for comparison (from its README; v1 did not commit its counts): 122,088 "
        "mined, about 45% kept, 37,898 training examples, of which it trained on 2,400.",
        "",
        "## Filters, in order, and what each removed",
        "",
        "| Reason | Train | Validation |",
        "| --- | ---: | ---: |",
    ]
    names = list(dict.fromkeys([*reasons["train"], *reasons["valid"]]))
    for name in names:
        if name.startswith("kept"):
            continue
        lines.append(
            f"| {name} | {reasons['train'].get(name, 0):,} | {reasons['valid'].get(name, 0):,} |"
        )
    lines.append(
        f"| **kept** | **{reasons['train']['kept']:,}** | **{reasons['valid']['kept']:,}** |"
    )
    small = reasons["train"].get("kept: too little text for the near-duplicate check", 0)
    test_copies = sum(n for why, n in reasons["train"].items() if why.endswith("test diff"))
    lines += [
        "",
        "- v1 filters: bots, merges, reverts, releases and bumps; subjects of 12-72 "
        "characters; at most 4 files; diffs of 40-3,000 characters; lock, minified and "
        "binary files removed. PR numbers and CI tags are stripped from subjects.",
        "- v2 adds: generic subjects (`metrics.is_generic`), subjects not in the imperative "
        f"mood, and near-duplicate diffs (MinHash, Jaccard >= {settings['near_dup_jaccard']} "
        "on word 5-grams of changed lines), checked against the test and validation diffs "
        f"first so training loses the copy. Training diffs removed as copies of a test "
        f"diff: {test_copies:,}.",
        f"- {small:,} kept training diffs change too little text (under "
        f"{settings['min_shingles']} shingles) for the near-duplicate check; they get the "
        "exact-duplicate check only.",
        f"- At most {settings['per_project_cap']:,} examples per training project; "
        f"validation is capped at {settings['max_valid']}.",
        "",
        "## Kept per project",
        "",
        "| Project | Mined | Kept |",
        "| --- | ---: | ---: |",
    ]
    for split in ("train", "valid"):
        for slug, n in stats["kept_per_repo"][split].items():
            lines.append(f"| {slug} ({split}) | {stats['mined_per_repo'][slug]:,} | {n:,} |")
    t = stats["train"]
    lines += [
        "",
        "## Shape",
        "",
        "| | Train | Validation |",
        "| --- | ---: | ---: |",
        f"| Examples | {t['examples']:,} | {stats['valid']['examples']:,} |",
        f"| Subject words, median (p10-p90) | {t['subject_words']['median']} "
        f"({t['subject_words']['p10']}-{t['subject_words']['p90']}) | "
        f"{stats['valid']['subject_words']['median']} |",
        f"| Diff characters, median (p10-p90) | {t['diff_chars']['median']:,} "
        f"({t['diff_chars']['p10']:,}-{t['diff_chars']['p90']:,}) | "
        f"{stats['valid']['diff_chars']['median']:,} |",
        f"| With repo context | {t['with_context_pct']}% | {stats['valid']['with_context_pct']}% |",
        "",
        "## Repo-style context",
        "",
        f"Each example carries the subjects of the {settings['context_subjects']} non-merge "
        "commits before it (cleaned, each cut to 100 characters). For "
        f"{round(100 * settings['context_dropout'])}% of examples, fixed per commit, the "
        "prompt leaves them out, so the model also works without history. Training takes "
        "them from the mined commit order; the test set and the CLI from `git log`, which "
        "can differ slightly where history branches.",
        "",
        "## Known limitations",
        "",
        "- Python projects only, and only their recent history.",
        "- Subjects are the authors' own and are noisy: in a blind rating I scored 21% of "
        "test-set author subjects 1 or 2 out of 5 (`eval/rouge_vs_human.md`).",
        "- Diffs over 3,000 characters are dropped, so large changes are not represented.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/v2"))
    parser.add_argument("--stats", type=Path, default=Path("docs/data_v2_stats.json"))
    args = parser.parse_args(argv)
    stats = build(args.raw, args.out)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    args.stats.with_name("DATA_CARD.md").write_text(card(stats))
    print(json.dumps({k: stats[k] for k in ("filter_reasons", "train", "valid")}, indent=2))


if __name__ == "__main__":
    main()

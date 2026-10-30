"""Rate the blind side-by-side sample by hand, then reveal which model won.

    uv run python scripts/rate.py           # rate eval/ab_sample.jsonl (resumes)
    uv run python scripts/rate.py --score   # reveal: fine-tuned vs base win rate

    # any other blind A/B sheet, e.g. the DPO spot check:
    uv run python scripts/rate.py --set eval/dpo/spot

You see a diff and two suggestions, A and B, in random order per example. You
never see which is which; the key lives in eval/ab_key.jsonl and is read only
by --score. One key per answer, saved at once: a, b, t (tie) or q (quit).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from keys import clear, getkey, paint, show_diff  # noqa: E402

from commit_suggestor.stats import wilson  # noqa: E402

EVAL = ROOT / "eval"


def paths(prefix: Path) -> tuple[Path, Path, Path]:
    """<prefix>_sample.jsonl, <prefix>_key.jsonl and <prefix>_ratings.jsonl."""
    return tuple(
        prefix.with_name(f"{prefix.name}_{part}.jsonl")  # type: ignore[return-value]
        for part in ("sample", "key", "ratings")
    )


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def rate(prefix: Path) -> None:
    sample, _, ratings = paths(prefix)
    done = {r["id"] for r in load(ratings)}
    todo = [s for s in load(sample) if s["id"] not in done]
    print(f"{len(done)} rated, {len(todo)} to go. Answer a, b, t (tie) or q (quit).\n")
    with ratings.open("a") as out:
        for n, s in enumerate(todo, len(done) + 1):
            clear()
            print(paint(f"[{n}/{len(done) + len(todo)}]", "1;33"))
            show_diff(s["diff"])
            print("-" * 78)
            print(f"{paint('A', '1')}: {s['A']}\n{paint('B', '1')}: {s['B']}")
            answer = getkey("better? [a/b/t/q]", "abtq")
            if answer == "q":
                return
            at = datetime.now(UTC).isoformat(timespec="seconds")
            out.write(json.dumps({"id": s["id"], "choice": answer, "at": at}) + "\n")
            out.flush()


def score(prefix: Path) -> None:
    _, key_path, ratings = paths(prefix)
    key = {k["id"]: k for k in load(key_path)}
    names = sorted({k[slot] for k in key.values() for slot in "AB"})
    wins = {**dict.fromkeys(names, 0), "tie": 0}
    for r in load(ratings):
        if r["choice"] == "t":
            wins["tie"] += 1
        else:
            wins[key[r["id"]][r["choice"].upper()]] += 1
    n = sum(wins.values())
    if not n:
        print("no ratings yet")
        return
    for name, count in wins.items():
        print(f"{name:10} {count:3}  ({count / n:.0%})")
    first = "finetuned" if "finetuned" in names else "chosen" if "chosen" in names else names[0]
    decided = n - wins["tie"]
    if decided:
        rate, low, high = wilson(wins[first], decided).scaled(100, 1)
        print(f"{first} win rate excluding ties: {rate}% (95% CI {low}-{high}, n={decided})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score", action="store_true")
    parser.add_argument(
        "--set", type=Path, default=EVAL / "ab", help="file prefix of the sheet (default eval/ab)"
    )
    args = parser.parse_args()
    score(args.set) if args.score else rate(args.set)

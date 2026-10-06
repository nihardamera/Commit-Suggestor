"""Rate the blind side-by-side sample by hand, then reveal which model won.

    python scripts/rate.py           # rate eval/ab_sample.jsonl (resumes where you stopped)
    python scripts/rate.py --score   # reveal: win rate of the fine-tuned model vs the base

You see a diff and two suggestions, A and B, in random order per example. You
never see which is which; the key lives in eval/ab_key.jsonl and is read only
by --score. Ratings are saved after every answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent / "eval"
SAMPLE, KEY, RATINGS = EVAL / "ab_sample.jsonl", EVAL / "ab_key.jsonl", EVAL / "ab_ratings.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def rate() -> None:
    done = {r["id"] for r in load(RATINGS)}
    todo = [s for s in load(SAMPLE) if s["id"] not in done]
    print(f"{len(done)} rated, {len(todo)} to go. Answer a, b, t (tie) or q (quit).\n")
    with RATINGS.open("a") as out:
        for s in todo:
            print("=" * 78)
            print(s["diff"][:2500])
            print("-" * 78)
            print(f"A: {s['A']}\nB: {s['B']}")
            answer = ""
            while answer not in {"a", "b", "t", "q"}:
                answer = input("better? [a/b/t/q] ").strip().lower()
            if answer == "q":
                return
            out.write(json.dumps({"id": s["id"], "choice": answer}) + "\n")
            out.flush()


def score() -> None:
    key = {k["id"]: k for k in load(KEY)}
    wins = {"finetuned": 0, "base": 0, "tie": 0}
    for r in load(RATINGS):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score", action="store_true")
    score() if parser.parse_args().score else rate()

"""Turn SFT-v2's sampled subjects into DPO preference pairs, labelled by the judge.

For each training diff the sampling job drew 4 subjects. Two distinct ones
are judged with the frozen judge (eval/judge/heldout.json) in both orders, and
a pair is kept only when both orders name the same winner: on my dev labels
those verdicts matched my choice 74% of the time. "Both bad" pairs are also
dropped: a winner that is generic or names an identifier missing from the
diff teaches nothing worth learning.

The length gap between chosen and rejected subjects is reported with a CI:
a judge that prefers short (or long) answers would teach DPO exactly that.

    uv run python -m commit_suggestor.dpo_data label --samples data/kaggle-output/sample/full
    uv run python -m commit_suggestor.dpo_data label ... --yes     # paid
    uv run python -m commit_suggestor.dpo_data spotcheck   # 50 pairs to rate blind, then:
    uv run python scripts/rate.py --set eval/dpo/spot
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from commit_suggestor.evaluate import normalise
from commit_suggestor.judge import (
    BUDGET_USD,
    CACHE,
    HELDOUT_LOCK,
    Cache,
    Judge,
    _lazy_backend,
    estimate_usd,
)
from commit_suggestor.metrics import check
from commit_suggestor.stats import stratified_bootstrap

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "data" / "dpo" / "pairs.jsonl"
STATS = ROOT / "eval" / "dpo" / "pairs_stats.json"
SPOT = ROOT / "eval" / "dpo"
SEED = 2


def diff_of(prompt: list[dict[str, str]]) -> str:
    """The diff inside a built prompt (between the ```diff fence and its end)."""
    user = prompt[-1]["content"]
    return user.split("```diff\n", 1)[1].rsplit("\n```", 1)[0]


def candidate_pairs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Two distinct samples per diff, chosen with a fixed seed."""
    rng = random.Random(SEED)
    out, too_few = [], 0
    for row in rows:
        distinct = list({normalise(s): s for s in row["samples"] if s.strip()}.values())
        if len(distinct) < 2:
            too_few += 1
            continue
        a, b = rng.sample(distinct, 2)
        out.append({**row, "a": a, "b": b, "diff": diff_of(row["prompt"])})
    return out, too_few


def label(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in (Path(args.samples) / "dpo_samples.jsonl").open()]
    candidates, too_few = candidate_pairs(rows)
    frozen = json.loads(HELDOUT_LOCK.read_text())
    cache = Cache(CACHE)
    judge = Judge(
        _lazy_backend(),
        rubric=frozen["rubric"],
        model=frozen["model"],
        effort=frozen["effort"],
        cache=cache,
    )
    items = [(c["diff"], c["a"], c["b"]) for c in candidates]
    todo = judge.uncached(items)
    cost, spent = estimate_usd(todo), cache.spent_usd()
    print(
        f"{len(rows)} diffs, {len(candidates)} with two distinct samples; "
        f"{cost['calls']} calls not cached, estimate ${cost['usd']:.4f}; "
        f"spent so far ${spent:.4f} of ${BUDGET_USD:.2f}"
    )
    if cost["calls"] and not args.yes:
        print("dry run: nothing was sent. Re-run with --yes to make these calls.")
        return
    if cost["usd"] > args.max_usd or spent + cost["usd"] > BUDGET_USD:
        sys.exit("refused: over --max-usd or the project budget")

    verdicts = judge.compare_many(items)
    reasons: dict[str, int] = {"fewer than two distinct samples": too_few}
    kept = []
    for c, v in zip(candidates, verdicts, strict=True):
        if not v.consistent:
            why = "orders disagree"
        elif v.winner == "tie":
            why = "judged a tie"
        else:
            chosen, rejected = (c["a"], c["b"]) if v.winner == "x" else (c["b"], c["a"])
            verdict = check(chosen, c["diff"])
            why = (
                "winner is generic"
                if verdict.generic
                else "winner names an identifier not in the diff"
                if verdict.hallucinated
                else "kept"
            )
        reasons[why] = reasons.get(why, 0) + 1
        if why == "kept":
            kept.append(
                {
                    "sha": c["sha"],
                    "repo": c["repo"],
                    "prompt": c["prompt"],
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                }
            )

    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with PAIRS.open("w") as fh:
        for pair in kept:
            fh.write(json.dumps(pair) + "\n")
    gap = [
        float(len(p["chosen"][0]["content"].split()) - len(p["rejected"][0]["content"].split()))
        for p in kept
    ]
    longer = [float(g > 0) for g in gap]
    repos = [p["repo"] for p in kept]
    stats = {
        "diffs_sampled": len(rows),
        "pairs_judged": len(candidates),
        "reasons": reasons,
        "pairs_kept": len(kept),
        "chosen_minus_rejected_words": stratified_bootstrap(gap, repos).scaled(1),
        "chosen_longer_pct": stratified_bootstrap(longer, repos).scaled(100),
        "judge": frozen,
        "spent_usd_total": round(cache.spent_usd(), 4),
    }
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def spotcheck(_: argparse.Namespace) -> None:
    """50 kept pairs, shuffled and blind, for scripts/rate.py."""
    pairs = [json.loads(line) for line in PAIRS.open()]
    rng = random.Random(SEED + 1)
    sample = rng.sample(range(len(pairs)), min(50, len(pairs)))
    SPOT.mkdir(parents=True, exist_ok=True)
    sample_path, key_path = SPOT / "spot_sample.jsonl", SPOT / "spot_key.jsonl"
    with sample_path.open("w") as fh, key_path.open("w") as key:
        for i in sample:
            p = pairs[i]
            chosen, rejected = p["chosen"][0]["content"], p["rejected"][0]["content"]
            flip = rng.random() < 0.5
            a, b = (rejected, chosen) if flip else (chosen, rejected)
            fh.write(json.dumps({"id": i, "diff": diff_of(p["prompt"]), "A": a, "B": b}) + "\n")
            key.write(
                json.dumps(
                    {
                        "id": i,
                        "A": "rejected" if flip else "chosen",
                        "B": "chosen" if flip else "rejected",
                    }
                )
                + "\n"
            )
    print(f"wrote {len(sample)} pairs to {SPOT.relative_to(ROOT)}/spot_sample.jsonl")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("label")
    lab.add_argument("--samples", required=True, help="folder with dpo_samples.jsonl")
    lab.add_argument("--yes", action="store_true")
    lab.add_argument("--max-usd", type=float, default=2.00)
    lab.set_defaults(run=label)
    sub.add_parser("spotcheck").set_defaults(run=spotcheck)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

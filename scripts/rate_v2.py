"""Rate 100 more test diffs blind: base, v1 fine-tuned and the author's own subject.

    uv run python scripts/rate_v2.py --build    # once: draw the sample (already committed)
    uv run python scripts/rate_v2.py            # rate; resumes where you stopped
    uv run python scripts/rate_v2.py --rerate   # a day later: 30 items again, for self-agreement
    uv run python scripts/rate_v2.py --score    # reveal and summarise

Each screen shows a diff and three subjects in random order. One key per answer:
a 1-5 score for each subject, a reason key for any score of 1 or 2, then the
best of the three. Which subject is which lives in eval/human/v2_key.jsonl and
is read only by --score. Every item is saved as soon as it is finished.

The 100 diffs come from the test set and exclude the 50 already in
eval/ab_sample.jsonl, so the two sets together give 150 different diffs.
--build also fixes, before any judge exists, which labels the judge's prompt
may be tuned on (dev, 2/3) and which it is scored on once (held-out, 1/3).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from keys import clear, getkey, paint, show_diff  # noqa: E402

from commit_suggestor.stats import (  # noqa: E402
    cohen_kappa,
    confusion,
    spearman,
    stratified_bootstrap,
    wilson,
)

EVAL = ROOT / "eval"
HUMAN = EVAL / "human"
SAMPLE, KEY = HUMAN / "v2_sample.jsonl", HUMAN / "v2_key.jsonl"
RATINGS, RERATINGS = HUMAN / "v2_ratings.jsonl", HUMAN / "v2_rerate.jsonl"
SPLIT = HUMAN / "split.json"
SYSTEMS = ("base", "finetuned", "reference")
SLOTS = "123"
N_ITEMS, N_RERATE, SEED = 100, 30, 2

RUBRIC = """\
  5  accurate, specific, imperative: what I would commit
  4  accurate and clear; a small wording or style issue
  3  roughly right, but vague or partly inaccurate
  2  misleading or mostly wrong, or badly formed
  1  wrong, empty or nonsense
Reason for a 1 or 2:  i inaccurate   v vague   f form (tense, length, style)   o other"""


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def build() -> None:
    if SAMPLE.exists():
        sys.exit(f"{SAMPLE.relative_to(ROOT)} exists; delete it first to draw a new sample")
    HUMAN.mkdir(exist_ok=True)
    tests = load(EVAL / "test.jsonl")
    preds = load(EVAL / "predictions.jsonl")
    ab_ids = sorted(s["id"] for s in load(EVAL / "ab_sample.jsonl"))
    rng = random.Random(SEED)
    pool = [i for i in range(len(tests)) if i not in set(ab_ids)]
    ids = sorted(rng.sample(pool, N_ITEMS))
    with SAMPLE.open("w") as sample, KEY.open("w") as key:
        for i in ids:
            order = list(SYSTEMS)
            rng.shuffle(order)
            text = {
                "base": preds[i]["base"],
                "finetuned": preds[i]["finetuned"],
                "reference": preds[i]["reference"],
            }
            sample.write(
                json.dumps(
                    {
                        "id": i,
                        "repo": tests[i]["repo"],
                        "diff": tests[i]["diff"],
                        "subjects": {s: text[o] for s, o in zip(SLOTS, order, strict=True)},
                    }
                )
                + "\n"
            )
            key.write(json.dumps({"id": i, **dict(zip(SLOTS, order, strict=True))}) + "\n")

    def halves(items: list[int]) -> dict[str, list[int]]:
        shuffled = items[:]
        rng.shuffle(shuffled)
        cut = round(len(shuffled) * 2 / 3)
        return {"dev": sorted(shuffled[:cut]), "heldout": sorted(shuffled[cut:])}

    SPLIT.write_text(json.dumps({"ab": halves(ab_ids), "v2": halves(ids)}, indent=2) + "\n")
    print(f"wrote {N_ITEMS} items, their key and the dev/held-out split to {HUMAN}")


def rate_items(items: list[dict], out_path: Path, restore: Callable[[dict], dict]) -> None:
    """Rate `items` in order, appending one record per item to `out_path`.

    `restore` turns a saved record back into the item it rated, for undo.
    """
    print(f"{len(items)} to go. Keys: 1-5 score, u undo the last item, ? rubric, q quit.")
    queue = list(items)
    while queue:
        item = queue[0]
        clear()
        print(paint(f"[{len(load(out_path)) + 1}] {item['repo']}", "1;33"))
        show_diff(item["diff"])
        print("-" * 78)
        for slot in SLOTS:
            print(f"{paint(slot, '1')}: {item['subjects'][slot]}")
        print("-" * 78)
        started = time.time()
        scores: dict[str, int] = {}
        reasons: dict[str, str] = {}
        for slot in SLOTS:
            key = getkey(f"score {slot} [1-5]:", "12345u?q")
            while key == "?":
                print(RUBRIC)
                key = getkey(f"score {slot} [1-5]:", "12345u?q")
            if key in "uq":
                break
            scores[slot] = int(key)
            if scores[slot] <= 2:
                reasons[slot] = getkey(f"  why is {slot} bad? [i/v/f/o]:", "ivfo")
        else:
            key = getkey("best overall? [1/2/3, 0 = no clear best]:", "1230q")
        if key == "q":
            return
        if key == "u":
            saved = load(out_path)
            if saved:
                out_path.write_text("".join(json.dumps(r) + "\n" for r in saved[:-1]))
                queue.insert(0, restore(saved[-1]))
            continue
        record = {
            "id": item["id"],
            "scores": scores,
            "reasons": reasons,
            "best": key,
            "seconds": round(time.time() - started, 1),
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if "slot_map" in item:
            record["slot_map"] = item["slot_map"]
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        queue.pop(0)


def rate() -> None:
    if not SAMPLE.exists():
        sys.exit("no sample yet: run with --build first")
    sample = {s["id"]: s for s in load(SAMPLE)}
    done = {r["id"] for r in load(RATINGS)}
    rate_items(
        [s for i, s in sample.items() if i not in done],
        RATINGS,
        lambda record: sample[record["id"]],
    )
    print(f"\n{len(load(RATINGS))} of {N_ITEMS} rated; ratings in {RATINGS.relative_to(ROOT)}")


def reshuffled(item: dict) -> dict:
    """The item with its subjects in a new order, fixed per item so resuming repeats it."""
    order = list(SLOTS)
    random.Random(f"{SEED}-rerate-{item['id']}").shuffle(order)
    slot_map = dict(zip(SLOTS, order, strict=True))  # slot shown now -> original slot
    return {
        **item,
        "subjects": {new: item["subjects"][old] for new, old in slot_map.items()},
        "slot_map": slot_map,
    }


def rerate() -> None:
    """Re-rate 30 items at least a day after first rating them, in a new order."""
    sample = {s["id"]: s for s in load(SAMPLE)}
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    old_enough = {r["id"] for r in load(RATINGS) if datetime.fromisoformat(r["at"]) < cutoff}
    done = {r["id"] for r in load(RERATINGS)}
    if len(done) >= N_RERATE:
        sys.exit("re-rating is complete")
    priority = sorted(sample)
    random.Random(SEED + 1).shuffle(priority)
    todo = [i for i in priority if i in old_enough and i not in done][: N_RERATE - len(done)]
    if not todo:
        sys.exit("no items were first rated more than 24 hours ago yet")
    rate_items(
        [reshuffled(sample[i]) for i in todo],
        RERATINGS,
        lambda record: reshuffled(sample[record["id"]]),
    )


def in_original_slots(record: dict) -> dict:
    """A re-rating with its scores moved back to the slots of the first rating."""
    m = record.get("slot_map")
    if not m:
        return record
    return {
        **record,
        "scores": {m[s]: v for s, v in record["scores"].items()},
        "reasons": {m[s]: v for s, v in record["reasons"].items()},
        "best": m.get(record["best"], "0"),
    }


def pairwise(record: dict, slot_a: str, slot_b: str) -> str:
    """'a', 'b' or 'tie' for one pair: the best pick decides, else the scores."""
    if record["best"] == slot_a:
        return "a"
    if record["best"] == slot_b:
        return "b"
    sa, sb = record["scores"][slot_a], record["scores"][slot_b]
    return "a" if sa > sb else "b" if sb > sa else "tie"


def score() -> None:
    from rouge_score import rouge_scorer

    key = {k["id"]: {slot: k[slot] for slot in SLOTS} for k in load(KEY)}
    sample = {s["id"]: s for s in load(SAMPLE)}
    ratings = load(RATINGS)
    if not ratings:
        sys.exit("no ratings yet")
    slot_of = {i: {system: slot for slot, system in k.items()} for i, k in key.items()}
    repos = [sample[r["id"]]["repo"] for r in ratings]

    summary: dict = {"items_rated": len(ratings), "mean_score": {}, "reasons": {}, "pairwise": {}}
    for system in SYSTEMS:
        values = [float(r["scores"][slot_of[r["id"]][system]]) for r in ratings]
        summary["mean_score"][system] = stratified_bootstrap(values, repos).scaled(1)
        summary["reasons"][system] = dict(
            Counter(
                r["reasons"][slot_of[r["id"]][system]]
                for r in ratings
                if slot_of[r["id"]][system] in r["reasons"]
            )
        )
    best = Counter(key[r["id"]].get(r["best"], "none") for r in ratings)
    summary["best_of_three"] = dict(best)

    for a, b in (("finetuned", "base"), ("finetuned", "reference"), ("base", "reference")):
        outcomes = Counter(pairwise(r, slot_of[r["id"]][a], slot_of[r["id"]][b]) for r in ratings)
        decided = outcomes["a"] + outcomes["b"]
        summary["pairwise"][f"{a}_vs_{b}"] = {
            "wins": outcomes["a"],
            "losses": outcomes["b"],
            "ties": outcomes["tie"],
            "win_rate_excluding_ties_pct": wilson(outcomes["a"], decided).scaled(100)
            if decided
            else None,
        }

    # Does ROUGE-L agree with me? Per subject, and per base-vs-fine-tuned pair.
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    human, rouge, agree, disagreements = [], [], 0, []
    for r in ratings:
        ref = sample[r["id"]]["subjects"][slot_of[r["id"]]["reference"]]
        per_system = {}
        for system in ("base", "finetuned"):
            text = sample[r["id"]]["subjects"][slot_of[r["id"]][system]]
            per_system[system] = scorer.score(ref, text)["rougeL"].fmeasure
            human.append(r["scores"][slot_of[r["id"]][system]])
            rouge.append(per_system[system])
        mine = pairwise(r, slot_of[r["id"]]["finetuned"], slot_of[r["id"]]["base"])
        if mine == "tie" or per_system["finetuned"] == per_system["base"]:
            continue
        rouge_pick = "a" if per_system["finetuned"] > per_system["base"] else "b"
        if rouge_pick == mine:
            agree += 1
        else:
            disagreements.append(r["id"])
    decided = agree + len(disagreements)
    summary["rouge_vs_human"] = {
        "spearman_score_vs_rougeL": round(spearman(human, rouge), 3),
        "same_winner_as_me_pct": wilson(agree, decided).scaled(100) if decided else None,
        "pairs_compared": decided,
        "disagreement_ids": disagreements,
    }

    reratings = {r["id"]: in_original_slots(r) for r in load(RERATINGS)}
    if reratings:
        firsts = {r["id"]: r for r in ratings}
        ids = sorted(i for i in reratings if i in firsts)
        pairs = ("finetuned", "base")
        first = [pairwise(firsts[i], slot_of[i][pairs[0]], slot_of[i][pairs[1]]) for i in ids]
        second = [pairwise(reratings[i], slot_of[i][pairs[0]], slot_of[i][pairs[1]]) for i in ids]
        summary["self_agreement"] = {
            "items": len(ids),
            "kappa_finetuned_vs_base": round(cohen_kappa(first, second, ["a", "b", "tie"]), 3),
            "confusion_first_rows_second_cols_a_b_tie": confusion(first, second, ["a", "b", "tie"]),
            "score_spearman": round(
                spearman(
                    [s for i in ids for s in firsts[i]["scores"].values()],
                    [reratings[i]["scores"][slot] for i in ids for slot in firsts[i]["scores"]],
                ),
                3,
            ),
        }

    summary["median_seconds_per_item"] = sorted(r["seconds"] for r in ratings)[len(ratings) // 2]
    (HUMAN / "v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--rerate", action="store_true")
    mode.add_argument("--score", action="store_true")
    args = parser.parse_args()
    try:
        build() if args.build else rerate() if args.rerate else score() if args.score else rate()
    except KeyboardInterrupt:
        print("\nstopped; everything finished so far is saved")

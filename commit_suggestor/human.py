"""My blind ratings, unblinded and compared with ROUGE-L and the rule-based metrics.

Two rating sets, 150 different test diffs in all:
  ab   50 diffs, base vs fine-tuned, pick the better (scripts/rate.py)
  v2   100 diffs, base, fine-tuned and the author's subject, each scored 1-5,
       with a reason for a 1 or 2 and a best-of-three pick (scripts/rate_v2.py)

`pairwise_labels()` turns both into one list of fine-tuned-vs-base labels, with
the dev/held-out split fixed in eval/human/split.json before any judge
existed. The LLM judge is calibrated against that list.

    uv run python -m commit_suggestor.human    # eval/human/analysis.json, eval/rouge_vs_human.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rouge_score import rouge_scorer

from commit_suggestor.metrics import check
from commit_suggestor.stats import kendall_tau_b, spearman, stratified_bootstrap, wilson

EVAL = Path(__file__).resolve().parent.parent / "eval"
HUMAN = EVAL / "human"
SLOTS = "123"


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()]


@dataclass(frozen=True)
class Pair:
    """One fine-tuned-vs-base judgement of mine."""

    set: str  # "ab" or "v2"
    id: int  # index into eval/test.jsonl and eval/predictions.jsonl
    repo: str
    diff: str
    reference: str
    finetuned: str
    base: str
    label: str  # "finetuned", "base" or "tie"
    split: str  # "dev" or "heldout"


def _v2_pair_label(record: dict[str, Any], slot_ft: str, slot_base: str) -> str:
    """The best-of-three pick decides; otherwise the 1-5 scores; equal scores tie."""
    if record["best"] in (slot_ft, slot_base):
        return "finetuned" if record["best"] == slot_ft else "base"
    ft, base = record["scores"][slot_ft], record["scores"][slot_base]
    return "finetuned" if ft > base else "base" if base > ft else "tie"


def v2_items() -> list[dict[str, Any]]:
    """Each v2 rating with its subjects, scores and reasons keyed by system."""
    key = {k["id"]: k for k in _load(HUMAN / "v2_key.jsonl")}
    sample = {s["id"]: s for s in _load(HUMAN / "v2_sample.jsonl")}
    items = []
    for r in _load(HUMAN / "v2_ratings.jsonl"):
        system_of = {slot: key[r["id"]][slot] for slot in SLOTS}
        items.append(
            {
                "id": r["id"],
                "repo": sample[r["id"]]["repo"],
                "diff": sample[r["id"]]["diff"],
                "subject": {system_of[s]: sample[r["id"]]["subjects"][s] for s in SLOTS},
                "score": {system_of[s]: r["scores"][s] for s in SLOTS},
                "reason": {system_of[s]: v for s, v in r["reasons"].items()},
                "best": system_of.get(r["best"], "none"),
                "label": _v2_pair_label(
                    r,
                    *(
                        next(s for s in SLOTS if system_of[s] == name)
                        for name in ("finetuned", "base")
                    ),
                ),
            }
        )
    return items


def pairwise_labels() -> list[Pair]:
    tests = _load(EVAL / "test.jsonl")
    preds = _load(EVAL / "predictions.jsonl")
    split = json.loads((HUMAN / "split.json").read_text())
    which = {
        (name, i): half
        for name, halves in split.items()
        for half, ids in halves.items()
        for i in ids
    }
    pairs = []
    key = {k["id"]: k for k in _load(EVAL / "ab_key.jsonl")}
    for r in _load(EVAL / "ab_ratings.jsonl"):
        label = "tie" if r["choice"] == "t" else key[r["id"]][r["choice"].upper()]
        pairs.append(_pair("ab", r["id"], label, tests, preds, which))
    for item in v2_items():
        pairs.append(_pair("v2", item["id"], item["label"], tests, preds, which))
    return pairs


def _pair(
    name: str,
    i: int,
    label: str,
    tests: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    which: dict[tuple[str, int], str],
) -> Pair:
    return Pair(
        set=name,
        id=i,
        repo=tests[i]["repo"],
        diff=tests[i]["diff"],
        reference=preds[i]["reference"],
        finetuned=preds[i]["finetuned"],
        base=preds[i]["base"],
        label=label,
        split=which[(name, i)],
    )


def win_rate(labels: Sequence[str]) -> dict[str, Any]:
    counts = Counter(labels)
    decided = counts["finetuned"] + counts["base"]
    return {
        "finetuned": counts["finetuned"],
        "base": counts["base"],
        "tie": counts["tie"],
        "finetuned_win_rate_excl_ties_pct": wilson(counts["finetuned"], decided).scaled(100, 1)
        if decided
        else None,
    }


def bootstrap_spearman(
    x: Sequence[float], y: Sequence[float], n: int = 5000, seed: int = 0
) -> list[float]:
    """[rho, low, high]: Spearman's rho with a 95% interval from resampling items."""
    rng = np.random.default_rng(seed)
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    draws = [
        spearman(xs[idx].tolist(), ys[idx].tolist())
        for idx in rng.integers(0, len(xs), size=(n, len(xs)))
    ]
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return [round(spearman(x, y), 3), round(float(low), 3), round(float(high), 3)]


def analyse() -> dict[str, Any]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def rouge(ref: str, text: str) -> float:
        return float(scorer.score(ref, text)["rougeL"].fmeasure)

    pairs = pairwise_labels()
    report: dict[str, Any] = {
        "pairs": len(pairs),
        "preference": {
            "ab": win_rate([p.label for p in pairs if p.set == "ab"]),
            "v2": win_rate([p.label for p in pairs if p.set == "v2"]),
            "all": win_rate([p.label for p in pairs]),
        },
    }

    # 1. Does ROUGE-L pick the winner I pick?
    agree, disagreements = 0, []
    for p in pairs:
        r_ft, r_base = rouge(p.reference, p.finetuned), rouge(p.reference, p.base)
        if p.label == "tie" or r_ft == r_base:
            continue
        if ("finetuned" if r_ft > r_base else "base") == p.label:
            agree += 1
        else:
            disagreements.append((abs(r_ft - r_base), p, r_ft, r_base))
    decided = agree + len(disagreements)
    report["rouge_picks_my_winner_pct"] = wilson(agree, decided).scaled(100, 1)
    report["rouge_pairs_compared"] = decided

    # 2. Per subject on v2: my 1-5 score against ROUGE-L and length.
    items = v2_items()
    scores, rouges, words = [], [], []
    for it in items:
        for system in ("base", "finetuned"):
            scores.append(float(it["score"][system]))
            rouges.append(rouge(it["subject"]["reference"], it["subject"][system]))
            words.append(float(len(it["subject"][system].split())))
    report["v2_score_vs_rougeL"] = {
        "spearman": bootstrap_spearman(scores, rouges),
        "kendall_tau_b": round(kendall_tau_b(scores, rouges), 3),
        "subjects": len(scores),
    }
    report["v2_score_vs_words_spearman"] = bootstrap_spearman(scores, words)

    # 3. The reference ROUGE treats as gold, as I rated it.
    repos = [it["repo"] for it in items]
    report["v2_mean_score"] = {
        s: stratified_bootstrap([float(it["score"][s]) for it in items], repos).scaled(1)
        for s in ("base", "finetuned", "reference")
    }
    report["v2_score_finetuned_minus_base"] = stratified_bootstrap(
        [float(it["score"]["finetuned"] - it["score"]["base"]) for it in items], repos
    ).scaled(1)
    ref_scores = [it["score"]["reference"] for it in items]
    report["reference_rated_1_or_2_pct"] = wilson(
        sum(s <= 2 for s in ref_scores), len(ref_scores)
    ).scaled(100, 1)
    report["score_vs_reference"] = {
        system: {
            "higher": sum(it["score"][system] > it["score"]["reference"] for it in items),
            "equal": sum(it["score"][system] == it["score"]["reference"] for it in items),
            "lower": sum(it["score"][system] < it["score"]["reference"] for it in items),
        }
        for system in ("base", "finetuned")
    }
    report["best_of_three"] = dict(Counter(it["best"] for it in items))
    report["reasons"] = {
        s: dict(Counter(it["reason"][s] for it in items if s in it["reason"]))
        for s in ("base", "finetuned", "reference")
    }

    # 4. Do the rule-based metrics agree with my reasons?
    rows = [
        (check(it["subject"][s], it["diff"]), it["reason"].get(s), it["score"][s])
        for it in items
        for s in ("base", "finetuned", "reference")
    ]
    vague = [r for r in rows if r[1] == "v"]
    report["metrics_vs_my_reasons"] = {
        "subjects": len(rows),
        "my_vague_flags": len(vague),
        "metric_specific_on_my_vague_pct": round(
            100 * sum(c.specific for c, _, _ in vague) / max(len(vague), 1), 1
        ),
        "metric_specific_on_rest_pct": round(
            100
            * sum(c.specific for c, why, _ in rows if why != "v")
            / max(sum(why != "v" for _, why, _ in rows), 1),
            1,
        ),
        "metric_hallucinated": sum(bool(c.hallucinated) for c, _, _ in rows),
        "metric_hallucinated_that_i_flagged_inaccurate": sum(
            bool(c.hallucinated) and why == "i" for c, why, _ in rows
        ),
        "my_inaccurate_flags": sum(why == "i" for _, why, _ in rows),
        "mean_score_specific_vs_not": [
            round(float(np.mean([s for c, _, s in rows if c.specific])), 2),
            round(float(np.mean([s for c, _, s in rows if not c.specific])), 2),
        ],
    }

    disagreements.sort(key=lambda d: -d[0])
    report["largest_disagreements"] = [
        {
            "set": p.set,
            "id": p.id,
            "repo": p.repo,
            "my_pick": p.label,
            "reference": p.reference,
            "finetuned": p.finetuned,
            "base": p.base,
            "rougeL_finetuned": round(100 * r_ft, 1),
            "rougeL_base": round(100 * r_base, 1),
        }
        for _, p, r_ft, r_base in disagreements[:10]
    ]
    return report


def markdown(r: dict[str, Any]) -> str:
    pref = r["preference"]

    def rate(x: dict[str, Any]) -> str:
        v = x["finetuned_win_rate_excl_ties_pct"]
        return f"{v[0]}% [{v[1]}, {v[2]}]"

    ms = r["v2_mean_score"]
    mv = r["metrics_vs_my_reasons"]
    lines = [
        "# What ROUGE-L missed: my ratings against the automatic metrics",
        "",
        "Generated by `uv run python -m commit_suggestor.human` from my blind ratings "
        "(`eval/ab_ratings.jsonl`, `eval/human/v2_ratings.jsonl`). Every number is in "
        "`eval/human/analysis.json`. Intervals are 95% (Wilson for rates, a bootstrap "
        "otherwise).",
        "",
        "## Do I prefer the fine-tuned model?",
        "",
        "| Set | Diffs | Fine-tuned wins | Base wins | Ties | Fine-tuned win rate, ties excluded |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, label in (("ab", "A/B (pick one)"), ("v2", "v2 (scored 1-5)"), ("all", "Both")):
        x = pref[name]
        n = x["finetuned"] + x["base"] + x["tie"]
        lines.append(f"| {label} | {n} | {x['finetuned']} | {x['base']} | {x['tie']} | {rate(x)} |")
    lines += [
        "",
        f"Mean 1-5 score on v2: base {ms['base'][0]} [{ms['base'][1]}, {ms['base'][2]}], "
        f"fine-tuned {ms['finetuned'][0]} [{ms['finetuned'][1]}, {ms['finetuned'][2]}], "
        f"the author's own subject {ms['reference'][0]} "
        f"[{ms['reference'][1]}, {ms['reference'][2]}]. Paired, fine-tuned minus base: "
        f"{r['v2_score_finetuned_minus_base'][0]} "
        f"[{r['v2_score_finetuned_minus_base'][1]}, {r['v2_score_finetuned_minus_base'][2]}].",
        "",
        "## Does ROUGE-L agree with me?",
        "",
        f"- On the {r['rouge_pairs_compared']} pairs where I picked a winner and ROUGE-L "
        f"differed, ROUGE-L picked my winner {r['rouge_picks_my_winner_pct'][0]}% of the "
        f"time [{r['rouge_picks_my_winner_pct'][1]}, {r['rouge_picks_my_winner_pct'][2]}].",
        f"- Per subject (v2, n = {r['v2_score_vs_rougeL']['subjects']}), Spearman's rho "
        f"between my score and ROUGE-L is {r['v2_score_vs_rougeL']['spearman'][0]} "
        f"[{r['v2_score_vs_rougeL']['spearman'][1]}, {r['v2_score_vs_rougeL']['spearman'][2]}]"
        f" (Kendall's tau-b {r['v2_score_vs_rougeL']['kendall_tau_b']}).",
        f"- My score against subject length: rho {r['v2_score_vs_words_spearman'][0]} "
        f"[{r['v2_score_vs_words_spearman'][1]}, {r['v2_score_vs_words_spearman'][2]}].",
        f"- ROUGE-L treats the author's subject as the answer, but I rated it 1 or 2 in "
        f"{r['reference_rated_1_or_2_pct'][0]}% of v2 items "
        f"[{r['reference_rated_1_or_2_pct'][1]}, {r['reference_rated_1_or_2_pct'][2]}]. "
        f"I scored the fine-tuned subject above the author's in "
        f"{r['score_vs_reference']['finetuned']['higher']} of 100 items, and the base "
        f"subject above it in {r['score_vs_reference']['base']['higher']}.",
        "",
        "## Do the rule-based metrics agree with my reasons?",
        "",
        f"- I flagged {mv['my_vague_flags']} of {mv['subjects']} subjects as vague. The "
        f"`specific` rule still called {mv['metric_specific_on_my_vague_pct']}% of those "
        f"specific, against {mv['metric_specific_on_rest_pct']}% of the rest.",
        f"- Mean score of subjects the rule calls specific vs not: "
        f"{mv['mean_score_specific_vs_not'][0]} vs {mv['mean_score_specific_vs_not'][1]}.",
        f"- The hallucination rule fired on {mv['metric_hallucinated']} subjects; I had "
        f"flagged {mv['metric_hallucinated_that_i_flagged_inaccurate']} of them inaccurate. "
        f"I flagged {mv['my_inaccurate_flags']} subjects inaccurate in all: most inaccuracy "
        "is a wrong description of the change, not an invented name.",
        "",
        "## The ten largest disagreements",
        "",
        "Where ROUGE-L preferred one subject by the widest margin and I preferred the other.",
        "",
        "| My pick | Fine-tuned (ROUGE-L) | Base (ROUGE-L) | Author's subject |",
        "| --- | --- | --- | --- |",
    ]
    for d in r["largest_disagreements"]:
        lines.append(
            f"| {d['my_pick']} | {d['finetuned']} ({d['rougeL_finetuned']}) | "
            f"{d['base']} ({d['rougeL_base']}) | {d['reference']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, default=HUMAN / "analysis.json")
    parser.add_argument("--markdown", type=Path, default=EVAL / "rouge_vs_human.md")
    args = parser.parse_args(argv)
    report = analyse()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    args.markdown.write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()

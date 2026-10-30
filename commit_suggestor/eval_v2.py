"""Compare base, v1 and SFT-v2 on the same 450 test diffs, every number with a CI.

Systems and where their subjects come from:
  base, v1            eval/predictions.jsonl (MLX 4-bit, greedy: the v1 evaluation)
  sft_v2, sft_v2_ctx  generated here from the converted MLX 4-bit model, the one
                      the CLI ships; _ctx adds the repo's five previous subjects
  *_fp16              the Kaggle job's fp16 predictions of the same model, kept
                      only to measure what the 4-bit conversion changed

    uv run python -m commit_suggestor.eval_v2 generate --model data/models/sft-v2-mlx --name sft_v2
    uv run python -m commit_suggestor.eval_v2 generate --model ... --name sft_v2_ctx --context
    uv run python -m commit_suggestor.eval_v2 import-kaggle --path data/kaggle-output/full
    uv run python -m commit_suggestor.eval_v2 report   # eval/v2/results.json, eval/v2_results.md
    uv run python -m commit_suggestor.eval_v2 judge --a sft_v2 --b v1          # dry run
    uv run python -m commit_suggestor.eval_v2 judge --a sft_v2 --b v1 --yes    # paid
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
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
from commit_suggestor.metrics import per_example
from commit_suggestor.prompt import build_messages, clean_reply
from commit_suggestor.stats import stratified_bootstrap, wilson

EVAL = Path(__file__).resolve().parent.parent / "eval"
OUT = EVAL / "v2"
FROM_V1 = {"base": "base", "v1": "finetuned"}


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()]


#: Validation diffs used for every choice made in v2 (repo context, DPO beta),
#: so that the test set is only ever reported, never tuned on.
N_SELECT = 300


def test_set() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    tests = _load(EVAL / "test.jsonl")
    contexts = {c["sha"]: c["context"] for c in _load(EVAL / "test_context.jsonl")}
    return tests, contexts


def split_set(split: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if split == "test":
        return test_set()
    rows = _load(OUT / "valid_set.jsonl")[:N_SELECT]
    return rows, {r["sha"]: r["context"] for r in rows}


def pred_path(name: str, split: str) -> Path:
    return OUT / (f"pred_{name}.jsonl" if split == "test" else f"valid_pred_{name}.jsonl")


def predictions(name: str, split: str = "test") -> list[str]:
    """One subject per example of the split, in its file order."""
    rows_in_order, _ = split_set(split)
    if name in FROM_V1 and split == "test":
        rows = _load(EVAL / "predictions.jsonl")
        assert [r["sha"] for r in rows] == [t["sha"] for t in rows_in_order]
        return [r[FROM_V1[name]] for r in rows]
    by_sha = {r["sha"]: r["prediction"] for r in _load(pred_path(name, split))}
    return [by_sha[t["sha"]] for t in rows_in_order]


def generate(args: argparse.Namespace) -> None:
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    tests, contexts = split_set(args.split)
    model, tokenizer = load(str(args.model))[:2]
    greedy = make_sampler(temp=0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with pred_path(args.name, args.split).open("w") as fh:
        for i, t in enumerate(tests, 1):
            messages = build_messages(t["diff"], contexts[t["sha"]] if args.context else None)
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            reply = mlx_generate(
                model, tokenizer, prompt=prompt, max_tokens=40, sampler=greedy, verbose=False
            )
            fh.write(json.dumps({"sha": t["sha"], "prediction": clean_reply(reply)}) + "\n")
            if i % 50 == 0:
                print(f"  {args.name}: {i}/{len(tests)}", flush=True)
    print(f"{args.name}: {(time.time() - started) / len(tests):.2f} s per example")


def import_kaggle(args: argparse.Namespace) -> None:
    rows = _load(Path(args.path) / "predictions_test.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    for with_context, name in ((False, "sft_v2_fp16"), (True, "sft_v2_ctx_fp16")):
        with (OUT / f"pred_{name}.jsonl").open("w") as fh:
            for r in rows:
                if r["with_context"] is with_context:
                    fh.write(json.dumps({"sha": r["sha"], "prediction": r["prediction"]}) + "\n")


def copy_rate(preds: Sequence[str], contexts: Sequence[list[str]]) -> list[float]:
    """1 where the subject repeats one of the context subjects it was shown."""
    return [
        float(normalise(p) in {normalise(c) for c in ctx})
        for p, ctx in zip(preds, contexts, strict=True)
    ]


def score_system(name: str) -> tuple[dict[str, Any], dict[str, list[float]]]:
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    tests, contexts = test_set()
    preds = predictions(name)
    refs = [t["subject"] for t in tests]
    repos = [t["repo"] for t in tests]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    values = {
        "rougeL": [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds, strict=True)],
        **per_example(preds, [t["diff"] for t in tests]),
    }
    if name.startswith("sft_v2_ctx"):
        values["copied_context_pct"] = copy_rate(preds, [contexts[t["sha"]] for t in tests])
    summary: dict[str, Any] = {
        "bleu": round(BLEU().corpus_score(list(preds), [refs]).score, 2),
    }
    for metric, vals in values.items():
        factor = 1 if metric == "words" else 100
        summary[metric] = {
            "micro": stratified_bootstrap(vals, repos).scaled(factor),
            "macro": stratified_bootstrap(vals, repos, macro=True).scaled(factor),
        }
    return summary, values


def available() -> list[str]:
    names = list(FROM_V1)
    for name in ("sft_v2", "sft_v2_ctx", "sft_v2_fp16", "sft_v2_ctx_fp16"):
        if (OUT / f"pred_{name}.jsonl").exists():
            names.append(name)
    return names


def report(_: argparse.Namespace) -> None:
    tests = test_set()[0]
    repos = [t["repo"] for t in tests]
    names = available()
    scored = {n: score_system(n) for n in names}
    result: dict[str, Any] = {"test_examples": len(tests), "systems": {}, "paired": {}}
    for name, (summary, _values) in scored.items():
        result["systems"][name] = summary
    for a, b in (("sft_v2", "v1"), ("sft_v2", "base"), ("sft_v2_ctx", "sft_v2"), ("v1", "base")):
        if a in scored and b in scored:
            va, vb = scored[a][1], scored[b][1]
            result["paired"][f"{a}_minus_{b}"] = {
                metric: stratified_bootstrap(
                    [x - y for x, y in zip(va[metric], vb[metric], strict=True)], repos
                ).scaled(1 if metric == "words" else 100)
                for metric in (
                    "rougeL",
                    "specific_pct",
                    "generic_pct",
                    "hallucinated_pct",
                    "well_formed_pct",
                    "words",
                )
            }
    for quantised, fp16 in (("sft_v2", "sft_v2_fp16"), ("sft_v2_ctx", "sft_v2_ctx_fp16")):
        if quantised in scored and fp16 in scored:
            same = [
                float(normalise(x) == normalise(y))
                for x, y in zip(predictions(quantised), predictions(fp16), strict=True)
            ]
            result.setdefault("conversion", {})[quantised] = {
                "same_subject_as_fp16_pct": stratified_bootstrap(same, repos).scaled(100),
                "rougeL_4bit_minus_fp16": stratified_bootstrap(
                    [
                        x - y
                        for x, y in zip(
                            scored[quantised][1]["rougeL"], scored[fp16][1]["rougeL"], strict=True
                        )
                    ],
                    repos,
                ).scaled(100),
            }
    for path in sorted(OUT.glob("judge_*.json")):
        result.setdefault("judge", {})[path.stem.removeprefix("judge_")] = json.loads(
            path.read_text()
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (EVAL / "v2_results.md").write_text(markdown(result))
    print(markdown(result))


def markdown(r: dict[str, Any]) -> str:
    def ci(v: list[float]) -> str:
        return f"{v[0]} [{v[1]}, {v[2]}]"

    cols = [
        ("rougeL", "ROUGE-L"),
        ("specific_pct", "Specific %"),
        ("generic_pct", "Generic %"),
        ("hallucinated_pct", "Invented names %"),
        ("well_formed_pct", "Well-formed %"),
        ("words", "Words"),
    ]
    lines = [
        "# v2 evaluation: base vs v1 vs SFT-v2",
        "",
        f"The same {r['test_examples']} test diffs from 3 held-out projects. Generated by "
        "`uv run python -m commit_suggestor.eval_v2 report`; numbers in `eval/v2/results.json`."
        " Micro means with 95% bootstrap intervals, resampled within each project.",
        "",
        "| System | " + " | ".join(label for _, label in cols) + " | BLEU |",
        "| --- | " + " | ".join("---" for _ in cols) + " | ---: |",
    ]
    for name, s in r["systems"].items():
        cells = " | ".join(ci(s[key]["micro"]) for key, _ in cols)
        lines.append(f"| {name} | {cells} | {s['bleu']} |")
    if r["paired"]:
        lines += [
            "",
            "## Paired differences",
            "",
            "| Comparison | " + " | ".join(label for _, label in cols) + " |",
            "| --- | " + " | ".join("---" for _ in cols) + " |",
        ]
        for name, d in r["paired"].items():
            lines.append(f"| {name} | " + " | ".join(ci(d[key]) for key, _ in cols) + " |")
    if "copied_context_pct" in r["systems"].get("sft_v2_ctx", {}):
        lines += [
            "",
            "Repo context: SFT-v2 repeated one of the shown subjects in "
            f"{ci(r['systems']['sft_v2_ctx']['copied_context_pct']['micro'])}% of answers.",
        ]
    if "conversion" in r:
        lines += ["", "## What 4-bit conversion changed", ""]
        for name, c in r["conversion"].items():
            lines.append(
                f"- {name}: same subject as fp16 in {ci(c['same_subject_as_fp16_pct'])}%"
                f"; ROUGE-L change {ci(c['rougeL_4bit_minus_fp16'])} points."
            )
    if "judge" in r:
        lines += ["", "## LLM judge (Claude Haiku 5.5, both orders, held-out kappa 0.23)", ""]
        for name, j in r["judge"].items():
            lines.append(
                f"- {name}: {j['a']} wins {j['a_wins']}, {j['b']} wins {j['b_wins']}, "
                f"ties {j['ties']}; {j['a']} win rate excluding ties "
                f"{ci(j['a_win_rate_excl_ties_pct'])}%."
            )
    return "\n".join(lines) + "\n"


def judge(args: argparse.Namespace) -> None:
    """Judge system a against b on a split's diffs, with the frozen calibrated judge."""
    tests, _ = split_set(args.split)
    frozen = json.loads(HELDOUT_LOCK.read_text())
    cache = Cache(CACHE)
    j = Judge(
        _lazy_backend(),
        rubric=frozen["rubric"],
        model=frozen["model"],
        effort=frozen["effort"],
        cache=cache,
    )
    items = [
        (t["diff"], a, b)
        for t, a, b in zip(
            tests, predictions(args.a, args.split), predictions(args.b, args.split), strict=True
        )
    ]
    todo = j.uncached(items)
    cost, spent = estimate_usd(todo), cache.spent_usd()
    print(
        f"{len(items)} pairs, {cost['calls']} calls not cached, estimate ${cost['usd']:.4f};"
        f" spent so far ${spent:.4f} of ${BUDGET_USD:.2f}"
    )
    if cost["calls"] and not args.yes:
        print("dry run: nothing was sent. Re-run with --yes to make these calls.")
        return
    if cost["usd"] > args.max_usd or spent + cost["usd"] > BUDGET_USD:
        sys.exit("refused: over --max-usd or the project budget")
    verdicts = j.compare_many(items)
    a_wins = sum(v.winner == "x" for v in verdicts)
    b_wins = sum(v.winner == "y" for v in verdicts)
    decided = a_wins + b_wins
    out = {
        "a": args.a,
        "b": args.b,
        "split": args.split,
        "pairs": len(items),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": len(items) - decided,
        "a_win_rate_excl_ties_pct": wilson(a_wins, decided).scaled(100, 1) if decided else None,
        "position_consistent_pct": round(
            100 * sum(v.consistent for v in verdicts) / len(verdicts), 1
        ),
        "judge": frozen,
        "spent_usd_total": round(cache.spent_usd(), 4),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = "judge" if args.split == "test" else "valid_judge"  # the report reads judge_*
    (OUT / f"{prefix}_{args.a}_vs_{args.b}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def decide_context(_: argparse.Namespace) -> None:
    """Ship repo context or not, by a rule fixed before the validation results were seen.

    Ship with context only if, on the N_SELECT validation diffs, the judge
    prefers it (win rate excluding ties above 50%) and its rate of names not
    in the diff is at most 1 point higher. Otherwise ship without: simpler.
    """
    verdict = json.loads((OUT / "valid_judge_sft_v2_ctx_vs_sft_v2.json").read_text())
    rows, _ = split_set("valid")
    diffs = [r["diff"] for r in rows]
    invented = {
        name: 100
        * sum(per_example(predictions(name, "valid"), diffs)["hallucinated_pct"])
        / len(rows)
        for name in ("sft_v2", "sft_v2_ctx")
    }
    win_rate = verdict["a_win_rate_excl_ties_pct"][0]
    ship = win_rate > 50 and invented["sft_v2_ctx"] - invented["sft_v2"] <= 1
    decision = {
        "rule": (decide_context.__doc__ or "").split("\n\n")[1].split("\n\n")[0],
        "judge_win_rate_ctx_vs_none": verdict["a_win_rate_excl_ties_pct"],
        "invented_names_pct": {k: round(v, 2) for k, v in invented.items()},
        "ship_with_context": ship,
    }
    (OUT / "context_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


CURVE_POINTS = (("step-600", 2400), ("step-2500", 10000), ("final", 27346))


def curve(_: argparse.Namespace) -> None:
    """Does more data help? The three SFT-v2 checkpoints on the 600 validation diffs."""
    from rouge_score import rouge_scorer

    valid = _load(OUT / "valid_set.jsonl")
    preds = _load(OUT / "sft_v2_valid_predictions.jsonl")
    repos = [v["repo"] for v in valid]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    per: dict[str, dict[str, list[float]]] = {}
    result: dict[str, Any] = {"valid_examples": len(valid), "checkpoints": {}}
    for name, examples in CURVE_POINTS:
        by_sha = {p["sha"]: p["prediction"] for p in preds if p["checkpoint"] == name}
        subjects = [by_sha[v["sha"]] for v in valid]
        values = {
            "rougeL": [
                scorer.score(v["subject"], s)["rougeL"].fmeasure
                for v, s in zip(valid, subjects, strict=True)
            ],
            **per_example(subjects, [v["diff"] for v in valid]),
        }
        per[name] = values
        result["checkpoints"][name] = {
            "training_examples": examples,
            **{
                m: stratified_bootstrap(vals, repos).scaled(1 if m == "words" else 100)
                for m, vals in values.items()
                if m
                in (
                    "rougeL",
                    "specific_pct",
                    "generic_pct",
                    "hallucinated_pct",
                    "well_formed_pct",
                    "words",
                )
            },
        }
    first, last = per["step-600"], per["final"]
    result["final_minus_step_600"] = {
        m: stratified_bootstrap(
            [b - a for a, b in zip(first[m], last[m], strict=True)], repos
        ).scaled(1 if m == "words" else 100)
        for m in ("rougeL", "specific_pct", "generic_pct", "words")
    }
    history = json.loads((OUT / "sft_v2_log_history.json").read_text())
    result["eval_loss"] = [
        [e["step"], round(e["eval_loss"], 4)] for e in history if "eval_loss" in e
    ]
    (OUT / "learning_curve.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", type=Path, required=True)
    g.add_argument("--name", required=True)
    g.add_argument("--context", action="store_true")
    g.add_argument("--split", choices=["test", "valid"], default="test")
    g.set_defaults(run=generate)
    k = sub.add_parser("import-kaggle")
    k.add_argument("--path", required=True)
    k.set_defaults(run=import_kaggle)
    sub.add_parser("report").set_defaults(run=report)
    sub.add_parser("curve", help="SFT-v2 checkpoints on validation").set_defaults(run=curve)
    sub.add_parser(
        "decide-context", help="ship repo context or not (validation only)"
    ).set_defaults(run=decide_context)
    jp = sub.add_parser("judge")
    jp.add_argument("--a", required=True)
    jp.add_argument("--b", required=True)
    jp.add_argument("--yes", action="store_true")
    jp.add_argument("--max-usd", type=float, default=0.50)
    jp.add_argument("--split", choices=["test", "valid"], default="test")
    jp.set_defaults(run=judge)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

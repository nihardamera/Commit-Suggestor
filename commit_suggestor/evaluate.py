"""Compare the fine-tuned model against the base model and a trivial baseline.

The test set comes from repositories that were not used for training, so this
measures whether the model learned to describe changes in general, not whether
it memorised one project's habits.

Three systems answer every test diff:
  heuristic   "Update <first changed file>". Sets the floor: any metric score
              a model gets should be read against what this gets for free.
  base        Qwen2.5-Coder-1.5B-Instruct with the same prompt, no fine-tuning.
  finetuned   The same model with the LoRA adapter.

Metrics against the human-written subject:
  ROUGE-L F1  word overlap in order; the standard automatic metric for this task
  BLEU        corpus-level, sacrebleu defaults
  exact match after lowercasing and stripping punctuation
  format      share of answers that are one line of at most 72 characters

Automatic metrics reward wording that matches the original author, which is
not the same as being a good message. `eval/ab_sample.jsonl` holds a blind,
randomised side-by-side sample for rating by hand with `scripts/rate.py`.

Usage:
    python -m commit_suggestor.evaluate --data data/mlx --adapter adapters
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from commit_suggestor.prompt import build_messages, clean_reply

BASE_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
FILE_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)


def heuristic(diff: str) -> str:
    match = FILE_HEADER.search(diff)
    return f"Update {Path(match.group(2)).name}" if match else "Update files"


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def well_formed(text: str) -> bool:
    return bool(text) and "\n" not in text and len(text) <= 72


def generate_all(rows: list[dict], adapter: str | None, max_tokens: int) -> list[str]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(BASE_MODEL, adapter_path=adapter)
    greedy = make_sampler(temp=0.0)
    out = []
    for i, row in enumerate(rows, 1):
        prompt = tokenizer.apply_chat_template(
            build_messages(row["diff"]), add_generation_prompt=True, tokenize=False)
        reply = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                         sampler=greedy, verbose=False)
        out.append(clean_reply(reply))
        if i % 50 == 0:
            print(f"  {'finetuned' if adapter else 'base'}: {i}/{len(rows)}", flush=True)
    return out


def score(refs: list[str], preds: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)]
    return {
        "rougeL_f1": round(100 * sum(rouge) / len(rouge), 2),
        "bleu": round(BLEU().corpus_score(preds, [refs]).score, 2),
        "exact_match_pct": round(100 * sum(normalise(r) == normalise(p)
                                           for r, p in zip(refs, preds)) / len(refs), 2),
        "well_formed_pct": round(100 * sum(map(well_formed, preds)) / len(preds), 1),
        "mean_words": round(sum(len(p.split()) for p in preds) / len(preds), 1),
        "_per_example_rougeL": rouge,
    }


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for mean(b - a), resampling test examples."""
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return round(100 * means[int(0.025 * n)], 2), round(100 * means[int(0.975 * n)], 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/mlx"))
    parser.add_argument("--adapter", default="adapters")
    parser.add_argument("--out", type=Path, default=Path("eval"))
    parser.add_argument("--limit", type=int, default=0, help="evaluate the first N only")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--only", choices=["base", "finetuned"],
                        help="generate one system's predictions and stop")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in (args.data / "test.meta.jsonl").open()]
    if args.limit:
        rows = rows[: args.limit]
    refs = [r["subject"] for r in rows]

    preds: dict[str, list[str]] = {"heuristic": [heuristic(r["diff"]) for r in rows]}
    timings: dict[str, float] = {}
    shas = [r["sha"] for r in rows]
    cache_dir = args.out / "cache"
    cache_dir.mkdir(exist_ok=True)
    systems = [("base", None), ("finetuned", args.adapter)]
    if args.only:
        systems = [s for s in systems if s[0] == args.only]
    for name, adapter in systems:
        # Generations are cached against the exact test set, so retraining the
        # adapter does not mean regenerating the base model's answers.
        cache = cache_dir / f"{name}.json"
        if name == "base" and cache.exists():
            cached = json.loads(cache.read_text())
            if cached["shas"] == shas:
                preds[name], timings[name] = cached["preds"], cached["seconds_per_example"]
                print(f"  base: reusing {len(shas)} cached predictions")
                continue
        start = time.time()
        preds[name] = generate_all(rows, adapter, args.max_tokens)
        timings[name] = round((time.time() - start) / len(rows), 2)
        cache.write_text(json.dumps({"shas": shas, "preds": preds[name],
                                     "seconds_per_example": timings[name]}))
    if args.only:
        print(f"generated {args.only} predictions only; run again without --only to score")
        return

    scores = {name: score(refs, p) for name, p in preds.items()}
    low, high = paired_bootstrap(scores["base"]["_per_example_rougeL"],
                                 scores["finetuned"]["_per_example_rougeL"])

    per_repo: dict[str, dict[str, float]] = {}
    for repo in sorted({r["repo"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["repo"] == repo]
        per_repo[repo] = {"n": len(idx), **{
            name: round(100 * sum(scores[name]["_per_example_rougeL"][i] for i in idx) / len(idx), 2)
            for name in preds}}

    results = {
        "test_examples": len(rows),
        "test_repos": sorted({r["repo"] for r in rows}),
        "base_model": BASE_MODEL,
        "scores": {n: {k: v for k, v in s.items() if not k.startswith("_")}
                   for n, s in scores.items()},
        "rougeL_gain_finetuned_vs_base_95ci": [low, high],
        "seconds_per_example": timings,
        "rougeL_by_repo": per_repo,
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    with (args.out / "predictions.jsonl").open("w") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({"repo": r["repo"], "sha": r["sha"], "reference": refs[i],
                                 **{n: preds[n][i] for n in preds}}) + "\n")

    # Blind side-by-side sample for rating by hand. Order is randomised per
    # example and the key is kept separately so the rater cannot see it.
    rng = random.Random(1)
    sample = rng.sample(range(len(rows)), min(50, len(rows)))
    with (args.out / "ab_sample.jsonl").open("w") as fh, \
         (args.out / "ab_key.jsonl").open("w") as key:
        for i in sample:
            flip = rng.random() < 0.5
            a, b = ("finetuned", "base") if flip else ("base", "finetuned")
            fh.write(json.dumps({"id": i, "diff": rows[i]["diff"],
                                 "A": preds[a][i], "B": preds[b][i]}) + "\n")
            key.write(json.dumps({"id": i, "A": a, "B": b}) + "\n")

    print(json.dumps({k: results[k] for k in ("scores", "rougeL_gain_finetuned_vs_base_95ci",
                                               "seconds_per_example")}, indent=2))


if __name__ == "__main__":
    main()

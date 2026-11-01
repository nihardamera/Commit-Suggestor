"""Write docs/MODEL_CARD.md from the committed evaluation files.

    uv run python scripts/model_card.py

Every number in the card is read from a file under eval/ or docs/, so the card
cannot drift from the evidence; re-run it after any evaluation changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def ci(v: list[float], unit: str = "") -> str:
    return f"{v[0]}{unit} [{v[1]}, {v[2]}]"


def main() -> None:
    r = load("eval/v2/results.json")
    systems, paired, judge = r["systems"], r["paired"], r["judge"]
    beta = load("eval/v2/beta_decision.json")
    beta_rule = (
        beta["rule"]
        .replace("the N_SELECT validation", "the 300 validation")
        .replace("MAX_EXTRA_WORDS words", "2 words")
    )
    ctx = load("eval/v2/context_decision.json")
    curve = load("eval/v2/learning_curve.json")
    data = load("docs/data_v2_stats.json")
    pairs = load("eval/dpo/pairs_stats.json")
    loss = load("eval/dpo/loss_check.json")
    gguf = load("eval/v2/gguf_info.json")
    serving = load("eval/serving.json")
    kappa = load("eval/judge/summary.json")
    run = load("eval/v2/dpo_run_info.json")
    first = judge["dpo_guarded_vs_sft_v2_guarded"]
    second = judge["dpo_guarded_vs_sft_v2_guarded_rubric-v2"]
    vs_v1 = judge["dpo_guarded_vs_v1"]
    d = paired["dpo_guarded_minus_sft_v2_guarded"]
    conv = r["conversion"]
    mlx_bench = next(v for v in serving.values() if v["runtime"] == "mlx")

    rows = [
        ("base", "Qwen2.5-Coder-1.5B-Instruct, prompt only"),
        ("v1", "v1: QLoRA in MLX, 2,400 examples"),
        ("sft_v2_guarded", "SFT-v2: LoRA, one epoch of the v2 data"),
        ("dpo_guarded", f"**this model**: SFT-v2 + DPO (beta {beta['beta']:g})"),
    ]
    cols = [
        ("rougeL", "ROUGE-L"),
        ("specific_pct", "Specific %"),
        ("generic_pct", "Generic %"),
        ("hallucinated_pct", "Names not in diff %"),
        ("well_formed_pct", "Well-formed %"),
        ("words", "Words"),
    ]
    table = [
        "| System | " + " | ".join(c for _, c in cols) + " |",
        "| --- | " + " | ".join("---:" for _ in cols) + " |",
    ]
    for name, label in rows:
        s = systems[name]
        table.append(f"| {label} | " + " | ".join(ci(s[k]["micro"]) for k, _ in cols) + " |")

    def judge_row(label: str, how: str, j: dict[str, Any]) -> str:
        counts = f"{j['a_wins']} / {j['b_wins']} / {j['ties']}"
        return f"| {label} | {how} | {counts} | {ci(j['a_win_rate_excl_ties_pct'], '%')} |"

    judge_rows = "\n".join(
        [
            judge_row("this model vs SFT-v2", "rubric v1 (also labelled the DPO pairs)", first),
            judge_row(
                "this model vs SFT-v2", "rubric v2 (length-neutral; labelled no pairs)", second
            ),
            judge_row("this model vs v1", "rubric v1", vs_v1),
        ]
    )
    kept = pairs["reasons"]["kept"]
    heldout_kappa = kappa["heldout"]["kappa_3class"]
    card = f"""# Commit Suggestor (v2)

Writes the subject line of a git commit from its diff. A LoRA fine-tune of
[Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct),
first supervised (SFT) and then preference-tuned (DPO), shipped as MLX 4-bit
for Apple Silicon and GGUF Q4_K_M for llama.cpp on any CPU.

Code, data pipeline and every evaluation file:
[github.com/nihardamera/Commit-Suggestor](https://github.com/nihardamera/Commit-Suggestor).

## Use

```console
$ pipx install "commit-suggestor[mlx] @ git+https://github.com/nihardamera/Commit-Suggestor"
$ # off a Mac: commit-suggestor[cpu] instead of [mlx]
$ commit-suggestor download
$ git add -p && commit-suggestor suggest
```

`commit-suggestor install-hook` pre-fills the message on `git commit`, with a
20-second limit (`COMMIT_SUGGESTOR_TIMEOUT`) and never a download inside a
commit. The hook loads the model afresh for each commit, so it waits several
seconds on a Mac and more on a CPU. The prompt is
`commit_suggestor/prompt.py` (prompt version 2, without repository context);
asking the model any other way is testing a prompt it was not trained on.
Subjects over 72 characters are shortened at a clause boundary
(`prompt.fit_subject`), as in every number below marked "guarded".

## Results

{r["test_examples"]} test commits from three projects never seen in training
(django-rest-framework, pydantic, sphinx). The reference is the author's own
subject. Means with 95% bootstrap intervals, resampled within each project.
Shipped artefacts as users run them: MLX 4-bit, greedy, guarded.

{chr(10).join(table)}

DPO against SFT-v2, paired on the same diffs: specific
{ci(d["specific_pct"])} points, generic {ci(d["generic_pct"])}, ROUGE-L
{ci(d["rougeL"])}, names not in the diff {ci(d["hallucinated_pct"])}, length
{ci(d["words"])} words.

**LLM judge** (Claude Haiku 5.5, pairwise, both orders; a split verdict is a tie):

| Comparison | Judge | Wins / losses / ties | Win rate, ties excluded |
| --- | --- | --- | --- |
{judge_rows}

The judge agrees with one human rater (the author) only moderately: Cohen's
kappa {heldout_kappa} on held-out labels, above ROUGE-L's but far from the 0.6
aimed for. Read the win rates as a strong signal from a noisy instrument, and
the second rubric as a check that DPO did not just learn the labelling
judge's taste. Both judges are the same model, so they share its biases.

**Specific** means the subject names a file or symbol changed in the diff;
**generic** means it would fit almost any diff, or was given to three or more
diffs. Both are rule-based (`commit_suggestor/metrics.py`), and the specificity
rule failed a check against human ratings in v1, so treat it as a proxy.

## Training

- **Data** ([data card](https://github.com/nihardamera/Commit-Suggestor/blob/main/docs/DATA_CARD.md)):
  commits mined from {len(data["mined_per_repo"])} other Python projects, filtered (bots,
  merges, releases, huge diffs, trivial subjects), near-duplicates removed with
  MinHash (Jaccard >= {data["settings"]["near_dup_jaccard"]}), capped at
  {data["settings"]["per_project_cap"]:,} per project. Split by project, never by commit.
- **SFT-v2**: LoRA r 8, alpha 160, the top 16 of 28 layers, all 7 linear
  projections (5.28M trainable parameters, 0.34%), learning rate 5e-5, one
  epoch, loss on the subject only. TRL on a Kaggle T4. More data helped on
  validation: ROUGE-L {ci(curve["checkpoints"]["step-600"]["rougeL"])} at 2,400
  examples, {ci(curve["checkpoints"]["final"]["rougeL"])} at the full epoch.
- **DPO**: 4 samples per training diff (T 0.9), ranked by the judge in both
  orders; {kept:,} pairs kept where both orders agreed and the winner neither
  invented names nor was generic. Chosen subjects are only
  {ci(pairs["chosen_minus_rejected_words"])} words longer than rejected ones. One epoch,
  learning rate 2e-5, sigmoid loss, beta in {{0.05, 0.1, 0.3}};
  {round(run["total_seconds"] / 3600, 1)} h on a T4 for all three. The loss was also
  written by hand (`commit_suggestor/dpo.py`) and matches TRL's on trained
  models within {max(c["abs_difference"] for c in loss):.4f}.
- **Beta** was chosen on 300 validation diffs by a rule fixed in code before
  the results were seen: {beta_rule} Result: beta {beta["beta"]:g}.
- **Repository context** (the last five subjects as extra input) was trained in
  but not shipped: on validation the judge preferred subjects without it
  ({ci(ctx["judge_win_rate_ctx_vs_none"], "%")} win rate for context).

## Conversion and serving

- MLX 4-bit gives the same subject as fp16 in only
  {ci(conv["dpo"]["same_subject_as_fp16_pct"], "%")} of diffs (greedy decoding
  flips on near-ties); quantisation costs no quality: ROUGE-L against fp16
  {ci(conv["dpo"]["rougeL_minus_fp16"])} points.
- GGUF Q4_K_M ({gguf["gguf_bytes"] / 1e6:.0f} MB, llama.cpp {gguf["llama_cpp"]}):
  same subject as fp16 in {ci(conv["dpo_gguf"]["same_subject_as_fp16_pct"], "%")};
  ROUGE-L against fp16 {ci(conv["dpo_gguf"]["rougeL_minus_fp16"])} points.
  {gguf["seconds_per_subject_cpu"]} s per subject on {gguf["cpu_count"]} Kaggle CPU cores,
  after loading. On a slow CPU the hook can pass its 20-second limit; raise it
  with `COMMIT_SUGGESTOR_TIMEOUT`.
- MLX on an {mlx_bench["machine"]}: {mlx_bench["cold_load_s"]} s to load,
  then {mlx_bench["warm_p50_s"]} s per subject (p95 {mlx_bench["warm_p95_s"]} s),
  {mlx_bench["peak_rss_mb"]:,} MB peak memory.

## Limitations

- Trained and tested on Python projects with English subjects. Other
  languages and commit conventions are untested.
- Subjects only: no message body.
- DPO made subjects longer ({systems["dpo_guarded"]["words"]["micro"][0]} words against
  {systems["sft_v2_guarded"]["words"]["micro"][0]}), and unguarded it writes past 72
  characters in {round(100 - systems["dpo"]["well_formed_pct"]["micro"][0], 1)}% of test
  subjects; the guard cuts those.
- It names an identifier that is not in the diff in
  {systems["dpo_guarded"]["hallucinated_pct"]["micro"][0]}% of test subjects
  (the authors' own subjects: about 5%). Read the suggestion before committing it.
- Human preference for DPO over SFT-v2 has not been measured yet.

## Privacy

The CLI runs locally and sends nothing anywhere. The opt-in feedback log
(accepted / edited / replaced) stays on your machine. The demo Space is a static
page of stored test-set subjects: it runs no model and takes no input.
"""
    (ROOT / "docs" / "MODEL_CARD.md").write_text(card)
    print(f"wrote docs/MODEL_CARD.md ({len(card.split())} words)")


if __name__ == "__main__":
    main()

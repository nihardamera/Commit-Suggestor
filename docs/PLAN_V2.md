# Commit Suggestor v2: reviewed plan (production-grade, 7 days)

## Context

`02_COMMIT_SUGGESTOR_V2.md` sets out v2 of `nihardamera/Commit-Suggestor`:
honest evaluation (ROUGE vs a calibrated LLM judge vs your own labels), then
SFT-v2 and DPO, then shipping on Hugging Face. You asked me to check the plan
against the repo, find gaps, and add whatever makes it production-grade
(evals plus everything after inference). Decisions taken: **extend to ~7
days** (P1 moves from 31 Oct to ~2 Nov), add **repo-style conditioning** and
**implicit feedback from the hook** as innovations, and **choose the judge
provider at M1** after pricing the options.

## 1. Repo check (cloned read-only, HEAD `61788a1`)

Everything the plan claims checks out: the modules, 20 tests, `results.json`
numbers, the 3 test repos, 450 test examples, 50 A/B pairs with no
`ab_ratings.jsonl`, the adapter on release `v1.0`, and `lora.yaml` (rank 8,
scale 20, 16 layers, 600 iterations, `mask_prompt`). Points the plan doesn't mention:

- The test set is **57% sphinx** (257 of 450). The pooled score is mostly sphinx's.
- `mine.py` doesn't record commit dates or parents, and `prepare.py` shuffles
  away the commit order. Repo-style conditioning needs both (fix in M2).
- Duplicate removal is exact sha1 only, so near-duplicates get through.
- `prepare-commit-msg` has **no timeout**, and on first use it **downloads
  1 GB inside `git commit`**. `get_adapter.sh` uses `curl` with no checksum.
- MLX only, so it runs on Apple Silicon and nowhere else. Nothing runs the
  real model in CI.

## 2. What the committed predictions already say (my analysis of `eval/predictions.jsonl`)

| Signal | base | v1 fine-tuned | Meaning |
|---|---|---|---|
| ROUGE-L by output length | flat | **2-word outputs score lowest (22)**; 4–8 words 29–45 | "ROUGE rewards short subjects" looks **wrong**: F1 penalises short outputs through recall |
| Content-only ROUGE (drop first word and stopwords) | 20.6 | 26.8 | Most of the gain is **content**, not just picking the right verb |
| Opening verb | "Update" 58% | Add/Fix/Update/Remove | Base defaults to "Update"; fine-tuning fixed that |
| Identifiers not in diff (50 A/B sample) | 2 of 21 | **0 of 24** | Hallucinated identifiers are **already rare**, so the resume line's "a% → b%" may be flat |
| References | — | — | Include lowercase "fix", past-tense "Added/Fixed" and "try switching…", which **break the instructed format** |

What follows from this: the plan's story ("my headline metric was
misleading") is a **conclusion written before the data**, and the first data
partly argues against it. Treat it as a hypothesis. The likelier finding is
"ROUGE ranks the systems correctly but can't tell a specific subject from a
generic one, and it punishes outputs that are better than a noisy
reference". Hallucination needs a partner metric, **specificity**, or DPO
will have nothing to move.

## 3. Method flaws in the plan, and fixes (must-do)

1. **Goodhart in DPO.** The same judge labels the DPO pairs *and* scores DPO's
   win rate, so DPO is optimised against its evaluator. Fix: the final
   evaluation uses a **second judge** (different prompt, and ideally a
   different provider or model tier) as well as the labelling judge, plus
   fresh human pairs. Report both judges' numbers.
2. **β chosen on the test set.** "DPO (best β)" on the 450 test examples leaks
   the test set. Fix: choose β on the **validation projects** (attrs, typer;
   300 examples) by judge win rate, then run the test **once**.
3. **The judge prompt overfits its calibration labels.** If the rubric is
   tuned until κ ≥ 0.6 on the same labels it is scored on, κ is inflated.
   Fix: split the human pairwise labels into **dev (~⅔, tune here)** and
   **held-out (~⅓, report κ here once)**.
4. **No ceiling for κ.** Fix: re-rate 30 items blind after ≥ 1 day to get
   your **intra-rater κ**. The judge can't be expected to beat how
   consistent you are with yourself, so report both.
5. **Statistics.** Use a bootstrap **stratified by repo** and report a
   **macro** (per-repo mean) as well as the micro score, because of the 57%
   sphinx share. Human win rates get Wilson CIs. Write the power note in the
   README: with 50 human pairs the 95% CI half-width is about ±14 points, so
   say so, or rate 80–100 pairs if time allows. Use 3-class κ (A/B/tie) and
   publish the confusion matrix.
6. **Evaluate what you ship.** v2 trains in bf16 on Kaggle but runs as MLX
   4-bit or GGUF Q4. The eval matrix scores the **converted artefacts** on one
   local runtime, and adds a **conversion-drift** row (bf16 vs MLX-4bit vs
   GGUF on the same 450).
7. **Confounded v1→v2.** v2 changes data, amount, framework and precision all
   at once. Fix: keep the **same base (Qwen2.5-Coder-1.5B-Instruct)**; a newer
   small coder can be an extra row only. Save checkpoints at about 2.4K, 10K
   and full-epoch examples and score them on validation. That learning curve
   tests "does more data help" with no extra GPU time.
8. **The DPO judge budget doesn't fit as written.** 3,000 diffs × 6 pairs × 2
   orders = 36K calls. Fix: **one listwise call ranks all 4 samples, run in 2
   orderings** (6K calls), and keep best ≻ worst only where both orderings
   agree. Use batch APIs and prompt caching. Rough total for M1 + M2 + M3 ≈ 15M
   input tokens; price it at M1 and cut to 2,000 diffs if it goes over $5.
   **Ask before the first paid call.**

## 4. Revised milestones (~7 days)

Reuse throughout: `paired_bootstrap`, `score`, and the generation cache in
`commit_suggestor/evaluate.py`; `clean_diff` / `keep` in `prepare.py`;
`build_messages` / `clean_reply` in `prompt.py`; the resumable rater pattern
in `scripts/rate.py`.

**Day 1: M0 baseline + M1 start**
- `pyproject.toml` (uv, Python 3.12) with extras `[mlx]`, `[cpu]`, `[eval]`,
  `[train]`; ruff; mypy strict on new modules; pre-commit with gitleaks;
  `WORKLOG.md`, `RESUME_CLAIMS.md`; the 20 tests stay green.
- GitHub Actions job 1: lint, types, tests, and the scorer smoke test on 20 cached predictions.
- You rate the 50 A/B pairs (`scripts/rate.py`, then `--score`).
- `commit_suggestor/metrics.py`: hallucinated-identifier rate (backticks,
  dotted, snake/Camel, file names, checked against the whole diff),
  **grounded-specificity rate** (names a changed file or symbol),
  **generic-subject rate** (a template list plus "the same subject on ≥ 3
  different diffs"), format rate, length distribution, opening-verb
  distribution, plus raw vs guarded well-formed.

**Day 2: M1 judge + "what ROUGE missed"**
- `scripts/rate_v2.py`: 100 new test diffs, with base, v1 and the reference
  shuffled blind. One keystroke per answer: 1–5 overall for each candidate,
  an optional flag key (inaccurate / vague / malformed), and a best-of-3 pick,
  which gives the pairwise labels. Resumable and saved after every key.
- `commit_suggestor/judge.py`: pairwise, position-swapped (disagreement = tie),
  non-Qwen family, provider-agnostic client, response cache on disk keyed by
  hash, a dry-run cost estimator, and a recorded-response mock for CI.
- `commit_suggestor/stats.py`: stratified bootstrap, Wilson, 3-class κ,
  Spearman/Kendall.
- `eval/rouge_vs_human.md`: rank correlation, the 10 largest disagreements,
  and how often you preferred a model over the reference.
- *Gate* (yours) plus: why tune on dev and report on held-out; what the
  intra-rater κ ceiling means.

**Days 3–4: M2 data audit + SFT-v2 with repo-style conditioning**
- `mine.py`: record `%aI` date and parents. `prepare.py`: MinHash
  near-duplicate removal (`datasketch` or a ~40-line shingle hash),
  trivial-subject filter, per-project cap to balance projects; `data/CARD.md`
  with before/after counts.
- **Innovation A, repo-style conditioning.** `prompt.py` gets
  `build_messages(diff, recent_subjects=None)` and a `PROMPT_VERSION`, and v1's
  prompt stays reproducible. Each training example carries the 5 subjects of
  the commits **before** it in the same repo (no future leakage), with 50%
  context dropout so it also works without history. The CLI gets them from
  `git log -5 --format=%s`. One trained model gives the ablation (with vs
  without context) on test: judge, ROUGE, a style-match metric (case, tense,
  `type:` prefix), and a **copy rate** (n-gram overlap with the context
  subjects) to catch the model copying old subjects.
- Kaggle notebook `notebooks/sft_v2.ipynb` (Unsloth or TRL; check versions
  and the GPU quota first): completion-only loss, ≥ 1 epoch, validation
  curve logged, checkpoints for the learning curve.
- Convert to MLX 4-bit and GGUF Q4_K_M. Conversion-drift row.
- `commit_suggestor/backends.py`: `mlx` and `llamacpp` behind one
  `generate(messages)`; `suggest.py` and `evaluate.py` both use it.
- Eval matrix (base / v1 / SFT-v2 ± context) with stratified CIs and macro averages.
- *Gate* (yours) plus: why evaluate the quantised artefact.

**Days 5–6: M3 DPO**
- Sample 4 outputs per **training** diff (T = 0.8–1.0) on Kaggle (vLLM if it
  fits on a T4). Listwise judge labels in 2 orderings; drop ties, pairs that
  disagree, and both-bad pairs (judge flags plus `metrics.py`). You spot-check
  50 pairs; report your agreement with the judge on them.
- `commit_suggestor/dpo_loss.py` (~30 lines) and
  `tests/test_dpo_loss.py`, matching TRL's loss on fixed tensors within 1e-6.
  Then TRL's `DPOTrainer`.
- β ∈ {0.05, 0.1, 0.3}, selected on **validation**. Log reward margins,
  chosen/rejected log-probs and length. Exploitation checks: mean length,
  identifier count, generic rate, specificity vs SFT-v2.
- Final test table, run once: base / v1 / SFT-v2 / DPO. Labelling-judge
  win rate and **second-judge win rate** vs SFT-v2 (CI), human win rate on
  fresh blind pairs (Wilson CI), ROUGE-L micro and macro, hallucinated and
  specific rates, mean length. If DPO doesn't help, the README says so.

**Day 7: M4 ship (production)**
- **HF Hub**: merged model and GGUF, model card (data, filters, split, prompt
  version, training, eval with CIs, limitations, intended use), pinned revision.
- **CLI**: `pipx install git+…` gives `commit-suggestor suggest | install-hook
  | stats | download`. The backend is chosen automatically (MLX on Apple
  Silicon, llama.cpp elsewhere). Downloads go through `huggingface_hub` with a
  pinned revision and sha256 check, replacing `get_adapter.sh`.
- **Hook hardening**: hard timeout (default 8 s, configurable), **never
  downloads inside a commit** (prints a one-line hint and skips), skips huge
  diffs, `COMMIT_SUGGESTOR_DISABLE=1`, and a `#` comment line naming the model
  version.
- **Innovation B, implicit feedback.** Opt-in with `git config
  commit-suggestor.feedback true`. `prepare-commit-msg` stores
  {diff hash, suggestion, model, latency}; a new `post-commit` hook compares
  it with the final subject (accepted / edited, as normalised edit distance /
  replaced) and appends to a local JSONL that never leaves the machine.
  `commit-suggestor stats` shows the acceptance rate. In the README this is
  dogfooding with n = 1 user, not a population metric. Edited pairs are
  recorded as future preference data.
- **Serving benchmark** `eval/serving.json`: cold load, warm p50/p95, peak
  RSS, for MLX on the M4 and GGUF on CPU.
- **CI job 2, model regression gate**: GGUF on CPU for 25 fixed diffs
  (cached on HF), which fails on well-formed < 100% guarded, any hallucinated
  identifier, or ROUGE dropping beyond a tolerance from the committed baseline.
  Check the runner time first.
- **Gradio Space** (free CPU, llama-cpp-python): input size cap, no input
  logging, privacy note.
- README with the results table first, then the story. Blog draft in `docs/`.
  You write "Design decisions"; I review it. Fill `RESUME_CLAIMS.md`.

Every results file records the model revision, `PROMPT_VERSION`, data hash and git sha.

## 5. Revised definition of done (additions in bold)

- [ ] ≥ 150 human ratings; judge κ on **held-out** labels, **plus your intra-rater κ**
- [ ] ROUGE-vs-human analysis with examples, **including how often you preferred a model over the reference**
- [ ] SFT-v2 vs v1 vs base, **stratified CIs and macro averages**, **learning curve**, **conversion drift**, data card
- [ ] **Style-conditioning ablation with copy rate**
- [ ] Hand-written DPO loss + test; β **chosen on validation**; final table with **two judges**, specificity, length and hallucinated identifiers
- [ ] HF model and card, Space, **cross-platform CLI, hardened hook, feedback stats, serving benchmark, CI model gate**
- [ ] README and blog draft; "Design decisions" in your words; `RESUME_CLAIMS.md`

Resume line: keep the template, but fill it from whatever the results turn
out to be. If hallucination stays near 0, the line uses the specificity /
generic-rate change. If DPO doesn't win, it says what was learned.

## 6. Deliberately not doing

Temporal contamination slice (not picked; it remains the cheapest strong
addition if a spare 2 h turns up), body generation, RL (project `04`), a
bigger model for its own sake, an IDE extension.

## 7. Verification

- `uv run pytest` green at every commit (the 20 existing tests plus new ones
  for metrics, stats/κ against a hand-computed 2×2, the DPO loss vs TRL, the
  judge with a mock, and hook timeout/skip behaviour).
- CI jobs 1 and 2 green on GitHub Actions.
- `commit-suggestor suggest` on staged changes on the Mac (MLX), and in a
  Linux container (llama.cpp); `git commit` in a scratch repo with the hook,
  also with the model missing and with a forced slow model, to confirm it
  never blocks.
- Every README/model-card number traced to a file under `eval/` by `RESUME_CLAIMS.md`.
- The Space answers a pasted diff; the HF download checksum verifies.

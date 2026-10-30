# Work log

The plan is [`docs/PLAN_V2.md`](docs/PLAN_V2.md): `02_COMMIT_SUGGESTOR_V2.md`
after review. A new chat resumes from here: read the plan, then the latest
entry below.

## 2026-10-24 (Sat): plan review, M0 started

**Done**
- Reviewed the v2 plan against the repo at `61788a1`. All of the plan's
  claims check out. Scope extended to ~7 days; added repo-style conditioning
  and implicit hook feedback; the judge provider is chosen at M1 after pricing.
- Branch `v2`. Moved to uv + `pyproject.toml` (same pins as the old
  `requirements.txt`; MLX restricted to Apple Silicon by markers), ruff
  (lint + format), mypy, pre-commit (ruff, gitleaks, hygiene hooks), GitHub
  Actions (checks + gitleaks history scan).
- `tests/test_results_reproduce.py`: every number in `eval/results.json`,
  including the bootstrap CI and the per-project scores, recomputes exactly
  from `eval/predictions.jsonl`. 26 tests pass.

**Findings from the committed v1 predictions** (quick analysis; to be redone
properly with `metrics.py` and committed under `eval/`)
- Shorter v1 outputs do *not* score higher on ROUGE-L: 2-word outputs score
  lowest (~22) and 4–8 words ~29–45. The "ROUGE rewards short subjects"
  hypothesis looks wrong.
- Content-only ROUGE (first word and stopwords dropped): base 20.6 → v1 26.8.
  Most of the gain is content, not the opening verb.
- Base opens with "Update" in 58% of answers.
- Identifiers absent from the diff, 50 A/B pairs: base 2 of 21, v1 0 of 24.
  Hallucination is already rare, so specificity / generic-rate is the
  partner metric.
- References break the instructed format (lowercase, past tense).
- The test set is 57% sphinx (257 of 450): report macro averages too.

**Gotchas**
- macOS sets the `hidden` flag on uv's editable `.pth` file and Python
  3.12.14 skips hidden `.pth` files, so `import commit_suggestor` failed under
  pytest. Fixed with `pythonpath = ["."]` in the pytest config. When the CLI
  is packaged (Day 7), test the `pipx` install from outside the repo.

**Done later the same day**
- `eval/test.jsonl`: the 450 test diffs, rebuilt from their source commits
  (`scripts/rebuild_test_set.py`). All 50 diffs in `eval/ab_sample.jsonl`
  match exactly. The diff driver comes from each repo's `.gitattributes`
  (sphinx sets `*.py diff=python`), which changes the `@@` hunk headers, so
  the rebuild checks that file out.
- `commit_suggestor/metrics.py` + `stats.py` (strict mypy, 56 tests).
  `eval/metrics_v1.json`, micro with 95% CI, n = 450:

  | | hallucinated | specific | generic | imperative | well-formed | words |
  |---|---|---|---|---|---|---|
  | reference | 4.9 | 74.2 | 5.1 | 92.7 | 100 | 5.9 |
  | base | 0.4 | 92.4 | 0.0 | 97.1 | 92.2 | 7.2 |
  | v1 | 0.4 | 80.7 | 6.2 | 98.9 | 98.9 | 4.7 |
  | v1 − base | 0.0 [−0.9, 0.9] | −11.8 [−16.0, −7.8] | +6.2 [4.0, 8.7] | +1.8 | +6.7 | −2.5 |

  Reading: v1 is shorter, *less specific and more generic* than base, and
  close to the human references on both. Hallucination is not a problem
  for either model; the references "hallucinate" most, because authors know
  context the diff doesn't show. "Specific" favours longer subjects (more
  words, more chances to match), so check it against the human "vague" flags.
- `scripts/rate_v2.py` (100 new diffs; base / v1 / reference blind) and a
  one-key `scripts/rate.py`. `eval/human/split.json` fixes the judge's
  dev/held-out labels in advance (A/B 33/17, v2 67/33).
- The venv and data now live in `.venv.nosync` / `data.nosync` behind
  symlinks: iCloud Drive syncs ~/Desktop and kept hiding the venv's files.

**Next**
1. Nihar: rate the 50 A/B pairs (`uv run python scripts/rate.py`, ~20 min),
   then the 100 v2 items (`uv run python scripts/rate_v2.py`, ~1.5 h, in
   sittings). On or after 26 Oct: `--rerate` (30 items, ~20 min). Don't
   open `eval/ab_key.jsonl`, `eval/human/v2_key.jsonl` or run `--score`
   until all rating is done.
2. Me: price judge candidates (non-Qwen, ≤ $5 total, batch + caching), then
   `commit_suggestor/judge.py` with a cache, a cost dry-run and a mock.
   Ask before the first paid call.
3. Then `eval/rouge_vs_human.md` and the M1 explain-back gate.

## 2026-10-30 (Fri): human results, judge built

**Done**
- Nihar rated all 50 A/B pairs and all 100 v2 items (26–29 Oct; median
  65 s per v2 item). **Re-rating skipped by decision**, so there is no
  intra-rater κ ceiling for the judge; the README must say that judge κ is
  reported without a self-agreement reference.
- `commit_suggestor/human.py` → `eval/rouge_vs_human.md`, `eval/human/analysis.json`:
  - v1 vs base, all 150 pairs: v1 wins 57.5% [49.0, 65.5] of decided pairs
    (A/B alone 48.9% [35.3, 62.8]; v2 alone 62.1% [51.6, 71.5]).
  - v2 1–5 scores: base 2.49, v1 3.02, author 3.64; v1 − base +0.53 [0.15, 0.90].
  - ROUGE-L picks my winner 60.0% [51.1, 68.3]; Spearman with my scores 0.39 [0.27, 0.50].
  - My score does not track length (ρ 0.02), so my ratings are not length-biased.
  - I rated the author's subject 1–2 in 21% of items.
  - The rule-based `specific` metric fails validation (82% on my "vague"
    subjects vs 81% on the rest). Not used to judge models. Most of my
    "inaccurate" flags (81) are wrong descriptions, not invented names (the
    hallucination rule fired twice).
  - Story so far: ROUGE-L's +10 points [7.8, 12.1] overstates a real but
    much smaller and less certain improvement.
- `commit_suggestor/judge.py`: Claude Haiku 5.5 (pricing checked on
  2026-10-30 at platform.claude.com: $0.10/$0.50 per MTok, batch half price;
  GPT-5.4 mini ~$0.75/$4.50 and current Gemini Flash $0.30–1.50 input, both
  from aggregator sites). Pairwise, both orders, reference-free, JSON output,
  dry run by default, a $5 hard budget, a cost ledger in the cache, and a
  held-out lock. anthropic SDK 2.3.0 (API version 2026-10-21: no
  temperature parameter exists, so judge answers are not deterministic; the
  cache makes reruns reproducible).

**Done later the same day: judge calibrated (M1 complete except the gate)**
- API key saved in the macOS Keychain (service `anthropic-api-key`); commands
  read it with `ANTHROPIC_API_KEY="$(security find-generic-password -s anthropic-api-key -w)"`.
  It is never printed or committed.
- Haiku 5.5's tokenizer packs 2.59 characters into a token on these prompts
  (lowest 2.34), so the cost estimate now assumes 2.2.
- Dev (100 pairs): v1 low effort κ 0.298, v2 rubric 0.246, v1 medium 0.272:
  a plateau (κ standard error ≈ 0.08). On dev, κ is 0.17 on the A/B labels and
  0.36 on the v2 labels. Nihar chose to freeze v1 at low effort.
- Held-out (50 pairs, scored once, locked): **κ 0.232** vs ROUGE-L 0.185.
  Not clearly different at this n. Judge lean toward fine-tuned vs me: +12
  points [−6, 30]. See `eval/judge_vs_human.md`.
- Total API spend: **$0.116 of $5.**
- Fixed: result files now include the effort level (one would have been
  overwritten); CI pinned `astral-sh/setup-uv@v10.3.3` (there is no `@v10` tag).

**What this means for M3 (DPO)**
- At κ ≈ 0.25 the judge's labels are noisy. "DPO on bad pairs makes things
  worse" is a live risk. Mitigations: keep only pairs where both orders agree
  and the judge picks a winner (on dev it then matches me 74% of the time), and
  evaluate DPO with fresh human pairs as the deciding measure, not this judge.
- The judge's possible lean toward short, v1-style subjects can be amplified
  by DPO. Watch mean length and generic rate.

**Next**
1. M1 explain-back gate (Nihar answers unaided): pairwise vs 1–5, position
   bias, κ by hand from a 2×2.
2. Then M2: data audit + SFT-v2 on Kaggle. Check the Kaggle GPU quota and
   the Unsloth/TRL versions first.

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

**Next**
1. Nihar: rate the 50 A/B pairs: `uv run python scripts/rate.py`, then `--score`.
2. Me: `commit_suggestor/metrics.py` (hallucinated identifiers, grounded
   specificity, generic rate, format, length, opening verb) with tests;
   `stats.py`; `scripts/rate_v2.py`.
3. Price the judge candidates (≤ $5 total); ask before the first paid call.

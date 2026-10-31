# Resume claims and their evidence

Nothing goes on the resume or in the README without a row here pointing at a
committed file that proves it.

| Claim | Evidence | Status |
| --- | --- | --- |
| QLoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct (4-bit, MLX) on an M4 | `config/lora.yaml` | v1, verified |
| 122,088 commits mined from 20 Python projects; ~45% kept after filtering | `commit_suggestor/mine.py`, `commit_suggestor/prepare.py`; README | v1; counts come from the run log, not a committed file. M2's data card will commit them |
| Split by project: 15 train / 2 validation / 3 test; 450 test examples | `commit_suggestor/mine.py` `REPOS`; `eval/results.json` | v1, verified |
| ROUGE-L 31.1 vs 21.1 for the base model on 3 held-out projects | `eval/results.json`; recomputed by `tests/test_results_reproduce.py` | v1, verified |
| Gain over base 10.0 ROUGE-L points, 95% paired-bootstrap CI [7.8, 12.1] | `eval/results.json`; `tests/test_results_reproduce.py` | v1, verified |
| Gain holds in each held-out project | `eval/results.json` `rougeL_by_repo` | v1, verified |
| Blind human preference, fine-tuned vs base, 150 pairs: 57.5% [49.0, 65.5]; 1-5 score +0.53 [0.15, 0.90] | `eval/rouge_vs_human.md`, `eval/human/analysis.json`; `tests/test_human.py` recomputes them | measured (one rater: me) |
| ROUGE-L picks the human winner only 60.0% [51.1, 68.3] of the time | `eval/rouge_vs_human.md` | measured |
| LLM judge (Claude Haiku 5.5, pairwise, position-swapped) calibrated to my labels: held-out κ = 0.23 (ROUGE-L 0.19); target 0.6 not reached | `eval/judge_vs_human.md`, `eval/judge/summary.json`; `tests/test_judge.py` recomputes it from the cache | measured; claim it as "κ = 0.23", never as "calibrated judge" alone |
| SFT-v2 (LoRA, TRL, Kaggle T4, 27,356 examples): validation ROUGE-L 26.8 → 30.1 with data (+3.3 [1.7, 4.9]); on test ties v1 (ROUGE-L −0.87 [−2.68, 0.95]; judge 49.8% [44.2, 55.5]) | `eval/v2/learning_curve.json`, `eval/v2/results.json`, `eval/v2/judge_sft_v2_vs_v1.json` | measured |
| Repo-style context trained in, not shipped: judge preferred no context on validation, 27.3% [21.0, 34.7] for context | `eval/v2/context_decision.json` | measured; rule fixed before results |
| DPO on 2,198 judge-labelled pairs; beta 0.3 chosen on validation by a pre-registered rule | `eval/dpo/pairs_stats.json`, `eval/v2/beta_decision.json` | measured |
| DPO vs SFT-v2 on test: judge win rate 83.2% [78.8, 86.9] (labelling rubric), 85.8% [81.7, 89.2] (second rubric that labelled nothing); vs v1 78.3% [73.6, 82.3] | `eval/v2/judge_dpo_guarded_vs_*.json` | measured; same judge model for both rubrics. Human check: `eval/human/dpo_*` pending |
| Hand-written DPO loss matches TRL 1.17 within 0.001 on trained models | `eval/dpo/loss_check.json`, `tests/test_dpo.py` | measured |
| v1 names an identifier absent from the diff in 0.4% of subjects (same as base; human references 4.9%) | `eval/metrics_v1.json`; `tests/test_metrics.py` recomputes it | v1, measured (rule-based; precision not yet checked by hand) |
| v1 vs base: specific −11.8 pts [−16.0, −7.8], generic +6.2 pts [4.0, 8.7] | `eval/metrics_v1.json` | v1, measured (proxy metrics) |
| DPO vs SFT-v2: specific +16.2 pts [12.7, 19.8], generic 8.4% → 0% (−8.4 [−11.1, −6.0]), ROUGE-L +0.97 [−0.79, 2.75] (no change), names not in diff +0.44 [−0.67, 1.78] | `eval/v2/results.json` `paired.dpo_guarded_minus_sft_v2_guarded` | measured (proxy metrics) |
| DPO length regression caught and guarded: 5.6% of raw test subjects > 72 chars; 100% well-formed after `fit_subject` (designed on validation) | `eval/v2/results.json` (`dpo` vs `dpo_guarded`) | measured |
| Serving: MLX 4-bit on M4, 2.7 s cold load, 0.43 s p50 / 0.98 s p95 per subject, 1.4 GB peak | `eval/serving.json` | measured (50 diffs) |
| Model and live demo on Hugging Face | — | pending (M4) |

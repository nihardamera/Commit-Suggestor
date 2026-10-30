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
| SFT-v2 vs v1 vs base with CIs | — | pending (M2) |
| DPO on … judge-labelled pairs; win rate … vs SFT | — | pending (M3) |
| v1 names an identifier absent from the diff in 0.4% of subjects (same as base; human references 4.9%) | `eval/metrics_v1.json`; `tests/test_metrics.py` recomputes it | v1, measured (rule-based; precision not yet checked by hand) |
| v1 vs base: specific −11.8 pts [−16.0, −7.8], generic +6.2 pts [4.0, 8.7] | `eval/metrics_v1.json` | v1, measured (proxy metrics) |
| Hallucinated identifiers / specificity after DPO … → … | — | pending (M3) |
| Model and live demo on Hugging Face | — | pending (M4) |

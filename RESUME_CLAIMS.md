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
| Human preference: fine-tuned vs base | `eval/ab_ratings.jsonl` (not yet rated) | pending (M1) |
| Judge calibrated to my labels, Cohen's κ = … | — | pending (M1) |
| SFT-v2 vs v1 vs base with CIs | — | pending (M2) |
| DPO on … judge-labelled pairs; win rate … vs SFT | — | pending (M3) |
| Hallucinated identifiers / specificity … → … | — | pending (M1–M3) |
| Model and live demo on Hugging Face | — | pending (M4) |

# Commit Suggestor (v2)

Writes the subject line of a git commit from its diff. A LoRA fine-tune of
[Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct),
first supervised (SFT) and then preference-tuned (DPO), shipped as MLX 4-bit
for Apple Silicon and GGUF Q4_K_M for llama.cpp on any CPU.

Code, data pipeline and every evaluation file:
[github.com/nihardamera/Commit-Suggestor](https://github.com/nihardamera/Commit-Suggestor).

## Use

```console
$ pipx install "git+https://github.com/nihardamera/Commit-Suggestor[mlx]"   # [cpu] off a Mac
$ commit-suggestor download
$ git add -p && commit-suggestor suggest
```

`commit-suggestor install-hook` pre-fills the message on `git commit`, with an
8-second limit and never a download inside a commit. The prompt is
`commit_suggestor/prompt.py` (prompt version 2, without repository context);
asking the model any other way is testing a prompt it was not trained on.
Subjects over 72 characters are shortened at a clause boundary
(`prompt.fit_subject`), as in every number below marked "guarded".

## Results

450 test commits from three projects never seen in training
(django-rest-framework, pydantic, sphinx). The reference is the author's own
subject. Means with 95% bootstrap intervals, resampled within each project.
Shipped artefacts as users run them: MLX 4-bit, greedy, guarded.

| System | ROUGE-L | Specific % | Generic % | Names not in diff % | Well-formed % | Words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-1.5B-Instruct, prompt only | 21.09 [19.43, 22.76] | 92.44 [90.0, 94.89] | 0.0 [0.0, 0.0] | 0.44 [0.0, 1.11] | 92.22 [89.56, 94.67] | 7.18 [6.98, 7.38] |
| v1: QLoRA in MLX, 2,400 examples | 31.06 [28.89, 33.2] | 80.67 [77.11, 84.22] | 6.22 [4.0, 8.67] | 0.44 [0.0, 1.11] | 98.89 [97.78, 99.78] | 4.71 [4.54, 4.89] |
| SFT-v2: LoRA, one epoch of the v2 data | 30.16 [27.97, 32.39] | 79.33 [75.56, 82.89] | 8.44 [6.0, 11.11] | 1.11 [0.22, 2.22] | 100.0 [100.0, 100.0] | 4.42 [4.28, 4.58] |
| **this model**: SFT-v2 + DPO (beta 0.3) | 31.13 [29.06, 33.2] | 95.56 [93.56, 97.33] | 0.0 [0.0, 0.0] | 1.56 [0.44, 2.89] | 100.0 [100.0, 100.0] | 6.28 [6.1, 6.46] |

DPO against SFT-v2, paired on the same diffs: specific
16.22 [12.67, 19.78] points, generic -8.44 [-11.11, -6.0], ROUGE-L
0.97 [-0.79, 2.75], names not in the diff 0.44 [-0.67, 1.78], length
1.85 [1.66, 2.04] words.

**LLM judge** (Claude Haiku 5.5, pairwise, both orders; a split verdict is a tie):

| Comparison | Judge | Wins / losses / ties | Win rate, ties excluded |
| --- | --- | --- | --- |
| this model vs SFT-v2 | rubric v1 (also labelled the DPO pairs) | 268 / 54 / 128 | 83.2% [78.8, 86.9] |
| this model vs SFT-v2 | rubric v2 (length-neutral; labelled no pairs) | 291 / 48 / 111 | 85.8% [81.7, 89.2] |
| this model vs v1 | rubric v1 | 267 / 74 / 109 | 78.3% [73.6, 82.3] |

The judge agrees with one human rater (the author) only moderately: Cohen's
kappa 0.232 on held-out labels, above ROUGE-L's but far from the 0.6
aimed for. Read the win rates as a strong signal from a noisy instrument, and
the second rubric as a check that DPO did not just learn the labelling
judge's taste. Both judges are the same model, so they share its biases.

**Specific** means the subject names a file or symbol changed in the diff;
**generic** means it would fit almost any diff, or was given to three or more
diffs. Both are rule-based (`commit_suggestor/metrics.py`), and the specificity
rule failed a check against human ratings in v1, so treat it as a proxy.

## Training

- **Data** ([data card](https://github.com/nihardamera/Commit-Suggestor/blob/main/docs/DATA_CARD.md)):
  commits mined from 17 other Python projects, filtered (bots,
  merges, releases, huge diffs, trivial subjects), near-duplicates removed with
  MinHash (Jaccard >= 0.8), capped at
  4,000 per project. Split by project, never by commit.
- **SFT-v2**: LoRA r 8, alpha 160, the top 16 of 28 layers, all 7 linear
  projections (5.28M trainable parameters, 0.34%), learning rate 5e-5, one
  epoch, loss on the subject only. TRL on a Kaggle T4. More data helped on
  validation: ROUGE-L 26.8 [24.86, 28.74] at 2,400
  examples, 30.1 [28.12, 32.13] at the full epoch.
- **DPO**: 4 samples per training diff (T 0.9), ranked by the judge in both
  orders; 2,198 pairs kept where both orders agreed and the winner neither
  invented names nor was generic. Chosen subjects are only
  0.38 [0.27, 0.5] words longer than rejected ones. One epoch,
  learning rate 2e-5, sigmoid loss, beta in {0.05, 0.1, 0.3};
  2.1 h on a T4 for all three. The loss was also
  written by hand (`commit_suggestor/dpo.py`) and matches TRL's on trained
  models within 0.0010.
- **Beta** was chosen on 300 validation diffs by a rule fixed in code before
  the results were seen: Each beta is judged against beta 0 (SFT-v2, fp16, same job) on the 300 validation diffs. A beta is eligible only if, compared with beta 0, its rate of names not in the diff and its generic rate are each at most 1 point higher, and its mean length is at most 2 words longer, since length is the first thing DPO exploits. The eligible beta with the highest judge win rate (ties excluded) is chosen. It ships instead of SFT-v2 only if the lower end of that win rate's 95% Wilson interval is above 50%. Result: beta 0.3.
- **Repository context** (the last five subjects as extra input) was trained in
  but not shipped: on validation the judge preferred subjects without it
  (27.3% [21.0, 34.7] win rate for context).

## Conversion and serving

- MLX 4-bit gives the same subject as fp16 in only
  19.56% [16.0, 23.11] of diffs (greedy decoding
  flips on near-ties); quantisation costs no quality: ROUGE-L against fp16
  1.29 [0.05, 2.55] points.
- GGUF Q4_K_M (986 MB, llama.cpp b11701):
  same subject as fp16 in 25.11% [21.33, 29.11];
  ROUGE-L against fp16 2.27 [1.16, 3.41] points.
  7.73 s per subject on 4 Kaggle CPU cores,
  close to the hook's 8-second limit: on a slow CPU, raise it with
  `COMMIT_SUGGESTOR_TIMEOUT`.
- MLX on an Apple M4, 16 GB: 2.68 s to load,
  0.433 s per subject (p95 0.978 s),
  1,446 MB peak memory.

## Limitations

- Trained and tested on Python projects with English subjects. Other
  languages and commit conventions are untested.
- Subjects only: no message body.
- DPO made subjects longer (6.28 words against
  4.42), and unguarded it writes past 72
  characters in 5.6% of test
  subjects; the guard cuts those.
- It names an identifier that is not in the diff in
  1.56% of test subjects
  (the authors' own subjects: about 5%). Read the suggestion before committing it.
- Human preference for DPO over SFT-v2 has not been measured yet.

## Privacy

The CLI runs locally and sends nothing anywhere. The opt-in feedback log
(accepted / edited / replaced) stays on your machine. The demo Space does not
store inputs.

# Commit Suggestor

A small code model, fine-tuned on my Mac, that writes the subject line of a git
commit message from the diff. It runs locally, takes about two seconds,
and can pre-fill the message whenever you run `git commit`.

```console
$ git add -p
$ python -m commit_suggestor.suggest
Add retries to connect()
```

## Results

Evaluated on 450 commits from three projects the model never saw during
training (django-rest-framework, pydantic, sphinx). Each system got the same diff and the same prompt;
the reference is the subject line the original author wrote.

| System | ROUGE-L F1 | BLEU | Exact match | One line, at most 72 chars | Mean words |
| --- | ---: | ---: | ---: | ---: | ---: |
| heuristic | 13.0 | 2.3 | 0.7% | 100.0% | 2.0 |
| base | 21.1 | 5.1 | 0.0% | 92.2% | 7.2 |
| **fine-tuned** | **31.1** | **8.6** | **0.4%** | **98.9%** | **4.7** |

- **heuristic** is `Update <first changed file>`. It sets the floor: a score is
  only meaningful compared with what this gets for nothing.
- **base** is Qwen2.5-Coder-1.5B-Instruct with the prompt alone.
- **fine-tuned** is the same model with the LoRA adapter trained here.

Fine-tuning raised ROUGE-L by 10.0 points over the base model (95% bootstrap interval 7.8 to 12.1, resampling test commits), and the gain holds in each held-out project separately (django-rest-framework 18.1 to 31.0, pydantic 24.6 to 35.2, sphinx 21.5 to 29.9). It also learned the house style: shorter subjects (4.7 words against 7.2) and fewer answers that break the one-line, 72-character rule. Exact matches stay near zero for every system, which is expected: two people rarely word the same change identically.

Automatic metrics compare wording with the original author's, which is not the
same as judging whether a message is good. `eval/ab_sample.jsonl` holds 50 test
diffs with the base and fine-tuned suggestions in random order, for rating by
hand with `python scripts/rate.py` (blind), then `python scripts/rate.py --score`.

Every prediction is in [`eval/predictions.jsonl`](eval/predictions.jsonl) and
the full numbers in [`eval/results.json`](eval/results.json).

A fixed-seed random sample of fifteen test commits, good and bad answers both, is in [`eval/samples.md`](eval/samples.md).

## How it was built

**Data.** `commit_suggestor/mine.py` clones 20 well-maintained Python
projects (shallow, newest 4,000 commits each) and reads every commit's patch
with `git log -p`, which is far faster than asking the GitHub API one commit at
a time and has no rate limit. That produced 122,088 commits.

`commit_suggestor/prepare.py` keeps about 45% of them. The filters are
where most of the quality is decided:

- bots, merges, reverts, releases and version bumps are dropped, because their
  subjects are mechanical;
- pull-request numbers like `(#1234)` are stripped, because they cannot be
  inferred from a diff and a model trained on them learns to invent them;
- commits touching more than four files, or with a diff over 3,000 characters,
  are dropped: the subject of a sprawling change rarely describes most of it;
- lock files, minified files and binary changes are removed from the diff.

The split is by project, not by commit: training uses 15 projects, validation
2 others, and the test set 3 more. Splitting commits at random would put
near-identical changes from the same codebase on both sides and flatter the
score.

**Training.** QLoRA: LoRA adapters trained on top of a 4-bit quantised
`mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`, using Apple's MLX
(`mlx_lm.lora`) on an M4 MacBook with 16 GB of memory. Only the commit subject
counts toward the loss (`mask_prompt`), so the model learns to answer rather
than to reproduce diffs. Rank 8 on the top 16 of 28 layers, which is
0.342% of the weights. I trained for 600 steps of 4 examples (2,400 of the 37,898 training commits), about 105 minutes of compute on the M4. Validation loss went 5.439 at step 1, 2.279 at step 100, 2.387 at step 200, 2.456 at step 300, 2.374 at step 400, 2.250 at step 500, 2.263 at step 600. It was flat after the first 100 steps, so the model learned the task quickly and training longer on more of the data would probably not have helped much; a larger base model or cleaner targets would be the next things to try. Settings are in
[`config/lora.yaml`](config/lora.yaml).

**One prompt everywhere.** Training, evaluation and the command-line tool all
build their input with `commit_suggestor/prompt.py`, and the CLI cleans a diff
with the same function as the training data. A model asked differently from how
it was trained is partly being tested on the mismatch.

An earlier version of this repository planned QLoRA on Mistral-7B with
bitsandbytes. That needs an NVIDIA GPU, which I did not have, and the pipeline
was never run; this version replaces it with one that was.

## Use it

Needs an Apple Silicon Mac (MLX) and Python 3.12.

```bash
git clone https://github.com/nihardamera/Commit-Suggestor.git
cd Commit-Suggestor
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/get_adapter.sh                      # downloads the trained adapter

# in any repository, after staging changes:
PYTHONPATH=/path/to/Commit-Suggestor /path/to/Commit-Suggestor/.venv/bin/python -m commit_suggestor.suggest

# or have `git commit` pre-fill the message:
scripts/install_hook.sh /path/to/your/repo
```

The hook only acts on a plain `git commit` (not `-m`, merges or amends) and
never blocks a commit: if the suggestion fails, the message is left as it was.
The first run downloads the 1 GB base model from Hugging Face.

## Reproduce

```bash
python -m commit_suggestor.mine --out data/raw          # clone and mine
python -m commit_suggestor.prepare --raw data/raw --out data/mlx
python -m mlx_lm lora --config config/lora.yaml         # train
python -m commit_suggestor.evaluate --data data/mlx --adapter adapters
python -m pytest tests
```

## Limitations

- It writes the subject line only, not a body.
- It learns the style of the projects it was trained on, which are Python
  libraries. Diffs over 3,000 characters are cut, so for large changes it sees
  only the start.
- It describes what a diff does, not why. The why is usually the part worth
  writing yourself.
- Apple Silicon only, because training and inference use MLX.

The commit data comes from public repositories under their own open-source
licences; only the trained adapter is distributed here, not the data.

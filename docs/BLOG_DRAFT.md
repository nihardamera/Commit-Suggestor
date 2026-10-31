# My metric said the model didn't improve. It had.

*Draft. Every number links to a file in the repo; check them before publishing.*

I fine-tuned a 1.5B code model to write git commit subjects from diffs. Version
1 looked like a clean win: ROUGE-L went from 21 to 31 over the base model, with
a tight confidence interval. Version 2 was meant to make it production-grade.
Most of the work turned out to be learning what my metric could and couldn't see.

## Step 1: rate it yourself

Before building anything new, I rated 150 pairs blind: base vs fine-tuned, in
random order, one keystroke each.

- I preferred the fine-tuned model **57.5%** of the time [49.0, 65.5]. That's a
  lean, not the landslide that "+10 ROUGE points" suggests.
- On pairs where I had a preference, ROUGE-L agreed with me **60%** of the time.
  Barely better than a coin.
- I rated the original author's subject, which ROUGE treats as *the answer*,
  1 or 2 out of 5 in **21%** of cases: "Update CHANGES for PR", "testing jenkins",
  "Refactor encoded filename feature per".

Fine-tuning had also made subjects shorter and **less specific**. ROUGE likes
that: short, conventional outputs overlap well with short, conventional
references.

## Step 2: build a judge, and measure the judge

An LLM judge (Claude Haiku 5.5) compares two subjects for the same diff. It runs
in both orders, and a split verdict counts as a tie. I split my labels into
dev and held-out sets *before* writing the first prompt. I tuned the rubric on
dev and scored held-out exactly once.

Result: Cohen's κ = **0.23** against my labels (ROUGE-L: 0.19). I'd aimed for 0.6
and didn't get close. I kept the judge anyway, but as a noisy instrument with a
known error bar, not as the truth.

## Step 3: more data — a tie

SFT-v2 trained on 11× more data (27K commits, near-duplicates removed, split by
project). On validation, more data helped: ROUGE-L 26.8 → 29.3 → 30.1 across
checkpoints. On the test projects it **tied** v1, by ROUGE (−0.9, CI spans 0)
and by the judge (49.8%). Data volume wasn't the bottleneck.

## Step 4: DPO — big change, and ROUGE saw nothing

For DPO, I sampled four subjects for each of 3,000 training diffs and had the
judge rank them in both orders. I kept 2,198 pairs where both orders agreed
and the winner was neither generic nor named something absent from the diff.
Chosen subjects were only 0.38 words longer than rejected ones, so the
data doesn't simply say "longer is better".

I chose β on validation diffs, by a rule I committed to code before seeing a
single result. On the test set:

| | SFT-v2 | DPO |
| --- | ---: | ---: |
| Generic subjects | 8.4% | **0.0%** |
| Names a changed file or symbol | 79.3% | **95.6%** |
| Judge prefers (labelling rubric) | | **83.2%** |
| Judge prefers (second rubric, never used in training) | | **85.8%** |
| ROUGE-L | 30.2 | 31.1 (+1.0, CI [−0.8, 2.8]) |

The largest quality change in the project, and ROUGE-L read it as noise.

What it looks like:

- "Fix typo in README" → *Change "level" to "tier" in README*
- "Fix changelog formatting" → *Update changelog wording about attr.ib metadata defaulting to dict*
- "Fix typo in docstring" → *Use :data:`True` and :data:`False` instead of `True` and `False`*

## Step 5: the thing my rule missed

DPO also learned to run long: **5.6%** of test subjects broke the 72-character
limit. My β rule capped the *mean* length increase at 2 words, but it never
checked the hard limit. Validation showed the problem (7 of 300), and I
didn't look.

Re-choosing β after seeing the test set would have turned it into a training
set. So the fix sits in the product instead: the CLI ends a long subject
before its last clause that fits ("… in azure-pipelines.yml" goes). I designed
it on the seven validation cases, and I report raw and guarded numbers side by side.

## What I'd tell myself at the start

1. **Rate a sample by hand before trusting any metric.** 150 ratings took an
   afternoon and reframed the whole project.
2. **Measure your judge like a model.** Use held-out labels, report κ with an
   interval, and use a second rubric when the judge also made your training data.
3. **Write the decision rule before you see the numbers.** It's the only way
   validation stays validation.
4. **Check the hard constraints, not just the averages.** A mean can stay inside
   a limit while the tail breaks it.

The model, the CLI and every evaluation file are on
[GitHub](https://github.com/nihardamera/Commit-Suggestor) and
[Hugging Face](https://huggingface.co/nihar-damera/commit-suggestor).

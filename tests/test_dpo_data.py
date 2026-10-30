"""Building DPO candidate pairs from sampled subjects."""

from __future__ import annotations

from commit_suggestor.dpo_data import candidate_pairs, diff_of
from commit_suggestor.prompt import build_messages

DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"


def test_the_diff_comes_back_out_of_the_prompt_exactly():
    assert diff_of(build_messages(DIFF)) == DIFF
    assert diff_of(build_messages(DIFF, ["Fix a", "Add b"])) == DIFF


def test_pairs_use_two_distinct_samples_and_count_diffs_with_fewer():
    rows = [
        {
            "sha": "1",
            "prompt": build_messages(DIFF),
            "samples": ["Set x to 2", "set x to 2.", "Bump x", "Change x"],
        },
        {"sha": "2", "prompt": build_messages(DIFF), "samples": ["Same", "same", "Same.", ""]},
    ]
    pairs, too_few = candidate_pairs(rows)
    assert too_few == 1
    (pair,) = pairs
    assert pair["a"].lower().rstrip(".") != pair["b"].lower().rstrip(".")
    assert pair["diff"] == DIFF


def test_the_choice_of_pair_is_reproducible():
    rows = [
        {
            "sha": str(i),
            "prompt": build_messages(DIFF),
            "samples": [f"s{i}a", f"s{i}b", f"s{i}c", f"s{i}d"],
        }
        for i in range(20)
    ]
    first, _ = candidate_pairs(rows)
    again, _ = candidate_pairs(rows)
    assert [(p["a"], p["b"]) for p in first] == [(p["a"], p["b"]) for p in again]

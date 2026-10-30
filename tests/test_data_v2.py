"""The v2 data filters: near-duplicates, context, and the context dropout."""

from __future__ import annotations

from commit_suggestor.data_v2 import (
    MIN_SHINGLES,
    NearDuplicateIndex,
    jaccard,
    keeps_context,
    shingles,
    to_example,
    with_context,
)
from commit_suggestor.prompt import CONTEXT_INTRO

BODY = "\n".join(f"+    value_{i} = compute(value_{i - 1}, factor={i})" for i in range(1, 12))
DIFF = f"diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,11 @@\n{BODY}"


class TestShingles:
    def test_only_changed_lines_count(self):
        header_only = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n context"
        assert shingles(header_only) == set()

    def test_a_diff_that_only_adds_blank_lines_is_too_small_to_compare(self):
        assert len(shingles("diff --git a/x b/x\n+\n+\n-")) < MIN_SHINGLES


class TestNearDuplicates:
    def test_a_lightly_edited_copy_is_found(self):
        index = NearDuplicateIndex()
        index.add(shingles(DIFF))
        edited = DIFF.replace("factor=11", "factor=12")
        assert jaccard(shingles(DIFF), shingles(edited)) >= 0.8
        assert index.find(shingles(edited)) == 0

    def test_an_unrelated_diff_is_not(self):
        index = NearDuplicateIndex()
        index.add(shingles(DIFF))
        other = "\n".join(f"+ docs line {i} about routers and views" for i in range(20))
        assert index.find(shingles(other)) is None


def test_context_is_the_older_commits_newest_first():
    rows = [{"subject": f"Change {i} (#{i})"} for i in (9, 8, 7, 6, 5, 4, 3)]
    with_context(rows)
    assert rows[0]["context"] == ["Change 8", "Change 7", "Change 6", "Change 5", "Change 4"]
    assert rows[-1]["context"] == []


def test_the_context_dropout_is_fixed_per_commit_and_near_half():
    shas = [f"{i:040x}" for i in range(2000)]
    assert [keeps_context(s) for s in shas] == [keeps_context(s) for s in shas]
    assert 0.45 < sum(map(keeps_context, shas)) / len(shas) < 0.55


def test_an_example_shows_context_only_when_kept():
    row = {"diff_clean": DIFF, "subject_clean": "Add values", "context": ["Fix a", "Add b"]}
    with_ctx, without = to_example(row, True), to_example(row, False)
    assert CONTEXT_INTRO in with_ctx["messages"][1]["content"]
    assert CONTEXT_INTRO not in without["messages"][1]["content"]
    assert with_ctx["messages"][-1] == {"role": "assistant", "content": "Add values"}

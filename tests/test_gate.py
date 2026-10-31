"""The CI model gate's pass/fail rules."""

from __future__ import annotations

from commit_suggestor.gate import compare, gate_examples, summarise

BASELINE = {"malformed": 0, "invented_names": 1, "rougeL": 30.0}


def test_matching_the_baseline_passes():
    assert compare({"malformed": 0, "invented_names": 1, "rougeL": 29.0}, BASELINE) == []


def test_a_malformed_subject_fails():
    (reason,) = compare({"malformed": 1, "invented_names": 0, "rougeL": 31.0}, BASELINE)
    assert "72 characters" in reason


def test_more_invented_names_fail():
    (reason,) = compare({"malformed": 0, "invented_names": 2, "rougeL": 31.0}, BASELINE)
    assert "names not in the diff" in reason


def test_a_drop_beyond_the_tolerance_fails():
    assert compare({"malformed": 0, "invented_names": 0, "rougeL": 27.1}, BASELINE) == []
    (reason,) = compare({"malformed": 0, "invented_names": 0, "rougeL": 26.9}, BASELINE)
    assert "ROUGE-L" in reason


def test_the_summary_counts_on_the_fixed_diffs():
    examples = gate_examples()
    assert len(examples) == 25
    summary = summarise([e["subject"] for e in examples], examples)
    assert summary["rougeL"] == 100.0 and summary["malformed"] == 0

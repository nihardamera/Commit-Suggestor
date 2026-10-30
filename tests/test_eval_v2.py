"""The v2 evaluation agrees with the v1 one, and its copy rule does what it says."""

from __future__ import annotations

import json
from pathlib import Path

from commit_suggestor.eval_v2 import copy_rate, predictions, score_system

EVAL = Path(__file__).resolve().parent.parent / "eval"


def test_copying_a_context_subject_counts_ignoring_case_and_punctuation():
    contexts = [["Fix the parser", "Add tests"], ["Fix the parser"]]
    assert copy_rate(["fix the parser.", "Fix the lexer"], contexts) == [1.0, 0.0]


def test_v1_scores_match_the_v1_evaluation():
    v1 = json.loads((EVAL / "results.json").read_text())["scores"]
    for name, column in (("base", "base"), ("v1", "finetuned")):
        summary, _ = score_system(name)
        assert summary["rougeL"]["micro"][0] == v1[column]["rougeL_f1"]
        assert summary["bleu"] == v1[column]["bleu"]


def test_predictions_follow_the_test_set_order():
    tests = [json.loads(line) for line in (EVAL / "test.jsonl").open()]
    rows = [json.loads(line) for line in (EVAL / "predictions.jsonl").open()]
    assert predictions("v1") == [r["finetuned"] for r in rows]
    assert len(predictions("base")) == len(tests)

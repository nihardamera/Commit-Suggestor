"""The v2 evaluation agrees with the v1 one, and its copy rule does what it says."""

from __future__ import annotations

import json
from pathlib import Path

from commit_suggestor.eval_v2 import choose_beta, copy_rate, predictions, score_system

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


def _candidate(win, low, invented=0.0, generic=0.0, words=0.0):
    return {
        "win_rate": [win, low, win + 5],
        "extra_invented_pts": invented,
        "extra_generic_pts": generic,
        "extra_words": words,
    }


def test_the_best_eligible_beta_is_chosen():
    chosen = choose_beta(
        {0.05: _candidate(55, 49), 0.1: _candidate(60, 53), 0.3: _candidate(70, 64, words=2.5)}
    )
    assert chosen == {"beta": 0.1, "ship_dpo": True}


def test_dpo_ships_only_if_it_clearly_beats_sft():
    assert choose_beta({0.1: _candidate(56, 49)}) == {"beta": 0.1, "ship_dpo": False}


def test_no_beta_is_eligible_when_all_exploit():
    chosen = choose_beta(
        {0.1: _candidate(70, 64, invented=1.5), 0.3: _candidate(70, 64, generic=2)}
    )
    assert chosen == {"beta": None, "ship_dpo": False}

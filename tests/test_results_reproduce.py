"""Every number in eval/results.json must follow from the committed predictions.

The README quotes results.json, so this is what keeps a reported number from
drifting away from the predictions it came from. No model is needed: scoring
the 450 cached predictions takes about a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from commit_suggestor.evaluate import paired_bootstrap, score

EVAL = Path(__file__).resolve().parent.parent / "eval"
SYSTEMS = ("heuristic", "base", "finetuned")


@pytest.fixture(scope="module")
def committed() -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in (EVAL / "predictions.jsonl").open()]
    return rows, json.loads((EVAL / "results.json").read_text())


@pytest.fixture(scope="module")
def rescored(committed: tuple[list[dict], dict]) -> dict[str, dict]:
    rows, _ = committed
    refs = [r["reference"] for r in rows]
    return {name: score(refs, [r[name] for r in rows]) for name in SYSTEMS}


def test_the_test_set_is_the_one_reported(committed):
    rows, results = committed
    assert len(rows) == results["test_examples"]
    assert sorted({r["repo"] for r in rows}) == results["test_repos"]


def test_the_committed_test_set_lines_up_with_the_predictions(committed):
    rows, _ = committed
    examples = [json.loads(line) for line in (EVAL / "test.jsonl").open()]
    assert [(e["sha"], e["subject"]) for e in examples] == [
        (r["sha"], r["reference"]) for r in rows
    ]


@pytest.mark.parametrize("name", SYSTEMS)
def test_headline_scores_recompute_exactly(committed, rescored, name):
    _, results = committed
    got = {k: v for k, v in rescored[name].items() if not k.startswith("_")}
    assert got == results["scores"][name]


def test_the_confidence_interval_recomputes_exactly(committed, rescored):
    _, results = committed
    low, high = paired_bootstrap(
        rescored["base"]["_per_example_rougeL"], rescored["finetuned"]["_per_example_rougeL"]
    )
    assert [low, high] == results["rougeL_gain_finetuned_vs_base_95ci"]


def test_per_project_scores_recompute_exactly(committed, rescored):
    rows, results = committed
    for repo, reported in results["rougeL_by_repo"].items():
        idx = [i for i, r in enumerate(rows) if r["repo"] == repo]
        assert len(idx) == reported["n"]
        for name in SYSTEMS:
            per_example = rescored[name]["_per_example_rougeL"]
            assert round(100 * sum(per_example[i] for i in idx) / len(idx), 2) == reported[name]

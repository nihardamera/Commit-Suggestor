"""Unblinding my ratings into fine-tuned-vs-base labels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from commit_suggestor.human import _v2_pair_label, main, pairwise_labels

EVAL = Path(__file__).resolve().parent.parent / "eval"


def test_the_best_pick_decides_and_scores_break_the_rest():
    record = {"best": "1", "scores": {"1": 2, "2": 5, "3": 4}}
    assert _v2_pair_label(record, slot_ft="1", slot_base="2") == "finetuned"
    # The reference (slot 3) was best: fall back to the scores.
    record = {"best": "3", "scores": {"1": 2, "2": 4, "3": 5}}
    assert _v2_pair_label(record, slot_ft="1", slot_base="2") == "base"
    record = {"best": "0", "scores": {"1": 3, "2": 3, "3": 3}}
    assert _v2_pair_label(record, slot_ft="1", slot_base="2") == "tie"


def test_every_rating_becomes_one_label_in_the_preregistered_split():
    pairs = pairwise_labels()
    assert len(pairs) == len({(p.set, p.id) for p in pairs}) == 150
    split = json.loads((EVAL / "human" / "split.json").read_text())
    assert Counter(p.split for p in pairs) == {
        "dev": len(split["ab"]["dev"]) + len(split["v2"]["dev"]),
        "heldout": len(split["ab"]["heldout"]) + len(split["v2"]["heldout"]),
    }


def test_labels_unblind_through_the_key():
    preds = [json.loads(line) for line in (EVAL / "predictions.jsonl").open()]
    for p in pairwise_labels():
        assert (p.finetuned, p.base, p.reference) == (
            preds[p.id]["finetuned"],
            preds[p.id]["base"],
            preds[p.id]["reference"],
        )


def test_the_committed_analysis_recomputes(tmp_path):
    out, md = tmp_path / "analysis.json", tmp_path / "report.md"
    main(["--out", str(out), "--markdown", str(md)])
    assert json.loads(out.read_text()) == json.loads((EVAL / "human" / "analysis.json").read_text())
    assert md.read_text() == (EVAL / "rouge_vs_human.md").read_text()

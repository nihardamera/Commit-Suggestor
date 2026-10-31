"""The demo's input handling, with a stand-in for the model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("gradio")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "space"))
import app

DIFF = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"


def test_a_diff_gets_a_cleaned_subject_built_from_the_shared_prompt():
    seen = []

    def fake(messages):
        seen.append(messages)
        return "`Set x to 2.`"

    subject, note = app.suggest(DIFF, fake)
    assert subject == "Set x to 2"
    assert "```diff" in seen[0][-1]["content"] and "s on a shared CPU" in note


def test_empty_and_oversized_input_never_reach_the_model():
    def fail(_):
        raise AssertionError("model called")

    assert app.suggest("   ", fail)[0] == ""
    subject, note = app.suggest("x" * (app.MAX_INPUT_CHARS + 1), fail)
    assert subject == "" and "characters" in note


def test_a_long_subject_is_flagged():
    _, note = app.suggest(DIFF, lambda _: "Change " + "x" * 80)
    assert "72 characters" in note

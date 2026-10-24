"""Reference-free checks of a subject against its diff."""

from __future__ import annotations

import json
from pathlib import Path

from commit_suggestor.metrics import (
    changed_names,
    check,
    identifiers,
    is_generic,
    is_imperative,
    main,
    per_example,
    summarise,
)

DIFF = """diff --git a/src/client.py b/src/client.py
--- a/src/client.py
+++ b/src/client.py
@@ -10,6 +10,9 @@ class Client:
     def connect(self):
-        sock = open_socket(self.host)
+        sock = open_socket(self.host, timeout=DEFAULT_TIMEOUT)
+        for attempt in range(self.max_retries):
+            pass"""

DOCS = """diff --git a/docs/api-guide/routers.md b/docs/api-guide/routers.md
--- a/docs/api-guide/routers.md
+++ b/docs/api-guide/routers.md
@@ -1,2 +1,3 @@
 # Routers
+See also django-rest-multiple-models for nested routes."""


class TestIdentifiers:
    def test_code_like_names_are_found(self):
        found = identifiers("Use `max_retries` in Client.connect and openSocket()")
        assert found == ["max_retries", "Client.connect", "openSocket"]

    def test_plain_words_abbreviations_and_product_names_are_not(self):
        assert identifiers("Fix the build on GitHub, e.g. for macOS and M2M") == []

    def test_a_call_counts_only_with_parentheses(self):
        assert identifiers("Retry connect() on failure") == ["connect"]


class TestHallucination:
    def test_names_in_the_diff_are_grounded(self):
        assert check("Pass DEFAULT_TIMEOUT to open_socket", DIFF).hallucinated == ()

    def test_names_missing_from_the_diff_are_hallucinated(self):
        assert check("Add retry_backoff to Client.reconnect", DIFF).hallucinated == (
            "retry_backoff",
            "Client.reconnect",
        )

    def test_a_dotted_name_is_grounded_when_each_part_is_present(self):
        assert check("Add a timeout to Client.connect", DIFF).hallucinated == ()

    def test_file_names_match_ignoring_case(self):
        assert check("Update Client.py", DIFF).hallucinated == ()


class TestSpecificity:
    def test_changed_files_and_definitions_are_collected(self):
        names = changed_names(DIFF)
        assert {"client.py", "client", "connect", "open_socket", "default_timeout"} <= names

    def test_naming_what_changed_is_specific(self):
        assert check("Add retries to connect", DIFF).specific
        assert check("Link django-rest-multiple-models from the routers guide", DOCS).specific

    def test_a_subject_that_fits_any_diff_is_not(self):
        assert not check("Fix bug", DIFF).specific
        assert not check("Improve performance", DIFF).specific


class TestGenericAndForm:
    def test_template_subjects_are_generic(self):
        for subject in ("Fix bug", "Update docs", "minor fixes", "Update client.py", "Clean up"):
            assert is_generic(subject), subject

    def test_descriptive_subjects_are_not(self):
        assert not is_generic("Fix crash when the config file is empty")

    def test_imperative_mood(self):
        for subject in ("Add retries", "Embed fonts", "Process hooks", "fix: handle None"):
            assert is_imperative(subject), subject
        for subject in ("Added retries", "Adds retries", "Fixed the crash"):
            assert not is_imperative(subject), subject

    def test_one_subject_repeated_across_diffs_is_generic(self):
        values = per_example(["Tidy the parser"] * 3 + ["Add retries"], [DIFF] * 4)
        assert values["generic_pct"] == [1.0, 1.0, 1.0, 0.0]


def test_summary_reports_intervals_and_quantiles():
    summary = summarise(["Add retries to connect", "Fix bug"], [DIFF, DIFF], ["r", "r"])
    assert summary["specific_pct"]["micro"][0] == 50.0
    assert summary["words_quantiles"]["median"] == 3.0
    assert summary["opening_word_top10"][0] == ("add", 1)


def test_the_committed_v1_metrics_recompute(tmp_path):
    eval_dir = Path(__file__).resolve().parent.parent / "eval"
    out = tmp_path / "metrics.json"
    main(
        [
            "--out",
            str(out),
            "--test",
            str(eval_dir / "test.jsonl"),
            "--predictions",
            str(eval_dir / "predictions.jsonl"),
        ]
    )
    assert json.loads(out.read_text()) == json.loads((eval_dir / "metrics_v1.json").read_text())

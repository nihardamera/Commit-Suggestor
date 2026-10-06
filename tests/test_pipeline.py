"""The data filters and prompt handling, which decide what the model learns.

None of these need the model, so they run in a second.
"""
from __future__ import annotations

from commit_suggestor.evaluate import heuristic, normalise, paired_bootstrap, well_formed
from commit_suggestor.prepare import clean_diff, clean_subject, keep
from commit_suggestor.prompt import (
    MAX_DIFF_CHARS,
    SYSTEM,
    build_messages,
    clean_reply,
    truncate_diff,
)

SMALL_DIFF = """diff --git a/src/app.py b/src/app.py
index 1a2b3c4..5d6e7f8 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,3 @@
-TIMEOUT = 5
+TIMEOUT = 30
 def run():
     pass"""

LOCKFILE = """diff --git a/poetry.lock b/poetry.lock
index 1111111..2222222 100644
--- a/poetry.lock
+++ b/poetry.lock
@@ -1 +1 @@
-version = 1
+version = 2"""


def row(subject: str = "Raise the default timeout to 30 seconds", diff: str = SMALL_DIFF,
        author: str = "Jane Dev", email: str = "jane@example.com") -> dict:
    return {"subject": subject, "diff": diff, "author": author, "email": email}


class TestSubjects:
    def test_pull_request_numbers_are_stripped(self):
        """They can't be inferred from a diff, so a model trained on them
        learns to invent them."""
        assert clean_subject("Fix redirect handling (#6512)") == "Fix redirect handling"
        assert clean_subject("Fix redirect handling (gh-6512)") == "Fix redirect handling"

    def test_ci_tags_and_trailing_period_go(self):
        assert clean_subject("[skip ci] Update docs.") == "Update docs"
        assert clean_subject("[svn r63149] always have a default chdir") == "always have a default chdir"

    def test_merges_releases_and_bumps_are_rejected(self):
        for subject in ("Merge branch 'main' into dev", "Release 2.31.0",
                        "Bump urllib3 from 1.26 to 2.0", "Revert \"Add cache\"",
                        "fixup! tidy imports", "2.31.0"):
            ok, why = keep(row(subject=subject))
            assert not ok, subject

    def test_bots_are_rejected(self):
        ok, why = keep(row(author="dependabot[bot]"))
        assert (ok, why) == (False, "bot author")
        ok, why = keep(row(author="pre-commit-ci[bot]", email="66853113+pre-commit-ci[bot]@users"))
        assert not ok

    def test_too_short_or_too_long_subjects_are_rejected(self):
        assert keep(row(subject="fix"))[1] == "subject length"
        assert keep(row(subject="x" * 80))[1] == "subject length"


class TestDiffs:
    def test_index_lines_are_removed(self):
        cleaned, files = clean_diff(SMALL_DIFF)
        assert files == 1
        assert "index 1a2b3c4" not in cleaned
        assert "+TIMEOUT = 30" in cleaned

    def test_lock_files_are_dropped_from_the_diff(self):
        cleaned, files = clean_diff(SMALL_DIFF + "\n" + LOCKFILE)
        assert files == 1
        assert "poetry.lock" not in cleaned

    def test_a_lockfile_only_change_is_not_an_example(self):
        assert keep(row(diff=LOCKFILE)) == (False, "no text diff")

    def test_sprawling_commits_are_rejected(self):
        many = "\n".join(SMALL_DIFF.replace("src/app.py", f"src/m{i}.py") for i in range(5))
        assert keep(row(diff=many)) == (False, "too many files")

    def test_a_good_commit_is_kept_with_its_cleaned_fields(self):
        r = row(subject="Raise the default timeout (#42)")
        assert keep(r) == (True, "kept")
        assert r["subject_clean"] == "Raise the default timeout"
        assert "index " not in r["diff_clean"]


class TestPrompt:
    def test_long_diffs_are_cut_at_a_line_and_marked(self):
        long = "\n".join(f"+line {i}" for i in range(2000))
        cut = truncate_diff(long)
        assert len(cut) <= MAX_DIFF_CHARS + 30
        assert cut.endswith("(diff truncated)")
        assert "\n+line" in cut and not cut.split("\n")[-2].endswith("+lin")

    def test_short_diffs_are_untouched(self):
        assert truncate_diff(SMALL_DIFF) == SMALL_DIFF

    def test_messages_carry_the_system_prompt_and_the_diff(self):
        messages = build_messages(SMALL_DIFF)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == SYSTEM
        assert "+TIMEOUT = 30" in messages[1]["content"]

    def test_replies_are_reduced_to_one_clean_line(self):
        assert clean_reply("```\nAdd retry to fetch.\n```") == "Add retry to fetch"
        assert clean_reply('"Fix typo in README"\n\nThis fixes...') == "Fix typo in README"
        assert clean_reply("   \n") == ""

    def test_inline_code_at_the_end_of_a_subject_is_kept_whole(self):
        assert clean_reply("Add note about :dudir:`include`") == "Add note about :dudir:`include`"
        assert clean_reply("Deprecate `unique_items` in `conlist`") == "Deprecate `unique_items` in `conlist`"

    def test_a_reply_wrapped_in_backticks_is_unwrapped(self):
        assert clean_reply("`Bump the timeout`") == "Bump the timeout"


class TestEvaluationHelpers:
    def test_the_heuristic_baseline_names_the_first_file(self):
        assert heuristic(SMALL_DIFF) == "Update app.py"

    def test_normalisation_ignores_case_and_punctuation(self):
        assert normalise("Fix: the Bug!") == normalise("fix the bug")

    def test_well_formed_means_one_short_line(self):
        assert well_formed("Add retry to fetch")
        assert not well_formed("")
        assert not well_formed("x" * 73)

    def test_bootstrap_interval_brackets_a_clear_improvement(self):
        low, high = paired_bootstrap([0.2] * 50, [0.5] * 50)
        assert low == high == 30.0

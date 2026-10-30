"""Repo-style context, and that adding it left the v1 prompt untouched."""

from __future__ import annotations

import os
import subprocess

import pytest

from commit_suggestor.context import clean_context, recent_subjects
from commit_suggestor.prompt import CONTEXT_INTRO, INSTRUCTION, SYSTEM, build_messages

DIFF = "diff --git a/a.py b/a.py\n+x = 1"


def test_without_context_the_prompt_is_exactly_v1():
    v1_user = f"{INSTRUCTION}\n\n```diff\n{DIFF}\n```"
    expected = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": v1_user}]
    assert (
        build_messages(DIFF) == build_messages(DIFF, None) == build_messages(DIFF, []) == expected
    )


def test_context_comes_before_the_diff_newest_first():
    user = build_messages(DIFF, ["Add b", "Add a"])[1]["content"]
    assert user.index(CONTEXT_INTRO) < user.index("- Add b") < user.index("- Add a")
    assert user.index("- Add a") < user.index("```diff")


def test_context_is_cleaned_like_training_targets():
    assert clean_context(["Fix bug (#12)", "", "[skip ci] Tidy.", "a", "b", "c", "d"]) == [
        "Fix bug",
        "Tidy",
        "a",
        "b",
        "c",
    ]


@pytest.fixture
def repo(tmp_path):
    clock = iter(range(1, 100))

    def git(*args: str) -> str:
        # One second apart: git log orders by date, and commits made within the
        # same second would come out in an arbitrary order.
        stamp = f"2026-01-01T00:00:{next(clock):02d}Z"
        env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    for i in range(1, 8):
        (tmp_path / "f.txt").write_text(str(i))
        git("add", "f.txt")
        git("commit", "-q", "-m", f"Change {i} (#{i})")
    git("checkout", "-q", "-b", "side")
    (tmp_path / "g.txt").write_text("side")
    git("add", "g.txt")
    git("commit", "-q", "-m", "Side change")
    git("checkout", "-q", "main")
    git("merge", "-q", "--no-ff", "side", "-m", "Merge branch side")
    return tmp_path, git


def test_for_a_new_commit_the_latest_commits_count_and_merges_do_not(repo):
    path, _ = repo
    assert recent_subjects(path) == ["Side change", "Change 7", "Change 6", "Change 5", "Change 4"]


def test_for_an_existing_commit_only_earlier_ones_count(repo):
    path, git = repo
    sha = git("log", "--format=%H", "--grep", "Change 7")
    assert recent_subjects(path, sha, include_before=False) == [
        "Change 6",
        "Change 5",
        "Change 4",
        "Change 3",
        "Change 2",
    ]


def test_no_history_gives_no_context(tmp_path):
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    assert recent_subjects(tmp_path) == []

"""The git hooks and the feedback record, with a stand-in for the model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from commit_suggestor import cli


def hook_with(tmp_path, fake_body: str) -> tuple:
    """The prepare-commit-msg hook, calling a fake 'python' instead of the model."""
    fake = tmp_path / "fake-python"
    fake.write_text(f"#!/bin/sh\n{fake_body}\n")
    fake.chmod(0o755)
    hook = tmp_path / "prepare-commit-msg"
    hook.write_text(cli.PREPARE_HOOK.replace("__PYTHON__", str(fake)))
    hook.chmod(0o755)
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text("\n# Please enter the commit message for your changes.\n")
    return hook, message


def run_hook(hook, message, *extra: str, timeout: str = "8", **env: str):
    return subprocess.run(
        ["sh", str(hook), str(message), *extra],
        capture_output=True,
        text=True,
        env={**os.environ, "COMMIT_SUGGESTOR_TIMEOUT": timeout, **env},
    )


class TestPrepareHook:
    def test_a_suggestion_is_put_above_the_template(self, tmp_path):
        hook, message = hook_with(tmp_path, 'echo "Add retries"; echo "# Suggested by x"')
        assert run_hook(hook, message).returncode == 0
        assert message.read_text().startswith("Add retries\n# Suggested by x\n\n# Please")

    def test_a_slow_model_is_killed_and_the_commit_goes_on(self, tmp_path):
        hook, message = hook_with(tmp_path, 'sleep 30; echo "too late"')
        before = message.read_text()
        started = time.time()
        assert run_hook(hook, message, timeout="1").returncode == 0
        assert time.time() - started < 5
        assert message.read_text() == before

    def test_a_missing_model_prints_a_hint_and_changes_nothing(self, tmp_path):
        hook, message = hook_with(tmp_path, "exit 3")
        before = message.read_text()
        result = run_hook(hook, message)
        assert result.returncode == 0 and "commit-suggestor download" in result.stderr
        assert message.read_text() == before

    @pytest.mark.parametrize("source", ["message", "merge", "commit"])
    def test_only_a_plain_git_commit_is_touched(self, tmp_path, source):
        hook, message = hook_with(tmp_path, 'echo "Add retries"')
        before = message.read_text()
        run_hook(hook, message, source)
        assert message.read_text() == before

    def test_it_can_be_switched_off(self, tmp_path):
        hook, message = hook_with(tmp_path, 'echo "Add retries"')
        before = message.read_text()
        run_hook(hook, message, COMMIT_SUGGESTOR_DISABLE="1")
        assert message.read_text() == before


def test_outcomes():
    assert cli.outcome("Add retries to connect", "add retries to connect.")[0] == "accepted"
    assert cli.outcome("Add retries to connect", "Add retries to the connect call")[0] == "edited"
    assert cli.outcome("Add retries to connect", "Bump version")[0] == "replaced"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    (tmp_path / "a.txt").write_text("1\n")
    git("add", "a.txt")
    git("commit", "-q", "-m", "First")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "FEEDBACK", tmp_path / "feedback.jsonl")
    return tmp_path, git


def test_feedback_compares_the_suggestion_with_the_commit(repo):
    path, git = repo
    (path / "a.txt").write_text("2\n")
    git("add", "a.txt")
    diff = git("diff", "--cached", "--no-color", "-M")
    cli.remember(diff, "Change a to 2", "test-model", 0.5)
    git("commit", "-q", "-m", "Change a.txt to 2")
    cli.cmd_record_feedback(None)
    (row,) = [json.loads(line) for line in (path / "feedback.jsonl").open()]
    assert (row["outcome"], row["committed"], row["model"]) == (
        "edited",
        "Change a.txt to 2",
        "test-model",
    )
    assert not cli.last_path().exists()


def test_feedback_ignores_a_suggestion_for_a_different_diff(repo):
    path, git = repo
    cli.remember("some other diff", "Unrelated", "m", 0.1)
    (path / "a.txt").write_text("3\n")
    git("commit", "-q", "-am", "Change a")
    cli.cmd_record_feedback(None)
    assert not (path / "feedback.jsonl").exists()


def test_the_cli_reads_a_diff_from_stdin_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "MODEL_HOME", tmp_path / "none")
    monkeypatch.setattr(cli, "V1_ADAPTER", tmp_path / "none")
    monkeypatch.delenv("COMMIT_SUGGESTOR_MODEL", raising=False)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("diff --git a/x b/x\n+y"))
    with pytest.raises(SystemExit, match="no model"):
        cli.main(["-"])

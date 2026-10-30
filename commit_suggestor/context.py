"""Recent commit subjects of a repository, shown to the v2 model as style examples.

Projects write subjects differently ("Fix x" vs "fix: x" vs "x: fix"), and
nothing in a diff says which. v2 shows the model the subjects of the five
commits before the one it describes, so it can follow the house style.

The same rules apply everywhere the context is built, so that training,
evaluation and the CLI agree:
  - only commits *before* the one being described (no peeking at the answer),
  - merges excluded (`git log --no-merges`), newest first,
  - each subject cleaned like a training target (PR numbers and CI tags removed).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from commit_suggestor.prepare import clean_subject

N_RECENT = 5


def clean_context(subjects: Sequence[str], n: int = N_RECENT) -> list[str]:
    """The first `n` non-empty subjects, cleaned. Input is newest first."""
    cleaned = (clean_subject(s) for s in subjects)
    return [s for s in cleaned if s][:n]


def recent_subjects(
    repo: Path, before: str = "HEAD", n: int = N_RECENT, include_before: bool = True
) -> list[str]:
    """Subjects of the `n` non-merge commits up to `before`, newest first.

    For a commit being written (the CLI) `before` is HEAD and it is included:
    those are the commits before the new one. For an existing commit, pass its
    SHA with `include_before=False`, so the commit itself is left out.
    """
    rev = before if include_before else f"{before}^"
    result = subprocess.run(
        ["git", "-C", str(repo), "log", "--no-merges", f"-{3 * n}", "--format=%s", rev],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:  # a new repository, or a root commit
        return []
    return clean_context(result.stdout.splitlines(), n)

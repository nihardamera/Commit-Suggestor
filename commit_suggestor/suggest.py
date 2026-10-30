"""Kept so `python -m commit_suggestor.suggest` still works; see commit_suggestor.cli."""

from __future__ import annotations

import sys

from commit_suggestor.cli import main

if __name__ == "__main__":
    main(["suggest", *sys.argv[1:]])

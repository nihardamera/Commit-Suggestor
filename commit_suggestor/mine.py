"""Mine (diff, commit subject) pairs from public Git repositories.

Works on local clones rather than the GitHub API. The first version of this
project asked the API for one comparison per commit, which is slow, rate
limited, and only ever looked at the first changed file. A clone gives every
commit's full patch in a single `git log -p`.

Clones are shallow (`--depth`), which bounds the download for large projects.
The oldest commit in a shallow clone has no parent, so its "diff" is the whole
tree; the size filter in `prepare.py` drops it.

Usage:
    python -m commit_suggestor.mine --out data/raw --depth 4000
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

#: Projects with a consistent habit of short, descriptive commit subjects.
#: The split is by repository, so the test set comes from projects the model
#: never saw during training, which is the honest way to measure it.
REPOS: dict[str, list[str]] = {
    "train": [
        "psf/requests", "pallets/flask", "pallets/click", "pallets/jinja",
        "pallets/werkzeug", "encode/httpx", "encode/starlette",
        "Textualize/rich", "pytest-dev/pytest", "psf/black", "pypa/pip",
        "pre-commit/pre-commit", "python-poetry/poetry",
        "marshmallow-code/marshmallow", "aio-libs/aiohttp",
    ],
    "valid": ["python-attrs/attrs", "fastapi/typer"],
    "test": ["sphinx-doc/sphinx", "encode/django-rest-framework",
             "pydantic/pydantic"],
}

# Field and record separators that never appear in commit text or diffs.
_FIELD, _RECORD, _END = "\x1f", "\x1e", "\x1d"
_FORMAT = f"{_RECORD}%H{_FIELD}%an{_FIELD}%ae{_FIELD}%s{_END}"


def clone(slug: str, cache: Path, depth: int, attempts: int = 3) -> Path | None:
    """Shallow-clone a repository, retrying transient network failures.

    Returns None if it still fails, so one unreachable project does not
    abandon the rest of the run.
    """
    target = cache / slug.replace("/", "__")
    for attempt in range(1, attempts + 1):
        if target.exists() and (target / ".git").exists():
            return target
        shutil.rmtree(target, ignore_errors=True)
        result = subprocess.run(
            ["git", "clone", "--quiet", "--single-branch", "--no-tags",
             f"--depth={depth}", f"https://github.com/{slug}.git", str(target)])
        if result.returncode == 0:
            return target
        print(f"  clone of {slug} failed (attempt {attempt}/{attempts})", flush=True)
    shutil.rmtree(target, ignore_errors=True)
    return None


def commits(repo_dir: Path) -> list[dict[str, str]]:
    """Every non-merge commit with its patch, newest first."""
    out = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "--no-merges", "-p", "--no-color",
         "--unified=3", "-M", f"--format={_FORMAT}"],
        check=True, capture_output=True)
    text = out.stdout.decode("utf-8", errors="replace")
    records = []
    for chunk in text.split(_RECORD)[1:]:
        header, _, patch = chunk.partition(_END)
        sha, author, email, subject = (header.split(_FIELD) + ["", "", "", ""])[:4]
        records.append({"sha": sha, "author": author, "email": email,
                        "subject": subject, "diff": patch.strip("\n")})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache", type=Path, default=Path("data/clones"))
    parser.add_argument("--depth", type=int, default=4000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    for split, slugs in REPOS.items():
        for slug in slugs:
            path = args.out / f"{slug.replace('/', '__')}.jsonl"
            if path.exists():
                print(f"{split:5}  {slug:32}  already mined", flush=True)
                continue
            repo_dir = clone(slug, args.cache, args.depth)
            if repo_dir is None:
                print(f"{split:5}  {slug:32}  SKIPPED: could not clone", flush=True)
                continue
            rows = commits(repo_dir)
            with path.open("w") as fh:
                for row in rows:
                    fh.write(json.dumps({"repo": slug, "split": split, **row}) + "\n")
            print(f"{split:5}  {slug:32}  {len(rows):6} commits", flush=True)


if __name__ == "__main__":
    main()

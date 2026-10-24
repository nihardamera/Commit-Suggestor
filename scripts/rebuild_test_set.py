"""Rebuild the 450 test examples (diff and subject) from the commits they came from.

`eval/predictions.jsonl` records each test example's repository and commit,
but not the diff, which lived in the ignored `data/` folder. This fetches the
same commits from the three test projects, cleans them with the same code as
training, checks the result against the 50 diffs already committed in
`eval/ab_sample.jsonl`, and writes `eval/test.jsonl`.

The three projects are BSD or MIT licensed; the file keeps each commit's
repository and SHA so every example can be traced to its source.

    uv run python scripts/rebuild_test_set.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from commit_suggestor.mine import commits
from commit_suggestor.prepare import clean_diff, clean_subject

ROOT = Path(__file__).resolve().parent.parent
EVAL, CLONES = ROOT / "eval", ROOT / "data" / "clones-full"


def clone(slug: str) -> Path:
    """Full history without file contents; git fetches the blobs a patch needs.

    `.gitattributes` is checked out because it picks the diff driver (sphinx
    sets `*.py diff=python`), which decides the function named in each `@@`
    hunk header. The original clones were checked out, so it applied there.
    """
    target = CLONES / slug.replace("/", "__")
    if not target.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--no-checkout",
                f"https://github.com/{slug}.git",
                str(target),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "HEAD", "--", ".gitattributes"], capture_output=True
    )  # fails harmlessly where there is none
    return target


def main() -> None:
    predictions = [json.loads(line) for line in (EVAL / "predictions.jsonl").open()]
    rebuilt: dict[str, dict[str, str]] = {}
    for slug in sorted({p["repo"] for p in predictions}):
        shas = [p["sha"] for p in predictions if p["repo"] == slug]
        for row in commits(clone(slug), shas):
            diff, _ = clean_diff(row["diff"])
            rebuilt[row["sha"]] = {
                "repo": slug,
                "sha": row["sha"],
                "subject": clean_subject(row["subject"]),
                "diff": diff,
            }
        print(f"{slug:32} {len(shas)} commits")

    missing = [p["sha"] for p in predictions if p["sha"] not in rebuilt]
    assert not missing, f"{len(missing)} commits not found, e.g. {missing[:3]}"
    wrong_subject = [
        p["sha"] for p in predictions if rebuilt[p["sha"]]["subject"] != p["reference"]
    ]
    assert not wrong_subject, f"subjects differ for {wrong_subject[:3]}"

    # The A/B sample stores the diff as it was at training time, so it is an
    # independent check that the rebuild matches.
    ab = [json.loads(line) for line in (EVAL / "ab_sample.jsonl").open()]
    mismatched = [s["id"] for s in ab if rebuilt[predictions[s["id"]]["sha"]]["diff"] != s["diff"]]
    assert not mismatched, f"diffs differ from eval/ab_sample.jsonl for ids {mismatched}"
    print(f"all {len(ab)} diffs in eval/ab_sample.jsonl match exactly")

    with (EVAL / "test.jsonl").open("w") as fh:
        for p in predictions:  # same order as predictions.jsonl
            fh.write(json.dumps(rebuilt[p["sha"]]) + "\n")
    print(f"wrote {len(predictions)} examples to eval/test.jsonl")


if __name__ == "__main__":
    main()

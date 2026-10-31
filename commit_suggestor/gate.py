"""CI model gate: the shipped GGUF model on 25 fixed test diffs, against a baseline.

Fails (exit 1) if, compared with eval/gate_baseline.json:
  - any subject is not one line of at most 72 characters,
  - more subjects name an identifier that is not in the diff,
  - mean ROUGE-L falls more than TOLERANCE points.

Subjects pass through the CLI's length guard (prompt.fit_subject) first, as
they do when shipped. The baseline holds the same model's subjects for the same
diffs, from the GGUF check on Kaggle (notebooks/gguf.py), so a change to the prompt, the diff
cleaning, the runtime or the model that makes things worse fails the build.

    uv run python -m commit_suggestor.gate --model commit-suggestor-q4_k_m.gguf
    uv run python -m commit_suggestor.gate --write-baseline --from predictions_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from commit_suggestor.metrics import check
from commit_suggestor.prompt import build_messages, clean_reply, fit_subject

EVAL = Path(__file__).resolve().parent.parent / "eval"
BASELINE = EVAL / "gate_baseline.json"
N_GATE = 25
TOLERANCE = 3.0


def gate_examples() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (EVAL / "test.jsonl").open()][:N_GATE]


def summarise(subjects: Sequence[str], examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    checks = [check(s, e["diff"]) for s, e in zip(subjects, examples, strict=True)]
    rouge = [
        scorer.score(e["subject"], s)["rougeL"].fmeasure
        for s, e in zip(subjects, examples, strict=True)
    ]
    return {
        "subjects": list(subjects),
        "malformed": sum(not c.well_formed for c in checks),
        "invented_names": sum(bool(c.hallucinated) for c in checks),
        "rougeL": round(100 * sum(rouge) / len(rouge), 2),
    }


def compare(
    now: dict[str, Any], baseline: dict[str, Any], tolerance: float = TOLERANCE
) -> list[str]:
    """The reasons to fail, empty when the gate passes."""
    failures = []
    if now["malformed"]:
        failures.append(f"{now['malformed']} subject(s) not one line of at most 72 characters")
    if now["invented_names"] > baseline["invented_names"]:
        failures.append(
            f"names not in the diff: {now['invented_names']} "
            f"(baseline {baseline['invented_names']})"
        )
    if now["rougeL"] < baseline["rougeL"] - tolerance:
        failures.append(
            f"ROUGE-L {now['rougeL']} vs baseline {baseline['rougeL']} (tolerance {tolerance})"
        )
    return failures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", type=Path, help="a .gguf file (or an MLX folder)")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        help="with --write-baseline: the GGUF job's predictions_test.jsonl",
    )
    args = parser.parse_args(argv)
    examples = gate_examples()

    if args.write_baseline:
        if args.source is None:
            sys.exit("--write-baseline needs --from predictions_test.jsonl")
        by_sha = {
            json.loads(line)["sha"]: json.loads(line)["prediction"] for line in args.source.open()
        }
        missing = [e["sha"] for e in examples if e["sha"] not in by_sha]
        if missing:
            sys.exit(
                f"{args.source} lacks {len(missing)} of the {N_GATE} gate diffs; "
                "use the full GGUF run's predictions"
            )
        baseline = summarise([fit_subject(by_sha[e["sha"]]) for e in examples], examples)
        BASELINE.write_text(json.dumps(baseline, indent=2) + "\n")
        print(json.dumps({k: v for k, v in baseline.items() if k != "subjects"}))
        return

    if args.model is None:
        sys.exit("pass --model")
    from commit_suggestor.backends import load_backend

    backend = load_backend(args.model)
    subjects = [
        fit_subject(clean_reply(backend.generate(build_messages(e["diff"])))) for e in examples
    ]
    now = summarise(subjects, examples)
    baseline = json.loads(BASELINE.read_text())
    changed = sum(a != b for a, b in zip(now["subjects"], baseline["subjects"], strict=True))
    print(
        f"{N_GATE} diffs: malformed {now['malformed']}, invented names "
        f"{now['invented_names']} (baseline {baseline['invented_names']}), ROUGE-L "
        f"{now['rougeL']} (baseline {baseline['rougeL']}); {changed} subjects changed"
    )
    failures = compare(now, baseline)
    if failures:
        sys.exit("model gate failed: " + "; ".join(failures))
    print("model gate passed")


if __name__ == "__main__":
    main()

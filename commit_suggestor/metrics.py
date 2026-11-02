"""Reference-free checks of a commit subject against its diff.

ROUGE-L and BLEU only compare wording with the original author's subject.
These checks look at the subject and the diff instead:

  hallucinated  names a code identifier (snake_case, camelCase, dotted name,
                file name, `backticked`, call()) that appears nowhere in the diff
  specific      not generic, and names something the diff changed: a changed file,
                a function or class from a hunk header or changed line, or a
                word from a changed line
  generic       could describe almost any diff ("Fix bug", "Update docs",
                "Update foo.py"), or the system gave the same subject to at
                least three different diffs
  imperative    starts like an instruction ("Add"), not "Added" or "Adds"
  well_formed   one line of at most 72 characters

They are deliberately simple pattern rules, so they can be read and argued
with. They are proxies: M1 checks them against my own "inaccurate" and
"vague" flags.

    uv run python -m commit_suggestor.metrics    # writes eval/metrics_v1.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from commit_suggestor.evaluate import normalise, well_formed
from commit_suggestor.stats import stratified_bootstrap

#: Product and format names that look like camelCase but are not identifiers.
KNOWN_NAMES = frozenset(
    """
GitHub GitLab PyPI PyPy CPython IPython macOS iOS JavaScript TypeScript LaTeX OpenAPI
GraphQL PostgreSQL MySQL SQLite OAuth ReadTheDocs NumPy SciPy WebSocket WebSockets
DjangoREST TravisCI CircleCI AppVeyor MkDocs XeTeX LuaTeX pdfLaTeX Node.js OpenSearch
""".split()
)

BACKTICKED = re.compile(r"`([^`]+)`")
IDENTIFIER = re.compile(
    r"(?<![\w.])(?:"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+"  # dotted: os.path, conf.py, Model.save
    r"|\w*_\w+"  # snake_case, _private, UPPER_CASE, __init__
    r"|[a-z]+[A-Z]\w*"  # camelCase
    r"|[A-Z][a-z]+[A-Z]\w*"  # PascalCase with an inner capital (not acronyms like M2M)
    r"|[A-Za-z_]\w*(?=\(\))"  # call()
    r")(?![\w])"
)
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEFINED = re.compile(r"\b(?:def|class|async def)\s+([A-Za-z_]\w*)")
CALLED = re.compile(r"([A-Za-z_]\w*)\s*\(")
FILE_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
FILE_NAME = re.compile(r"^[\w.-]+\.[A-Za-z0-9]{1,5}$")

#: Words too common in commit subjects to count as naming the change.
STOPWORDS = frozenset(
    """
a an the to of in on for and or with from by at as is are be not no when if into
add added adds fix fixed fixes update updated updates remove removed removes use uses
change changed changes improve make allow support handle move rename refactor clean
cleanup test tests testing doc docs documentation file files code bug bugs typo typos
error errors issue new old some more less all any it its this that minor small
""".split()
)

GENERIC = re.compile(
    r"^(?:update|fix|add|remove|change|improve|refactor|clean ?up|tidy|tweak|minor)"
    r"(?: (?:the|a|some|minor|small|more))?"
    r"(?: (?:code|files?|docs?|documentation|tests?|bugs?|typos?|issues?|it|stuff|things|"
    r"changes|readme|changelog|imports?|formatting|style|lint(?:ing)?|comments?|logic|"
    r"config(?:uration)?|dependencies|requirements))?"
    r"(?: changes| fixes)?$",
    re.I,
)
UPDATE_ONE_FILE = re.compile(r"^update [\w./-]+$", re.I)
CONVENTIONAL_PREFIX = re.compile(r"^\w+(?:\([^)]*\))?!?:\s*")

#: Verbs that end in "ed" or "s" in their base form.
NOT_PAST = frozenset({"embed", "speed", "feed", "seed", "shed", "proceed", "exceed", "succeed"})
NOT_THIRD_PERSON = frozenset(
    {
        "pass",
        "process",
        "address",
        "access",
        "focus",
        "discuss",
        "bypass",
        "is",
        "has",
        "does",
        "this",
        "always",
    }
)


def identifiers(subject: str) -> list[str]:
    """Code-like names in a subject, in order, without duplicates."""
    found = [m.group(0) for m in IDENTIFIER.finditer(subject)]
    for inner in BACKTICKED.findall(subject):
        found += WORD.findall(inner)
    out: list[str] = []
    for name in found:
        if name in KNOWN_NAMES or name in out:
            continue
        if "." in name and any(len(part) < 2 for part in name.split(".")):
            continue  # "e.g", "i.e"
        out.append(name)
    return out


def in_diff(name: str, diff: str) -> bool:
    """A name is grounded if it, or each part of a dotted name, occurs in the diff.

    File names match ignoring case ("Sphinxdoc.css" for sphinxdoc.css).
    """
    if name in diff or (FILE_NAME.match(name) and name.lower() in diff.lower()):
        return True
    parts = [p for p in name.split(".") if p]
    return len(parts) > 1 and all(re.search(rf"\b{re.escape(p)}\b", diff) for p in parts)


def changed_names(diff: str) -> set[str]:
    """Lower-cased names of what the diff touched.

    Changed files (name, stem and stem parts), names defined or called in hunk
    headers and changed lines, definitions on context lines (the enclosing
    function), identifiers on changed lines and their snake_case parts, and the
    words of changed lines. Words count because documentation changes have no
    identifiers, yet "Add link to django-rest-multiple-models" is specific.
    """
    names: set[str] = set()
    for old, new in FILE_HEADER.findall(diff):
        for path in (old, new):
            base = path.rsplit("/", 1)[-1]
            stem = base.split(".", 1)[0]
            names |= {base.lower(), stem.lower()}
            names |= {p.lower() for p in re.split(r"[_\-.]", stem) if len(p) >= 3}
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("@@"):
            context = line.split("@@", 2)[-1]
            names |= {m.lower() for m in DEFINED.findall(context) + CALLED.findall(context)}
        elif line.startswith(("+", "-")):
            names |= {m.lower() for m in DEFINED.findall(line) + CALLED.findall(line)}
            for m in IDENTIFIER.finditer(line):
                names.add(m.group(0).lower())
                names |= {p.lower() for p in re.split(r"[_.]", m.group(0)) if len(p) >= 4}
            names |= {w.lower() for w in WORD.findall(line[1:]) if len(w) >= 4}
        else:  # context: the function or class the change sits in
            names |= {m.lower() for m in DEFINED.findall(line)}
    return {n for n in names if len(n) >= 3 and n not in STOPWORDS}


def is_imperative(subject: str) -> bool:
    words = CONVENTIONAL_PREFIX.sub("", subject).split()
    if not words:
        return False
    first = words[0].lower()
    if first.endswith("ed") and first not in NOT_PAST:
        return False
    return not (
        first.endswith("s")
        and not first.endswith("ss")
        and first not in NOT_THIRD_PERSON
        and len(first) > 3
    )


def is_generic(subject: str) -> bool:
    text = subject.strip().rstrip(".")
    return bool(GENERIC.match(text) or UPDATE_ONE_FILE.match(text))


@dataclass(frozen=True)
class Check:
    identifiers: tuple[str, ...]
    hallucinated: tuple[str, ...]
    specific: bool
    generic: bool
    imperative: bool
    well_formed: bool
    words: int


def check(subject: str, diff: str) -> Check:
    names = identifiers(subject)
    touched = changed_names(diff)
    words = {w.lower() for w in WORD.findall(subject)} | {n.lower() for n in names}
    generic = is_generic(subject)
    return Check(
        identifiers=tuple(names),
        hallucinated=tuple(n for n in names if not in_diff(n, diff)),
        specific=not generic and bool(words & touched),
        generic=generic,
        imperative=is_imperative(subject),
        well_formed=well_formed(subject),
        words=len(subject.split()),
    )


def per_example(subjects: Sequence[str], diffs: Sequence[str]) -> dict[str, list[float]]:
    """One 0/1 (or count) per example for every metric, ready for bootstrapping."""
    checks = [check(s, d) for s, d in zip(subjects, diffs, strict=True)]
    # A subject given to three or more different diffs is generic whatever it says.
    repeats = Counter(normalise(s) for s in subjects)
    return {
        "hallucinated_pct": [float(bool(c.hallucinated)) for c in checks],
        "names_identifier_pct": [float(bool(c.identifiers)) for c in checks],
        "specific_pct": [float(c.specific) for c in checks],
        "generic_pct": [
            float(c.generic or repeats[normalise(s)] >= 3)
            for c, s in zip(checks, subjects, strict=True)
        ],
        "imperative_pct": [float(c.imperative) for c in checks],
        "well_formed_pct": [float(c.well_formed) for c in checks],
        "words": [float(c.words) for c in checks],
    }


def summarise(
    subjects: Sequence[str], diffs: Sequence[str], repos: Sequence[str]
) -> dict[str, Any]:
    values = per_example(subjects, diffs)
    out: dict[str, Any] = {}
    for name, vals in values.items():
        factor = 1 if name == "words" else 100
        out[name] = {
            "micro": stratified_bootstrap(vals, repos).scaled(factor),
            "macro": stratified_bootstrap(vals, repos, macro=True).scaled(factor),
        }
    lengths = sorted(values["words"])
    out["words_quantiles"] = {
        "p10": lengths[len(lengths) // 10],
        "median": statistics.median(lengths),
        "p90": lengths[(9 * len(lengths)) // 10],
    }
    identifier_subjects = [
        c for c in (check(s, d) for s, d in zip(subjects, diffs, strict=True)) if c.identifiers
    ]
    out["hallucinated_among_identifier_subjects"] = {
        "n": len(identifier_subjects),
        "hallucinated": sum(bool(c.hallucinated) for c in identifier_subjects),
    }
    first = Counter((s.split() or [""])[0].lower() for s in subjects)
    out["opening_word_top10"] = first.most_common(10)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--test", type=Path, default=Path("eval/test.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("eval/predictions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/metrics_v1.json"))
    args = parser.parse_args(argv)

    examples = [json.loads(line) for line in args.test.open()]
    predictions = [json.loads(line) for line in args.predictions.open()]
    if [e["sha"] for e in examples] != [p["sha"] for p in predictions]:
        raise SystemExit("test set and predictions are not in the same order")
    diffs, repos = [e["diff"] for e in examples], [e["repo"] for e in examples]
    systems = {"reference": [p["reference"] for p in predictions]}
    systems |= {n: [p[n] for p in predictions] for n in ("heuristic", "base", "finetuned")}

    report: dict[str, Any] = {
        "test_examples": len(examples),
        "systems": {n: summarise(s, diffs, repos) for n, s in systems.items()},
    }
    base = per_example(systems["base"], diffs)
    tuned = per_example(systems["finetuned"], diffs)
    report["finetuned_minus_base"] = {
        name: stratified_bootstrap(
            [t - b for t, b in zip(tuned[name], base[name], strict=True)], repos
        ).scaled(1 if name == "words" else 100)
        for name in base
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for name, summary in report["systems"].items():
        row = {k: v["micro"][0] for k, v in summary.items() if isinstance(v, dict) and "micro" in v}
        print(f"{name:10}", row)
    print("finetuned - base:", {k: v for k, v in report["finetuned_minus_base"].items()})


if __name__ == "__main__":
    main()

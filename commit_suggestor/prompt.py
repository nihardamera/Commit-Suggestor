"""The one prompt, shared by training, evaluation and the CLI.

If the model is trained on one wording and asked with another at inference
time, any difference in quality is partly a prompt mismatch rather than the
fine-tune. Keeping a single builder rules that out.
"""

from __future__ import annotations

SYSTEM = "You write concise, accurate git commit messages."

INSTRUCTION = (
    "Write the subject line of a git commit message for the diff below. "
    "Use the imperative mood, at most 72 characters, and no trailing period. "
    "Reply with the subject line only."
)

#: Diffs longer than this are cut. Roughly 850 tokens with the Qwen tokenizer,
#: which keeps a full training example inside a 1,024-token sequence.
MAX_DIFF_CHARS = 3000


def truncate_diff(diff: str, limit: int = MAX_DIFF_CHARS) -> str:
    """Cut at a line boundary and say so, rather than ending mid-line."""
    if len(diff) <= limit:
        return diff
    cut = diff[:limit]
    newline = cut.rfind("\n")
    if newline > limit // 2:
        cut = cut[:newline]
    return cut + "\n... (diff truncated)"


#: 1: the diff alone (v1). 2: optionally preceded by the repository's recent
#: subjects. Without them, version 2 builds exactly the version 1 prompt.
PROMPT_VERSION = 2

CONTEXT_INTRO = (
    "Recent commit subjects in this repository, newest first. Follow their style, "
    "but describe only the diff below:"
)


def build_messages(diff: str, recent_subjects: list[str] | None = None) -> list[dict[str, str]]:
    """Chat messages for one diff, without the answer."""
    context = ""
    if recent_subjects:
        listed = "\n".join(f"- {s}" for s in recent_subjects)
        context = f"{CONTEXT_INTRO}\n{listed}\n\n"
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"{INSTRUCTION}\n\n{context}```diff\n{truncate_diff(diff)}\n```",
        },
    ]


def clean_reply(text: str) -> str:
    """Take the first line of a model reply that is not a code fence, and tidy it.

    Quotes or backticks are removed only when they wrap the whole line. Stripping
    them from both ends regardless cut the closing backtick off a subject that
    merely ended in inline code, e.g. "Add note about the `include` directive".
    """
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        wrap = line[0]
        if len(line) > 1 and wrap in "`\"'" and line[-1] == wrap and line.count(wrap) == 2:
            line = line[1:-1].strip()
        if line:
            return line.rstrip(".")
    return ""


MAX_SUBJECT_CHARS = 72
#: Where a long subject can end and still read as a whole: before the clause
#: that adds detail ("... in docs/x.rst", "... instead of y", "...: details").
CUT_BEFORE = (
    ": ",
    ", ",
    " in ",
    " for ",
    " to ",
    " with ",
    " instead of ",
    " from ",
    " when ",
    " so ",
    " and ",
    " because ",
    " by ",
    " on ",
    " via ",
)


def fit_subject(subject: str, limit: int = MAX_SUBJECT_CHARS) -> str:
    """Shorten a subject over the limit by dropping its last clauses.

    The DPO model sometimes writes past 72 characters (2% of validation
    subjects). Rather than cut mid-word, end before the last connective that
    leaves at most `limit` characters, at least three words, and no inline
    code left open; failing that, at the last word boundary that fits.
    Subjects within the limit are returned unchanged.
    """
    if len(subject) <= limit:
        return subject

    def ok(prefix: str) -> bool:
        return len(prefix.split()) >= 3 and prefix.count("`") % 2 == 0

    cuts = [
        i
        for sep in CUT_BEFORE
        for i in range(len(subject))
        if subject.startswith(sep, i) and i <= limit and ok(subject[:i])
    ]
    if cuts:
        return subject[: max(cuts)].rstrip(" .,:;-")
    words = subject[: limit + 1].rsplit(" ", 1)[0]
    if words.count("`") % 2:
        words = words.rsplit("`", 1)[0]
    return words.rstrip(" .,:;-")

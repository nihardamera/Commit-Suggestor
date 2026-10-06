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


def build_messages(diff: str) -> list[dict[str, str]]:
    """Chat messages for one diff, without the answer."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{INSTRUCTION}\n\n```diff\n{truncate_diff(diff)}\n```"},
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

"""The commit-suggestor command: suggest a subject, install the git hooks, show stats.

    commit-suggestor                       # suggest a subject for the staged changes
    git diff HEAD~1 | commit-suggestor -   # or for any diff on stdin
    commit-suggestor install-hook [--feedback] [REPO]
    commit-suggestor stats                 # how often suggestions were kept (needs --feedback)
    commit-suggestor download              # fetch the published model (MLX or GGUF)

The model is the first of: --model, $COMMIT_SUGGESTOR_MODEL, a model installed in
~/.cache/commit-suggestor/, or the v1 adapter in this checkout. A model folder (MLX)
or .gguf file may carry a commit-suggestor.json saying whether it was trained
with repo-style context; if so, the last five subjects of the repository are
shown to it, as in training.

Hook mode (--hook, used by the prepare-commit-msg hook) never downloads, skips
very large diffs, and with feedback on records the suggestion so the
post-commit hook can compare it with the subject actually committed. Feedback
stays in ~/.local/share/commit-suggestor/feedback.jsonl and is never sent anywhere.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from commit_suggestor.prompt import build_messages, clean_reply, fit_subject

V1_BASE = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
#: The published model, at a fixed revision (set by scripts/publish_hf.py's output).
HF_REPO = "nihar-damera/commit-suggestor"
HF_REVISION: str | None = None
V1_ADAPTER = Path(__file__).resolve().parent.parent / "adapters"
MODEL_HOME = Path(
    os.environ.get("COMMIT_SUGGESTOR_HOME", Path.home() / ".cache" / "commit-suggestor")
)
FEEDBACK = Path(
    os.environ.get(
        "COMMIT_SUGGESTOR_FEEDBACK",
        Path.home() / ".local" / "share" / "commit-suggestor" / "feedback.jsonl",
    )
)
#: In hook mode, diffs larger than this are skipped: the model sees only the
#: first 3,000 characters anyway, and the subject of such a change is yours to write.
HOOK_MAX_DIFF_CHARS = 40_000
EXIT_NO_MODEL, EXIT_SKIPPED = 3, 4


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout


def staged_diff() -> str:
    return git("diff", "--cached", "--no-color", "-M")


def find_model(explicit: str | None) -> tuple[Path | None, dict[str, Any]]:
    """The model to use and its metadata; (None, {}) means the v1 adapter or nothing."""
    for candidate in (explicit, os.environ.get("COMMIT_SUGGESTOR_MODEL")):
        if candidate:
            path = Path(candidate).expanduser()
            return path, model_info(path)
    if MODEL_HOME.exists():
        installed = sorted(
            p for p in MODEL_HOME.iterdir() if p.suffix == ".gguf" or (p / "config.json").exists()
        )
        if installed:
            return installed[-1], model_info(installed[-1])
    return None, {}


def model_info(path: Path) -> dict[str, Any]:
    meta = path / "commit-suggestor.json" if path.is_dir() else path.with_suffix(".json")
    return json.loads(meta.read_text()) if meta.exists() else {}


def suggest(diff: str, model: Path | None, info: dict[str, Any], context: list[str] | None) -> str:
    from commit_suggestor.backends import MLXBackend, load_backend
    from commit_suggestor.prepare import clean_diff

    cleaned, files = clean_diff(diff)
    if not files:
        cleaned = diff  # e.g. a diff without `diff --git` headers; use it as given
    if model is not None:
        backend = load_backend(model)
    elif (V1_ADAPTER / "adapters.safetensors").exists():
        backend = MLXBackend(V1_BASE, V1_ADAPTER)
    else:
        sys.exit("no model: run 'commit-suggestor download', or pass --model")
    use_context = context if info.get("repo_context") else None
    return fit_subject(clean_reply(backend.generate(build_messages(cleaned, use_context))))


def cmd_suggest(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if args.hook:
        os.environ["HF_HUB_OFFLINE"] = "1"  # a commit must never wait for a download
    diff = sys.stdin.read() if args.source == "-" else staged_diff()
    if not diff.strip():
        sys.exit("nothing to describe: no staged changes (git add something first)")
    if args.hook and len(diff) > HOOK_MAX_DIFF_CHARS:
        sys.exit(EXIT_SKIPPED)
    model, info = find_model(args.model)
    if args.hook and model is None and not (V1_ADAPTER / "adapters.safetensors").exists():
        sys.exit(EXIT_NO_MODEL)
    context = None
    if not args.no_context and args.source != "-":
        from commit_suggestor.context import recent_subjects

        context = recent_subjects(Path.cwd())
    started = time.time()
    subject = suggest(diff, model, info, context)
    if not subject:
        sys.exit(1)
    print(subject)
    if args.hook:
        name = info.get("name", model.name if model else "v1")
        print(f"# Suggested by commit-suggestor ({name}). Edit or replace it.")
        if git("config", "--bool", "commit-suggestor.feedback", check=False).strip() == "true":
            remember(diff, subject, name, time.time() - started)


def last_path() -> Path:
    return Path(git("rev-parse", "--git-dir").strip()) / "commit-suggestor" / "last.json"


def remember(diff: str, subject: str, model: str, seconds: float) -> None:
    path = last_path()
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "suggestion": subject,
                "model": model,
                "seconds": round(seconds, 2),
                "at": time.time(),
            }
        )
    )


def outcome(suggestion: str, final: str) -> tuple[str, float]:
    """accepted (unchanged), edited (mostly kept) or replaced, with the similarity."""
    a, b = suggestion.strip().rstrip(".").lower(), final.strip().rstrip(".").lower()
    similarity = difflib.SequenceMatcher(None, a, b).ratio()
    return ("accepted" if a == b else "edited" if similarity >= 0.5 else "replaced"), similarity


def cmd_record_feedback(_: argparse.Namespace) -> None:
    """Run by the post-commit hook: compare the last suggestion with what was committed."""
    path = last_path()
    if not path.exists():
        return
    last = json.loads(path.read_text())
    path.unlink()
    if time.time() - last["at"] > 3600:
        return  # stale: not the suggestion for this commit
    committed = git("diff", "--no-color", "-M", "HEAD~1", "HEAD", check=False)
    if hashlib.sha256(committed.encode()).hexdigest() != last["diff_sha256"]:
        return  # the commit is not the diff the suggestion was for
    final = git("log", "-1", "--format=%s").strip()
    result, similarity = outcome(last["suggestion"], final)
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "repo": Path(git("rev-parse", "--show-toplevel").strip()).name,
                    "model": last["model"],
                    "suggestion": last["suggestion"],
                    "committed": final,
                    "outcome": result,
                    "similarity": round(similarity, 3),
                    "seconds": last["seconds"],
                }
            )
            + "\n"
        )


def cmd_download(args: argparse.Namespace) -> None:
    """Fetch the published model at its pinned revision: MLX on Apple Silicon, else GGUF."""
    from huggingface_hub import snapshot_download

    from commit_suggestor.backends import mlx_available

    if HF_REVISION is None:
        sys.exit("no published model yet; pass --model with a local model instead")
    use_mlx = mlx_available() if args.format == "auto" else args.format == "mlx"
    patterns = ["mlx/*"] if use_mlx else ["*.gguf", "commit-suggestor-q4_k_m.json"]
    target = MODEL_HOME / f"download-{HF_REVISION[:12]}"
    # The Hub checks each file against the hash recorded in the repository.
    snapshot_download(HF_REPO, revision=HF_REVISION, allow_patterns=patterns, local_dir=target)
    model = target / "mlx" if use_mlx else target / "commit-suggestor-q4_k_m.gguf"
    link = MODEL_HOME / ("commit-suggestor-mlx" if use_mlx else "commit-suggestor-q4_k_m.gguf")
    link.unlink(missing_ok=True)
    link.symlink_to(model)
    if not use_mlx:
        link.with_suffix(".json").unlink(missing_ok=True)
        link.with_suffix(".json").symlink_to(target / "commit-suggestor-q4_k_m.json")
    print(f"model ready: {link}")


def cmd_stats(_: argparse.Namespace) -> None:
    from commit_suggestor.stats import wilson

    rows = [json.loads(line) for line in FEEDBACK.open()] if FEEDBACK.exists() else []
    if not rows:
        print("no feedback yet: install the hooks with --feedback and commit as usual")
        return
    print(f"{len(rows)} suggested commits, {FEEDBACK}")
    for name in ("accepted", "edited", "replaced"):
        n = sum(r["outcome"] == name for r in rows)
        rate, low, high = wilson(n, len(rows)).scaled(100, 1)
        print(f"  {name:9} {n:4}  {rate}% [{low}, {high}]")
    seconds = sorted(r["seconds"] for r in rows)
    print(f"  median time to suggest: {seconds[len(seconds) // 2]:.2f} s")


PREPARE_HOOK = """#!/bin/sh
# commit-suggestor: pre-fill the commit message with a suggested subject line.
# Installed by `commit-suggestor install-hook`. Acts only on a plain `git commit`
# (no -m, merge, squash or amend) and never blocks one: on any failure, or after
# $COMMIT_SUGGESTOR_TIMEOUT seconds (default 8), the message is left as it was.
MSG_FILE="$1"; SOURCE="$2"
[ -n "$SOURCE" ] && exit 0
[ -n "$COMMIT_SUGGESTOR_DISABLE" ] && exit 0
OUT=$(mktemp) || exit 0
"__PYTHON__" -m commit_suggestor.cli --hook > "$OUT" 2>/dev/null &
PID=$!
( sleep "${COMMIT_SUGGESTOR_TIMEOUT:-8}"; kill -9 "$PID" 2>/dev/null ) &
WATCH=$!
wait "$PID" 2>/dev/null; STATUS=$?
kill "$WATCH" 2>/dev/null
if [ "$STATUS" -eq 3 ]; then
  echo "commit-suggestor: no model installed yet; run 'commit-suggestor download'" >&2
fi
if [ "$STATUS" -eq 0 ] && [ -s "$OUT" ]; then
  { cat "$OUT"; cat "$MSG_FILE"; } > "$MSG_FILE.tmp" && mv "$MSG_FILE.tmp" "$MSG_FILE"
fi
rm -f "$OUT"
exit 0
"""

POST_HOOK = """#!/bin/sh
# commit-suggestor: record whether the suggested subject was kept (local only).
# Installed by `commit-suggestor install-hook --feedback`. Never fails a commit.
"__PYTHON__" -m commit_suggestor.cli record-feedback >/dev/null 2>&1
exit 0
"""


def cmd_install_hook(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    hooks = Path(git("-C", str(repo), "rev-parse", "--git-path", "hooks").strip())
    hooks = hooks if hooks.is_absolute() else repo / hooks
    wanted = {"prepare-commit-msg": PREPARE_HOOK}
    if args.feedback:
        wanted["post-commit"] = POST_HOOK
    for name, body in wanted.items():
        path = hooks / name
        if path.exists() and "commit-suggestor" not in path.read_text():
            sys.exit(f"{path} exists and is not ours; not overwriting")
        path.write_text(body.replace("__PYTHON__", sys.executable))
        path.chmod(0o755)
        print(f"installed {path}")
    if args.feedback:
        git("-C", str(repo), "config", "commit-suggestor.feedback", "true")
        print(f"feedback on: outcomes go to {FEEDBACK} and never leave this machine")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    commands = {"install-hook", "stats", "record-feedback", "suggest", "download"}
    if not argv or argv[0] not in commands:
        argv = ["suggest", *argv]
    parser = argparse.ArgumentParser(
        prog="commit-suggestor", description=(__doc__ or "").splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("suggest", help="suggest a subject (the default)")
    s.add_argument("source", nargs="?", help="'-' to read a diff from stdin")
    s.add_argument("--model", help="an MLX model folder or a .gguf file")
    s.add_argument(
        "--no-context", action="store_true", help="do not show the repository's recent subjects"
    )
    s.add_argument("--hook", action="store_true", help=argparse.SUPPRESS)
    s.set_defaults(run=cmd_suggest)
    h = sub.add_parser("install-hook", help="pre-fill messages in a repository")
    h.add_argument("repo", nargs="?", default=".")
    h.add_argument(
        "--feedback", action="store_true", help="also record locally whether suggestions were kept"
    )
    h.set_defaults(run=cmd_install_hook)
    sub.add_parser("stats", help="how often suggestions were kept").set_defaults(run=cmd_stats)
    d = sub.add_parser("download", help="fetch the published model")
    d.add_argument("--format", choices=["auto", "mlx", "gguf"], default="auto")
    d.set_defaults(run=cmd_download)
    sub.add_parser("record-feedback", help=argparse.SUPPRESS).set_defaults(run=cmd_record_feedback)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

"""Single-keystroke input and diff colouring for the rating tools."""

from __future__ import annotations

import sys

COLOUR = sys.stdout.isatty()


def getkey(prompt: str, allowed: str) -> str:
    """Read one key from `allowed` without waiting for Enter (Enter if not a terminal)."""
    print(prompt, end=" ", flush=True)
    while True:
        key = _read_one().lower()
        if key == "\x03":  # Ctrl-C
            raise KeyboardInterrupt
        if key in allowed:
            print(key)
            return key


def _read_one() -> str:
    if not sys.stdin.isatty():
        return (sys.stdin.readline().strip() or "\n")[0]
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOUR else text


def show_diff(diff: str) -> None:
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            print(paint(line, "1"))
        elif line.startswith(("+++", "---")):
            print(paint(line, "2"))
        elif line.startswith("@@"):
            print(paint(line, "36"))
        elif line.startswith("+"):
            print(paint(line, "32"))
        elif line.startswith("-"):
            print(paint(line, "31"))
        else:
            print(line)


def clear() -> None:
    if COLOUR:
        print("\033[2J\033[H", end="")

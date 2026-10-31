"""Serving benchmark: cold load, warm latency and peak memory of a shipped model.

    uv run --extra mlx python scripts/bench_serving.py --model data/models/dpo-mlx

Loads the model through commit_suggestor.backends, as the CLI does, then times
one subject per test diff on the first N diffs (one untimed warm-up first).
Run in a fresh process, so the load time includes the import and the read from
disk. Writes or updates eval/serving.json, one entry per runtime and machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from commit_suggestor.prompt import build_messages, clean_reply, fit_subject  # noqa: E402

OUT = ROOT / "eval" / "serving.json"


def machine() -> str:
    if sys.platform == "darwin":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        mem = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            ).stdout.strip()
        )
        return f"{chip}, {mem // 2**30} GB"
    return platform.processor() or platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    tests = [json.loads(line) for line in (ROOT / "eval" / "test.jsonl").open()][: args.n + 1]
    started = time.perf_counter()
    from commit_suggestor.backends import load_backend

    backend = load_backend(args.model)
    load_s = time.perf_counter() - started
    backend.generate(build_messages(tests[0]["diff"]))  # warm-up, untimed
    seconds = []
    for t in tests[1:]:
        t0 = time.perf_counter()
        fit_subject(clean_reply(backend.generate(build_messages(t["diff"]))))
        seconds.append(time.perf_counter() - t0)
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10
    q = statistics.quantiles(seconds, n=20)
    entry = {
        "runtime": backend.name,
        "model": args.model.name,
        "machine": machine(),
        "diffs_timed": len(seconds),
        "cold_load_s": round(load_s, 2),
        "warm_p50_s": round(statistics.median(seconds), 3),
        "warm_p95_s": round(q[18], 3),
        "warm_max_s": round(max(seconds), 3),
        "peak_rss_mb": round(rss_mb),
        "measured": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results[f"{entry['runtime']} / {entry['machine']}"] = entry
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()

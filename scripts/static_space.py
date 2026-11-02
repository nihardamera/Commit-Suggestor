"""Build the static demo Space: every test diff with the subjects each model gave.

    uv run python scripts/static_space.py data/static-space

Hugging Face hosts static Spaces for free but needs PRO for Gradio ones, so the
public demo is one page built from committed predictions: the 450 test diffs
(three projects never seen in training), the author's subject, and the
subjects from v1, SFT-v2 and the shipped DPO model (MLX 4-bit, guarded). It
runs no model. space/app.py is the live version.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from commit_suggestor.eval_v2 import predictions, test_set  # noqa: E402

SYSTEMS = {"dpo_guarded": "DPO (shipped)", "sft_v2_guarded": "SFT-v2", "v1": "v1"}

README = """---
title: Commit Suggestor
emoji: 📝
colorFrom: gray
colorTo: green
sdk: static
pinned: false
license: apache-2.0
short_description: Commit subjects a 1.5B model wrote for 450 unseen diffs
---

A static page: the 450 test diffs and the subject each model wrote for them,
from the committed evaluation files. It runs no model. The model is
[nihar-damera/commit-suggestor](https://huggingface.co/nihar-damera/commit-suggestor);
source and evaluation:
[github.com/nihardamera/Commit-Suggestor](https://github.com/nihardamera/Commit-Suggestor).
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Commit Suggestor</title>
<style>
:root { --bg: #fbfbfa; --fg: #1d1d1b; --muted: #6b6b66; --line: #e3e2de;
  --card: #fff; --add: #e6f4ea; --del: #fbe9e7; --hunk: #eef1f8; --accent: #2f6f4e; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161615; --fg: #e8e8e4; --muted: #9a9a94; --line: #2e2e2b;
    --card: #1e1e1c; --add: #17331f; --del: #3a1d1a; --hunk: #1f2533; --accent: #7cc49c; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
p { margin: 0 0 12px; color: var(--muted); }
a { color: var(--accent); }
.bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 16px 0; }
select, button { font: inherit; padding: 6px 12px; border: 1px solid var(--line);
  border-radius: 6px; background: var(--card); color: var(--fg); cursor: pointer; }
.pos { color: var(--muted); margin-left: auto; font-variant-numeric: tabular-nums; }
table { width: 100%; border-collapse: collapse; background: var(--card);
  border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
td { padding: 8px 12px; border-top: 1px solid var(--line); vertical-align: top; }
tr:first-child td { border-top: 0; }
td:first-child { width: 9.5em; color: var(--muted); white-space: nowrap; }
tr.ship td:last-child { font-weight: 600; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
pre { margin: 16px 0 0; padding: 12px 0; background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; overflow-x: auto;
  font: 12.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
pre span { display: block; padding: 0 12px; white-space: pre; min-width: max-content; }
.a { background: var(--add); } .d { background: var(--del); }
.h { background: var(--hunk); color: var(--muted); }
.f { font-weight: 600; }
footer { margin-top: 24px; font-size: 0.9em; color: var(--muted); }
</style>
</head>
<body>
<main>
<h1>Commit Suggestor</h1>
<p>A 1.5B code model fine-tuned to write git commit subjects. These are the __N__
test diffs, from three projects it never saw in training, with the subject each
version wrote. Stored results: this page runs no model.
<a href="https://huggingface.co/nihar-damera/commit-suggestor">Model</a> ·
<a href="https://github.com/nihardamera/Commit-Suggestor">source and evaluation</a></p>
<div class="bar">
  <select id="repo" aria-label="Project"></select>
  <button id="prev" aria-label="Previous diff">←</button>
  <button id="next" aria-label="Next diff">→</button>
  <button id="rand">Random diff</button>
  <span class="pos" id="pos"></span>
</div>
<table id="subjects"></table>
<pre id="diff"></pre>
<footer>__SUMMARY__ The test set holds diffs of up to 3,000 characters, shown
exactly as the model saw them.</footer>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const repoSel = document.getElementById("repo");
const repos = [...new Set(D.rows.map(r => r.repo))];
for (const v of ["all", ...repos]) {
  const o = document.createElement("option");
  o.value = v;
  o.textContent = v === "all" ? `All projects (${D.rows.length})`
    : `${v} (${D.rows.filter(r => r.repo === v).length})`;
  repoSel.append(o);
}
let rows = D.rows, i = 0;
function esc(s) { const e = document.createElement("span"); e.textContent = s; return e.innerHTML; }
function show() {
  const r = rows[i];
  document.getElementById("pos").textContent = `${i + 1} / ${rows.length}`;
  const t = [[`Author`, r.author, ""]].concat(D.systems.map(([k, label], j) =>
    [label, r[k], j === 0 ? "ship" : ""]));
  document.getElementById("subjects").innerHTML =
    `<tr><td>Commit</td><td><a href="https://github.com/${r.repo}/commit/${r.sha}">` +
    `<code>${esc(r.repo)}@${r.sha.slice(0, 10)}</code></a></td></tr>` +
    t.map(([k, v, c]) => `<tr class="${c}"><td>${k}</td><td>${esc(v)}</td></tr>`).join("");
  document.getElementById("diff").innerHTML = r.diff.split("\\n").map(l => {
    const c = l.startsWith("diff --git") ? "f" : l.startsWith("@@") ? "h"
      : l.startsWith("+") && !l.startsWith("+++") ? "a"
      : l.startsWith("-") && !l.startsWith("---") ? "d" : "";
    return `<span class="${c}">${esc(l) || " "}</span>`;
  }).join("");
  try { history.replaceState(null, "", `#${r.sha.slice(0, 10)}`); } catch (e) {}
}
function filter() {
  rows = repoSel.value === "all" ? D.rows : D.rows.filter(r => r.repo === repoSel.value);
  i = 0; show();
}
repoSel.onchange = filter;
function go(k) { i = (k + rows.length) % rows.length; show(); }
document.getElementById("prev").onclick = () => go(i - 1);
document.getElementById("next").onclick = () => go(i + 1);
document.getElementById("rand").onclick = () => go(Math.floor(Math.random() * rows.length));
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") document.getElementById("next").click();
  if (e.key === "ArrowLeft") document.getElementById("prev").click();
});
const start = location.hash.slice(1);
const found = start ? D.rows.findIndex(r => r.sha.startsWith(start)) : -1;
if (found >= 0) i = found;
show();
</script>
</body>
</html>
"""


def build(out: Path) -> None:
    tests, _ = test_set()
    preds = {name: predictions(name) for name in SYSTEMS}
    rows = [
        {
            "repo": t["repo"],
            "sha": t["sha"],
            "author": t["subject"],
            "diff": t["diff"],
            **{name: preds[name][k] for name in SYSTEMS},
        }
        for k, t in enumerate(tests)
    ]
    results = json.loads((ROOT / "eval" / "v2" / "results.json").read_text())
    judge = results["judge"]["dpo_guarded_vs_sft_v2_guarded"]["a_win_rate_excl_ties_pct"]
    summary = (
        f"On these diffs an LLM judge preferred DPO to SFT-v2 {judge[0]}% of the time "
        f"[{judge[1]}, {judge[2]}], ties excluded."
    )
    data = json.dumps({"systems": list(SYSTEMS.items()), "rows": rows}, ensure_ascii=False)
    page = (
        PAGE.replace("__N__", str(len(rows)))
        .replace("__SUMMARY__", html.escape(summary))
        .replace("__DATA__", data.replace("</", "<\\/"))
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    print(f"static space -> {out} ({len(rows)} diffs, {len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "static-space")

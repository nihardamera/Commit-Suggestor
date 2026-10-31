"""Commit Suggestor demo: paste a diff, get a commit subject line.

Runs on a free CPU Hugging Face Space with llama.cpp and the Q4_K_M GGUF
build of the model. The diff is cleaned and the prompt built by the same code
as training, evaluation and the CLI (commit_suggestor/prompt.py and
prepare.py, copied here when the Space is published by scripts/publish_hf.py).

Nothing that is pasted here is stored or logged.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from functools import lru_cache

import gradio as gr

from commit_suggestor.prepare import clean_diff
from commit_suggestor.prompt import build_messages, clean_reply, fit_subject

MODEL_REPO = os.environ.get("MODEL_REPO", "nihar-damera/commit-suggestor")
MODEL_FILE = "commit-suggestor-q4_k_m.gguf"
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
MAX_INPUT_CHARS = 20_000

EXAMPLE = """diff --git a/src/client.py b/src/client.py
--- a/src/client.py
+++ b/src/client.py
@@ -10,6 +10,9 @@ class Client:
     def connect(self):
-        sock = open_socket(self.host)
+        for attempt in range(self.max_retries):
+            try:
+                sock = open_socket(self.host, timeout=self.timeout)
+                break
+            except TimeoutError:
+                time.sleep(2 ** attempt)
         return sock"""


@lru_cache(maxsize=1)
def model() -> Callable[[list[dict[str, str]]], str]:
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    path = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION)
    llm = Llama(model_path=path, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)

    def generate(messages: list[dict[str, str]]) -> str:
        out = llm.create_chat_completion(messages=messages, max_tokens=40, temperature=0.0)
        return str(out["choices"][0]["message"]["content"] or "")

    return generate


def suggest(
    diff: str, generate: Callable[[list[dict[str, str]]], str] | None = None
) -> tuple[str, str]:
    """(subject, note) for a pasted diff."""
    if not diff.strip():
        return "", "Paste the output of `git diff --cached` (or any unified diff)."
    if len(diff) > MAX_INPUT_CHARS:
        return "", (
            f"That diff is {len(diff):,} characters; the demo takes up to "
            f"{MAX_INPUT_CHARS:,}. The model only reads the first 3,000 anyway."
        )
    cleaned, files = clean_diff(diff)
    if not files:
        cleaned = diff  # no `diff --git` headers: use it as given
    started = time.time()
    subject = fit_subject(clean_reply((generate or model())(build_messages(cleaned))))
    note = f"{time.time() - started:.1f} s on a shared CPU."
    if len(subject) > 72:
        note += " Longer than 72 characters: worth shortening."
    return subject, note


with gr.Blocks(title="Commit Suggestor", analytics_enabled=False) as demo:
    gr.Markdown(
        "# Commit Suggestor\n"
        "A 1.5B code model fine-tuned to write the subject line of a git commit from its "
        "diff. Paste a diff, get a subject. The evaluation, including where it falls "
        "short, is in the [model card](https://huggingface.co/" + MODEL_REPO + ")."
    )
    diff_box = gr.Textbox(
        label="Diff",
        lines=18,
        max_lines=40,
        placeholder="git diff --cached | pbcopy, then paste here",
    )
    button = gr.Button("Suggest a subject", variant="primary")
    subject_box = gr.Textbox(label="Suggested subject", buttons=["copy"])
    note_box = gr.Markdown()
    button.click(suggest, inputs=diff_box, outputs=[subject_box, note_box])
    gr.Examples([[EXAMPLE]], inputs=diff_box)
    gr.Markdown("Nothing you paste is stored. Runs on llama.cpp with a 4-bit model.")

if __name__ == "__main__":
    demo.queue(max_size=16).launch()

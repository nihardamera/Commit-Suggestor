---
title: Commit Suggestor
emoji: 📝
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Paste a git diff, get a commit subject line
---

# Commit Suggestor demo

Paste a diff, get a commit subject. The model is
[nihar-damera/commit-suggestor](https://huggingface.co/nihar-damera/commit-suggestor),
a LoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct, run here as a 4-bit GGUF with
llama.cpp on a free CPU. Source, evaluation and limitations:
[github.com/nihardamera/Commit-Suggestor](https://github.com/nihardamera/Commit-Suggestor).

Nothing pasted into the demo is stored.

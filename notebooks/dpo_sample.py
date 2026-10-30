"""Sample candidate subjects from SFT-v2 for DPO preference pairs (Kaggle T4).

Pushed by scripts/kaggle_job.py. Reads the SFT-v2 job's output (its final
adapter) and the v2 dataset, takes a seeded sample of TRAINING diffs (never
validation or test), and draws 4 subjects for each at temperature 0.9. The
judge then labels pairs of them on the Mac (commit_suggestor.dpo_data).

Output: /kaggle/working/dpo_samples.jsonl, one line per diff:
    {"sha", "repo", "with_context", "prompt": [messages], "samples": [4 strings]}
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

MODE = "smoke"  # replaced by scripts/kaggle_job.py: "smoke" or "full"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU; see sft_v2.py

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts",
     "transformers==5.22.1", "peft==0.21.2", "accelerate==1.15.0"],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=True)

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
INPUT = Path("/kaggle/input")
DATA = next(INPUT.rglob("train.meta.jsonl")).parent
ADAPTER = next(p.parent for p in INPUT.rglob("adapter_config.json") if p.parent.name == "final")
OUT = Path("/kaggle/working")
N_DIFFS = 3000 if MODE == "full" else 16
N_SAMPLES, TEMPERATURE, TOP_P = 4, 0.9, 0.95
SEED = 1

sys.path.insert(0, str(DATA))
from prompt import build_messages, clean_reply  # noqa: E402

torch.manual_seed(SEED)
rows = [json.loads(line) for line in (DATA / "train.meta.jsonl").open()]
rows = random.Random(SEED).sample(rows, N_DIFFS)

tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base, ADAPTER).eval()

with (OUT / "dpo_samples.jsonl").open("w") as fh, torch.no_grad():
    for i in range(0, len(rows), 16):
        batch = rows[i:i + 16]
        prompts = [build_messages(r["diff"], r["context"] if r["with_context"] else None)
                   for r in batch]
        texts = [tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
                 for p in prompts]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=40, do_sample=True, temperature=TEMPERATURE,
                             top_p=TOP_P, num_return_sequences=N_SAMPLES,
                             pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
        for j, (row, prompt) in enumerate(zip(batch, prompts, strict=True)):
            samples = [clean_reply(t) for t in decoded[j * N_SAMPLES:(j + 1) * N_SAMPLES]]
            fh.write(json.dumps({"sha": row["sha"], "repo": row["repo"],
                                 "with_context": row["with_context"], "prompt": prompt,
                                 "samples": samples}) + "\n")
        print(f"{i + len(batch)}/{len(rows)}", flush=True)

(OUT / "sample_info.json").write_text(json.dumps({
    "mode": MODE, "diffs": len(rows), "samples_per_diff": N_SAMPLES,
    "temperature": TEMPERATURE, "top_p": TOP_P, "seed": SEED, "adapter": str(ADAPTER),
}, indent=2))

"""SFT-v2 on a Kaggle T4: LoRA on Qwen2.5-Coder-1.5B-Instruct with the v2 data.

Pushed and run by scripts/kaggle_job.py, which sets MODE below. Everything
except the data and the number of steps matches v1 (config/lora.yaml), so the
v1 -> v2 comparison is about the data:

    v1 (MLX, M4)                    v2 (this script, T4)
    4-bit base (QLoRA)              fp16 base (T4 has no bf16)
    rank 8, scale 20                rank 8, lora_alpha 160 (scale = alpha / rank = 20)
    top 16 of 28 layers, all 7      same
      linear modules
    lr 5e-5 cosine, 50 warmup       same
    batch 4, 600 steps              batch 4, one epoch of 27K examples
    loss on the subject only        same (prompt-completion data)

Outputs in /kaggle/working:
    adapters/step-600, adapters/step-2500, adapters/final   LoRA adapters
    merged/                       the final adapter merged into the fp16 base
    log_history.json              training and validation loss
    predictions_valid.jsonl       greedy subjects on validation, per checkpoint
    predictions_test.jsonl        final model on the 450 test diffs, with and
                                  without repo context (for the conversion check)
    run_info.json                 versions, GPU, timings, settings
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODE = "smoke"  # replaced by scripts/kaggle_job.py: "smoke" or "full"

# The "T4 x2" machine shows two GPUs, and with more than one visible the
# Trainer wraps the model in nn.DataParallel, which TRL's SFTTrainer refuses.
# This is a single-GPU job, so show it one. Must happen before torch loads.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-warn-conflicts",
        "transformers==5.22.1",
        "trl==1.17.0",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "datasets==5.1.0",
    ],
    check=True,
)
# Kaggle's image ships torchao 0.10, and PEFT 0.21 refuses to load LoRA layers
# while a torchao older than 0.16 is installed. Nothing here uses torchao.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=True)

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA = next(Path("/kaggle/input").rglob("train.jsonl")).parent
OUT = Path("/kaggle/working")
MAX_LENGTH = 1280
CURVE_STEPS = (600, 2500)  # 2,400 examples (v1's amount) and 10,000
SEED = 0
started = time.time()


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).open()]


tokenizer = AutoTokenizer.from_pretrained(BASE)


def as_prompt_completion(rows: list[dict]) -> tuple[list[dict], int]:
    """Split the chat into prompt and answer; drop examples over MAX_LENGTH.

    Truncating would cut the end of the sequence, which is the answer, so long
    examples are dropped instead (10 of 27,356 when the data was built).
    """
    out, dropped = [], 0
    for row in rows:
        messages = row["messages"]
        ids = tokenizer.apply_chat_template(messages, tokenize=True)
        n = len(ids["input_ids"] if hasattr(ids, "keys") else ids)
        if n > MAX_LENGTH:
            dropped += 1
            continue
        out.append({"prompt": messages[:-1], "completion": messages[-1:]})
    return out, dropped


train_rows, train_dropped = as_prompt_completion(load("train.jsonl"))
valid_rows, valid_dropped = as_prompt_completion(load("valid.jsonl"))
if MODE == "smoke":
    train_rows, valid_rows = train_rows[:64], valid_rows[:16]

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={"": 0})
lora = LoraConfig(
    r=8,
    lora_alpha=160,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    layers_to_transform=list(range(12, 28)),
    task_type="CAUSAL_LM",
)


class SaveCurvePoints(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in CURVE_STEPS:
            model.save_pretrained(OUT / "adapters" / f"step-{state.global_step}")


config = SFTConfig(
    output_dir=str(OUT / "trainer"),
    max_length=MAX_LENGTH,
    completion_only_loss=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=20 if MODE == "smoke" else -1,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=50 if MODE == "full" else 2,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=25 if MODE == "full" else 2,
    eval_strategy="steps",
    eval_steps=500 if MODE == "full" else 10,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=Dataset.from_list(train_rows),
    eval_dataset=Dataset.from_list(valid_rows),
    processing_class=tokenizer,
    peft_config=lora,
    callbacks=[SaveCurvePoints()],
)
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
trainer.train()
train_seconds = time.time() - started
trainer.model.save_pretrained(OUT / "adapters" / "final")
tokenizer.save_pretrained(OUT / "adapters" / "final")
(OUT / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))


tokenizer.padding_side = "left"  # batched generation pads on the left


@torch.no_grad()
def generate(peft_model, prompts: list[list[dict]], batch: int = 32) -> list[str]:
    peft_model.eval()
    replies = []
    for i in range(0, len(prompts), batch):
        texts = [
            tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in prompts[i : i + batch]
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(peft_model.device)
        out = peft_model.generate(
            **enc, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        replies += tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
    return replies


# The prompt builder lives in the repository; it is copied in next to this file.
sys.path.insert(0, str(DATA))
from prompt import build_messages, clean_reply  # noqa: E402

# Remove the trained LoRA layers, then load each saved adapter onto the clean base.
base_model = trainer.model.unload()
valid_meta = load("valid.meta.jsonl")[: len(valid_rows)]
valid_prompts = [
    build_messages(m["diff"], m["context"] if m["with_context"] else None) for m in valid_meta
]
tests = load("test.jsonl")
contexts = {c["sha"]: c["context"] for c in load("test_context.jsonl")}
if MODE == "smoke":
    tests = tests[:16]

with (OUT / "predictions_valid.jsonl").open("w") as fh:
    for name in [*(f"step-{s}" for s in CURVE_STEPS), "final"]:
        path = OUT / "adapters" / name
        if not path.exists():
            continue
        peft_model = PeftModel.from_pretrained(base_model, path)
        for meta, reply in zip(valid_meta, generate(peft_model, valid_prompts), strict=True):
            fh.write(
                json.dumps(
                    {
                        "checkpoint": name,
                        "sha": meta["sha"],
                        "repo": meta["repo"],
                        "reference": meta["subject"],
                        "prediction": clean_reply(reply),
                    }
                )
                + "\n"
            )
        base_model = peft_model.unload()

final = PeftModel.from_pretrained(base_model, OUT / "adapters" / "final")
with (OUT / "predictions_test.jsonl").open("w") as fh:
    for use_context in (False, True):
        prompts = [
            build_messages(t["diff"], contexts[t["sha"]] if use_context else None) for t in tests
        ]
        for test, reply in zip(tests, generate(final, prompts), strict=True):
            fh.write(
                json.dumps(
                    {
                        "sha": test["sha"],
                        "repo": test["repo"],
                        "with_context": use_context,
                        "prediction": clean_reply(reply),
                    }
                )
                + "\n"
            )

merged = final.merge_and_unload()
merged.save_pretrained(OUT / "merged", safe_serialization=True)
tokenizer.save_pretrained(OUT / "merged")

info = {
    "mode": MODE,
    "base": BASE,
    "gpu": torch.cuda.get_device_name(0),
    "versions": {
        p: __import__(p).__version__ for p in ("torch", "transformers", "trl", "peft", "datasets")
    },
    "train_examples": len(train_rows),
    "dropped_over_max_length": {"train": train_dropped, "valid": valid_dropped},
    "trainable_parameters": trainable,
    "steps": trainer.state.global_step,
    "train_seconds": round(train_seconds),
    "total_seconds": round(time.time() - started),
    "seed": SEED,
    "env": {k: os.environ.get(k) for k in ("KAGGLE_KERNEL_RUN_TYPE",)},
}
(OUT / "run_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))

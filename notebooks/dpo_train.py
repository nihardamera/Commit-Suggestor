"""DPO on top of SFT-v2 with a beta sweep (Kaggle T4).

Pushed by scripts/kaggle_job.py. Inputs: the v2 dataset, the DPO pairs
dataset (commit_suggestor.dpo_data), and the SFT-v2 job's output.

For each beta the policy starts from the SFT-v2 adapter, and a frozen copy of
the same adapter (named "ref", which TRL uses as the reference) gives the
reference log-probs. Before training, TRL's loss and commit_suggestor.dpo's
are computed on the same batch with dropout off, and both are saved.

Beta is chosen afterwards, on validation only (eval_v2 / judge); test
predictions are written for every beta but read only for the chosen one.

Outputs in /kaggle/working:
    adapters/beta-<b>/            DPO adapter per beta
    loss_check.json               TRL vs hand-written loss on one batch
    log_history_beta-<b>.json     loss, reward margins and accuracies
    predictions_valid.jsonl       300 validation diffs, per beta
    predictions_test.jsonl        450 test diffs, per beta, with and without context
    run_info.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODE = "smoke"  # replaced by scripts/kaggle_job.py: "smoke" or "full"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU; see sft_v2.py

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
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=True)

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working")
DATA = next(INPUT.rglob("valid.meta.jsonl")).parent
PAIRS = next(INPUT.rglob("pairs.jsonl"))
SFT_ADAPTER = next(p.parent for p in INPUT.rglob("adapter_config.json") if p.parent.name == "final")
BETAS = (0.05, 0.1, 0.3)
N_VALID = 300
SEED = 0
started = time.time()

# prompt.py ships in the v2 dataset and dpo.py in the pairs dataset: a Kaggle
# script job uploads only the script itself.
sys.path[:0] = [str(DATA), str(PAIRS.parent)]
from dpo import dpo_loss, sequence_logps  # noqa: E402
from prompt import build_messages, clean_reply  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


tokenizer = AutoTokenizer.from_pretrained(BASE)
pairs = load(PAIRS)
valid_meta = load(DATA / "valid.meta.jsonl")[:N_VALID]
tests = load(DATA / "test.jsonl")
contexts = {c["sha"]: c["context"] for c in load(DATA / "test_context.jsonl")}
if MODE == "smoke":
    pairs, valid_meta, tests = pairs[:32], valid_meta[:8], tests[:8]
train_dataset = Dataset.from_list(
    [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
)


def fresh_policy() -> PeftModel:
    """SFT-v2 as a trainable adapter, plus a frozen copy named "ref" for TRL."""
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)
    model.load_adapter(SFT_ADAPTER, adapter_name="ref")
    model.set_adapter("default")
    return model


def make_trainer(model: PeftModel, beta: float) -> DPOTrainer:
    config = DPOConfig(
        output_dir=str(OUT / "trainer"),
        beta=beta,
        loss_type=["sigmoid"],
        max_length=1280,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=4 if MODE == "smoke" else -1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.1 * len(pairs) / 8)),  # 10% of one epoch
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10 if MODE == "full" else 1,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    return DPOTrainer(
        model=model, args=config, train_dataset=train_dataset, processing_class=tokenizer
    )


def check_loss(trainer: DPOTrainer, beta: float) -> dict:
    """TRL's loss and the hand-written one on the same batch, dropout off."""
    model = trainer.model
    model.eval()
    batch = trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))
    with torch.no_grad():
        trl_loss = float(trainer.compute_loss(model, batch))
        mask = batch["completion_mask"]
        model.set_adapter("default")
        policy = sequence_logps(
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits,
            batch["input_ids"],
            mask,
        )
        model.set_adapter("ref")
        reference = sequence_logps(
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits,
            batch["input_ids"],
            mask,
        )
        model.set_adapter("default")
    pc, pr = policy.chunk(2)
    rc, rr = reference.chunk(2)
    ours = float(dpo_loss(pc, pr, rc, rr, beta).loss)
    model.train()
    return {
        "beta": beta,
        "pairs_in_batch": int(pc.numel()),
        "trl_loss": trl_loss,
        "our_loss": ours,
        "abs_difference": abs(trl_loss - ours),
    }


@torch.no_grad()
def generate(model, prompts: list[list[dict]], batch: int = 32) -> list[str]:
    model.eval()
    tokenizer.padding_side = "left"
    replies = []
    for i in range(0, len(prompts), batch):
        texts = [
            tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in prompts[i : i + batch]
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        replies += tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
    tokenizer.padding_side = "right"
    return [clean_reply(r) for r in replies]


checks, timings = [], {}
valid_prompts = [
    build_messages(m["diff"], m["context"] if m["with_context"] else None) for m in valid_meta
]
with (
    (OUT / "predictions_valid.jsonl").open("w") as valid_fh,
    (OUT / "predictions_test.jsonl").open("w") as test_fh,
):
    for beta in BETAS:
        t0 = time.time()
        model = fresh_policy()
        trainer = make_trainer(model, beta)
        try:
            checks.append(check_loss(trainer, beta))
        except Exception as error:  # the check must not cost the run
            checks.append({"beta": beta, "error": repr(error)})
        print("loss check:", checks[-1], flush=True)
        trainer.train()
        name = f"beta-{beta}"
        trainer.model.save_pretrained(OUT / "adapters" / name, selected_adapters=["default"])
        (OUT / f"log_history_{name}.json").write_text(
            json.dumps(trainer.state.log_history, indent=1)
        )
        trainer.model.set_adapter("default")
        for meta, pred in zip(valid_meta, generate(trainer.model, valid_prompts), strict=True):
            valid_fh.write(
                json.dumps(
                    {
                        "beta": beta,
                        "sha": meta["sha"],
                        "repo": meta["repo"],
                        "reference": meta["subject"],
                        "prediction": pred,
                    }
                )
                + "\n"
            )
        for use_context in (False, True):
            prompts = [
                build_messages(t["diff"], contexts[t["sha"]] if use_context else None)
                for t in tests
            ]
            for test, pred in zip(tests, generate(trainer.model, prompts), strict=True):
                test_fh.write(
                    json.dumps(
                        {
                            "beta": beta,
                            "sha": test["sha"],
                            "repo": test["repo"],
                            "with_context": use_context,
                            "prediction": pred,
                        }
                    )
                    + "\n"
                )
        timings[name] = round(time.time() - t0)
        del trainer, model
        torch.cuda.empty_cache()

(OUT / "loss_check.json").write_text(json.dumps(checks, indent=2))
info = {
    "mode": MODE,
    "betas": BETAS,
    "pairs": len(pairs),
    "valid_for_selection": len(valid_meta),
    "gpu": torch.cuda.get_device_name(0),
    "versions": {
        p: __import__(p).__version__ for p in ("torch", "transformers", "trl", "peft", "datasets")
    },
    "seconds_per_beta": timings,
    "total_seconds": round(time.time() - started),
    "learning_rate": 2e-5,
    "effective_batch_pairs": 8,
    "seed": SEED,
}
(OUT / "run_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))

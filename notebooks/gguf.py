"""Convert the shipped model to GGUF (Q4_K_M) and check it with llama.cpp (Kaggle, CPU).

Pushed by scripts/kaggle_job.py. Runs llama.cpp's own tools at the release
tag below in Kaggle's sandbox: its converter (with the versions it pins) and
its prebuilt llama-quantize for Linux. Then runs the GGUF file with
llama-cpp-python, the runtime of the CPU demo, on the 450 test diffs.

SOURCE picks the model: "sft" uses the SFT-v2 job's merged/ folder; a DPO
adapter folder name (e.g. "beta-0.1") merges that adapter into the base first.

Outputs in /kaggle/working:
    commit-suggestor-q4_k_m.gguf    the model for llama.cpp
    commit-suggestor-q4_k_m.json    what it is (read by the CLI)
    predictions_test.jsonl          its greedy subjects on the test diffs, no context
    gguf_info.json                  versions, sizes, seconds per subject
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

MODE = "smoke"  # replaced by scripts/kaggle_job.py: "smoke" or "full"
SOURCE = "sft"
TAG = "b11701"  # llama.cpp release of 2026-10-16
INPUT, OUT, WORK = Path("/kaggle/input"), Path("/kaggle/working"), Path("/kaggle/tmp")
WORK.mkdir(parents=True, exist_ok=True)
started = time.time()


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


# 1. The model as a Hugging Face folder.
if SOURCE == "sft":
    merged = next(p.parent for p in INPUT.rglob("config.json") if p.parent.name == "merged")
else:
    run(sys.executable, "-m", "pip", "install", "-q", "peft==0.21.2")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter = next(p.parent for p in INPUT.rglob("adapter_config.json") if p.parent.name == SOURCE)
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", dtype=torch.float16
    )
    merged = WORK / "merged"
    PeftModel.from_pretrained(base, adapter).merge_and_unload().save_pretrained(merged)

# The converter pins transformers 4.57, which cannot read tokenizer files saved by
# transformers 5 (extra_special_tokens became a list). Fine-tuning does not change
# the tokenizer, so convert the fine-tuned weights with the base model's own files.
from huggingface_hub import hf_hub_download  # noqa: E402

model_dir = WORK / "model"
model_dir.mkdir(exist_ok=True)
for item in merged.iterdir():
    if item.suffix in (".safetensors", ".json") and not item.name.startswith("tokenizer"):
        (model_dir / item.name).unlink(missing_ok=True)
        (model_dir / item.name).symlink_to(item)
for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
    hf_hub_download("Qwen/Qwen2.5-Coder-1.5B-Instruct", name, local_dir=model_dir)
merged = model_dir

# 2. llama.cpp's converter and quantizer at a fixed release.
src = WORK / "llama.cpp"
run(
    "git",
    "clone",
    "-q",
    "--depth",
    "1",
    "--branch",
    TAG,
    "--filter=blob:none",
    "--sparse",
    "https://github.com/ggml-org/llama.cpp",
    str(src),
)
run("git", "-C", str(src), "sparse-checkout", "set", "conversion", "gguf-py", "requirements")
archive = WORK / "llama-bin.tar.gz"
urllib.request.urlretrieve(
    f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/llama-{TAG}-bin-ubuntu-x64.tar.gz",
    archive,
)
with tarfile.open(archive) as tar:
    tar.extractall(WORK / "bin", filter="data")
quantize = next((WORK / "bin").rglob("llama-quantize"))
quantize.chmod(0o755)

# The converter pins its own versions, so it gets its own environment. Kaggle's
# Python has no ensurepip, so `python -m venv` fails there; uv makes the venv.
venv = WORK / "convert-env"
run(sys.executable, "-m", "pip", "install", "-q", "uv")
run(sys.executable, "-m", "uv", "venv", "-q", "--python", sys.executable, str(venv))
py = str(venv / "bin" / "python")
run(
    sys.executable,
    "-m",
    "uv",
    "pip",
    "install",
    "-q",
    "--python",
    py,
    "--extra-index-url",
    "https://download.pytorch.org/whl/cpu",
    "torch==2.11.0",
    "numpy~=2.2.6",
    "sentencepiece>=0.1.98,<0.3.0",
    "transformers==4.57.6",
    "protobuf>=4.21.0,<5.0.0",
    "-e",
    str(src / "gguf-py"),
)

f16 = WORK / "model-f16.gguf"
gguf = OUT / "commit-suggestor-q4_k_m.gguf"
run(py, str(src / "convert_hf_to_gguf.py"), str(merged), "--outtype", "f16", "--outfile", str(f16))
env = {
    **os.environ,
    "LD_LIBRARY_PATH": f"{quantize.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}",
}
subprocess.run([str(quantize), str(f16), str(gguf), "Q4_K_M"], check=True, env=env)
(OUT / "commit-suggestor-q4_k_m.json").write_text(
    json.dumps(
        {
            "name": "sft-v2" if SOURCE == "sft" else f"dpo-{SOURCE}",
            "prompt_version": 2,
            "repo_context": False,
            "quantization": "Q4_K_M",
            "llama_cpp": TAG,
        },
        indent=2,
    )
)

# 3. Check it with the runtime of the CPU demo, on the test diffs.
run(
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "llama-cpp-python==0.3.36",
    "--extra-index-url",
    "https://abetlen.github.io/llama-cpp-python/whl/cpu",
)
from llama_cpp import Llama  # noqa: E402

DATA = next(INPUT.rglob("test.jsonl")).parent
sys.path.insert(0, str(DATA))
from prompt import build_messages, clean_reply  # noqa: E402

llm = Llama(model_path=str(gguf), n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
tests = [json.loads(line) for line in (DATA / "test.jsonl").open()]
if MODE == "smoke":
    tests = tests[:5]
t0 = time.time()
with (OUT / "predictions_test.jsonl").open("w") as fh:
    for i, t in enumerate(tests, 1):
        out = llm.create_chat_completion(
            messages=build_messages(t["diff"]), max_tokens=40, temperature=0.0
        )
        reply = clean_reply(out["choices"][0]["message"]["content"] or "")
        fh.write(json.dumps({"sha": t["sha"], "repo": t["repo"], "prediction": reply}) + "\n")
        if i % 50 == 0:
            print(f"{i}/{len(tests)}", flush=True)
seconds = (time.time() - t0) / len(tests)

info = {
    "mode": MODE,
    "source": SOURCE,
    "llama_cpp": TAG,
    "gguf_bytes": gguf.stat().st_size,
    "test_examples": len(tests),
    "seconds_per_subject_cpu": round(seconds, 2),
    "cpu_count": os.cpu_count(),
    "total_seconds": round(time.time() - started),
}
(OUT / "gguf_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))

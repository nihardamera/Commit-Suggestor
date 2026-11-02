"""Merge a PEFT LoRA adapter into the base model's weights, with MLX only.

For every adapted linear layer, W' = W + (lora_alpha / r) * B @ A, which is
what peft's merge_and_unload computes (no DoRA, no rsLoRA: checked below).
The result is a Hugging Face folder that mlx_lm convert and the Hub take.

    uv run --extra mlx python scripts/merge_lora.py \
        --adapter data/kaggle-output/dpo/full/adapters/beta-0.3 --out data/models/dpo-merged
    uv run --extra mlx mlx_lm convert --hf-path data/models/dpo-merged \
        --mlx-path data/models/dpo-mlx -q --q-bits 4 --q-group-size 64
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

PREFIX = "base_model.model."


def merge(base: dict[str, mx.array], adapter: dict[str, mx.array], scale: float) -> int:
    """Add each LoRA product to its base weight in place; returns how many were merged."""
    merged = 0
    for key in adapter:
        if not key.endswith("lora_A.weight"):
            continue
        name = key[len(PREFIX) :].removesuffix(".lora_A.weight") + ".weight"
        a, b = adapter[key], adapter[key.replace("lora_A", "lora_B")]
        w = base[name]
        delta = (b.astype(mx.float32) @ a.astype(mx.float32)) * scale
        base[name] = (w.astype(mx.float32) + delta).astype(w.dtype)
        merged += 1
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.adapter / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA" and not config["use_dora"] and not config["use_rslora"]
    scale = config["lora_alpha"] / config["r"]
    base_dir = Path(snapshot_download(config["base_model_name_or_path"]))
    weights: dict[str, mx.array] = {}
    for f in sorted(base_dir.glob("*.safetensors")):
        weights.update(mx.load(str(f)))  # type: ignore[arg-type]
    adapter = mx.load(str(args.adapter / "adapter_model.safetensors"))
    assert isinstance(adapter, dict)
    expected = len(config["layers_to_transform"]) * len(config["target_modules"])
    n = merge(weights, adapter, scale)
    assert n == expected, f"merged {n} layers, expected {expected}"

    args.out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out / "model.safetensors"), weights, metadata={"format": "pt"})
    for item in base_dir.iterdir():
        if item.suffix in (".json", ".txt") and item.name != "model.safetensors.index.json":
            shutil.copy(item, args.out / item.name)
    print(f"merged {n} layers (scale {scale:g}) into {args.out}")


if __name__ == "__main__":
    main()

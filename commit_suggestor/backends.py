"""Run the model: MLX on Apple Silicon, llama.cpp (a GGUF file) everywhere else.

Both take the chat messages from commit_suggestor.prompt and return the raw
reply, decoded greedily, so the CLI, the git hook, the Gradio demo and the
evaluation all ask the model the same way.

    backend = load_backend(Path("~/.cache/commit-suggestor/sft-v2-mlx"))
    backend.generate(build_messages(diff))
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Protocol

MAX_NEW_TOKENS = 40


class Backend(Protocol):
    name: str

    def generate(self, messages: list[dict[str, str]]) -> str: ...


class MLXBackend:
    """An MLX model folder, optionally with a LoRA adapter folder (v1)."""

    name = "mlx"

    def __init__(self, model: str, adapter: Path | None = None) -> None:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        # load() is typed as returning two or three items; it returns two here.
        self.model, self.tokenizer = load(model, adapter_path=str(adapter) if adapter else None)[:2]
        self.sampler = make_sampler(temp=0.0)

    def generate(self, messages: list[dict[str, str]]) -> str:
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return str(
            generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=MAX_NEW_TOKENS,
                sampler=self.sampler,
                verbose=False,
            )
        )


class LlamaCppBackend:
    """A GGUF file run by llama.cpp; the chat template comes from the file."""

    name = "llama.cpp"

    def __init__(self, gguf: Path, threads: int | None = None) -> None:
        from llama_cpp import Llama

        self.llm = Llama(model_path=str(gguf), n_ctx=2048, n_threads=threads, verbose=False)

    def generate(self, messages: list[dict[str, str]]) -> str:
        out = self.llm.create_chat_completion(
            messages=messages, max_tokens=MAX_NEW_TOKENS, temperature=0.0
        )
        return str(out["choices"][0]["message"]["content"] or "")


def mlx_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


def load_backend(model: Path, adapter: Path | None = None) -> Backend:
    """A GGUF file goes to llama.cpp; a model folder to MLX (Apple Silicon only)."""
    if model.suffix == ".gguf":
        return LlamaCppBackend(model)
    if not mlx_available():
        sys.exit(
            f"{model} is an MLX model, which needs an Apple Silicon Mac; "
            "use the GGUF build on this machine"
        )
    return MLXBackend(str(model), adapter)

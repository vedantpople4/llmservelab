"""Prompt construction: exact token counts drawn from a real corpus (ADR-005).

Prompts are token-ID arrays, not text, so the length the config asks for is the length the server
sees. Every prompt starts with a unique nonce block drawn from the corpus, so two requests can
never share a prefix and a prefix cache (where one exists) cannot make TTFT lie.

The tokenizer is loaded once and cached; the bundled corpus is tokenized once per process.
Everything is seed-driven: the same `np.random.Generator` state produces the same prompt.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import tokenizers

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.txt"
TOKENIZER_URL = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer.json"
DEFAULT_CACHE_DIR = Path(os.environ.get("LLMSERVE_CACHE", Path.home() / ".cache" / "llmserve"))
NONCE_TOKENS = 16


@runtime_checkable
class Tokenizer(Protocol):
    """What `PromptBuilder` needs from a tokenizer. Tests inject a stub."""

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class QwenTokenizer:
    """`tokenizers.Tokenizer` behind the `Tokenizer` protocol, with no special tokens."""

    def __init__(self, raw: tokenizers.Tokenizer) -> None:
        self._raw = raw

    @property
    def vocab_size(self) -> int:
        size: int = self._raw.get_vocab_size(with_added_tokens=False)
        return size

    def encode(self, text: str) -> list[int]:
        ids: list[int] = self._raw.encode(text, add_special_tokens=False).ids
        return ids

    def decode(self, ids: list[int]) -> str:
        text: str = self._raw.decode(list(ids))
        return text


def load_tokenizer(cache_dir: Path | None = None) -> QwenTokenizer:
    """Load the pinned Qwen tokenizer, downloading it once into the cache directory."""
    cache = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    path = cache / "tokenizer.json"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        partial = cache / "tokenizer.json.part"
        try:
            with urllib.request.urlopen(TOKENIZER_URL, timeout=60) as resp, partial.open("wb") as f:
                shutil.copyfileobj(resp, f)
            partial.replace(path)
        except OSError as e:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"could not download the tokenizer from {TOKENIZER_URL} ({e}). Place "
                f"tokenizer.json at {path} manually, or set LLMSERVE_CACHE to a directory "
                "that already contains it."
            ) from e
    return QwenTokenizer(tokenizers.Tokenizer.from_file(str(path)))


@lru_cache(maxsize=1)
def load_corpus_tokens() -> npt.NDArray[np.int64]:
    """Tokenize the bundled corpus once per process."""
    tokenizer = load_tokenizer()
    text = CORPUS_PATH.read_text(encoding="utf-8")
    return np.asarray(tokenizer.encode(text), dtype=np.int64)


class PromptBuilder:
    """Builds prompts of an exact token length from a tokenized corpus."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        corpus_tokens: npt.NDArray[np.int64],
        *,
        nonce_tokens: int = NONCE_TOKENS,
    ) -> None:
        if corpus_tokens.ndim != 1 or len(corpus_tokens) == 0:
            raise ValueError("corpus_tokens must be a non-empty 1-D array of token ids")
        self.tokenizer = tokenizer
        self.corpus_tokens = corpus_tokens
        self.nonce_tokens = nonce_tokens

    @classmethod
    def default(cls) -> PromptBuilder:
        """The bundled corpus plus the pinned Qwen tokenizer (downloaded on first use)."""
        return cls(load_tokenizer(), load_corpus_tokens())

    def build(self, n_tokens: int, rng: np.random.Generator) -> list[int]:
        """`n_tokens` ids: a unique nonce block, then a corpus slice at a random offset."""
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")
        if n_tokens > len(self.corpus_tokens):
            raise ValueError(
                f"prompt of {n_tokens} tokens exceeds the corpus ({len(self.corpus_tokens)} tokens)"
            )
        nonce_len = min(self.nonce_tokens, n_tokens)
        nonce = [
            int(self.corpus_tokens[i]) for i in rng.integers(0, len(self.corpus_tokens), nonce_len)
        ]
        remaining = n_tokens - nonce_len
        if remaining == 0:
            return nonce
        last_start = len(self.corpus_tokens) - remaining
        start = int(rng.integers(0, last_start + 1))
        body = [int(t) for t in self.corpus_tokens[start : start + remaining]]
        return nonce + body

    def text(self, ids: list[int]) -> str:
        """Decode ids back to text, for backends that cannot take token IDs (ADR-009)."""
        return self.tokenizer.decode(ids)


def build_prompt(
    n_tokens: int,
    rng: np.random.Generator,
    *,
    builder: PromptBuilder | None = None,
) -> list[int]:
    """`build_prompt(n_tokens, rng) -> list[int]` from PLAN.md, with a cached default builder."""
    return (builder or PromptBuilder.default()).build(n_tokens, rng)

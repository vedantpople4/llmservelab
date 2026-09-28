import numpy as np
import pytest

from llmserve.workload.prompts import PromptBuilder, build_prompt


class StubTokenizer:
    """Character-level stand-in: reversible, deterministic, and needs no download."""

    vocab_size = 1000

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 997 + 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr((i - 1) + 32) for i in ids)


@pytest.fixture()
def builder() -> PromptBuilder:
    rng = np.random.default_rng(7)
    corpus = rng.integers(1, 1000, size=20_000).astype(np.int64)
    return PromptBuilder(StubTokenizer(), corpus)


@pytest.mark.parametrize("n", [1, 15, 16, 17, 128, 4096])
def test_build_gives_exact_length(builder: PromptBuilder, n: int) -> None:
    ids = builder.build(n, np.random.default_rng(0))
    assert len(ids) == n


def test_same_seed_gives_the_same_prompt(builder: PromptBuilder) -> None:
    assert builder.build(64, np.random.default_rng(1)) == builder.build(
        64, np.random.default_rng(1)
    )


def test_different_requests_get_different_prompts(builder: PromptBuilder) -> None:
    rng = np.random.default_rng(2)
    prompts = {tuple(builder.build(64, rng)) for _ in range(50)}
    assert len(prompts) == 50


def test_nonce_block_differs_between_requests(builder: PromptBuilder) -> None:
    rng = np.random.default_rng(3)
    nonces = {tuple(builder.build(64, rng)[:16]) for _ in range(50)}
    assert len(nonces) == 50


def test_nonce_is_drawn_from_the_corpus(builder: PromptBuilder) -> None:
    ids = builder.build(32, np.random.default_rng(4))
    corpus = set(int(t) for t in builder.corpus_tokens)
    assert set(ids[:16]) <= corpus


def test_prompt_exceeding_the_corpus_is_rejected(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="exceeds the corpus"):
        builder.build(len(builder.corpus_tokens) + 1, np.random.default_rng(0))


def test_zero_or_negative_length_is_rejected(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        builder.build(0, np.random.default_rng(0))


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PromptBuilder(StubTokenizer(), np.array([], dtype=np.int64))


def test_build_prompt_uses_the_given_builder(builder: PromptBuilder) -> None:
    assert len(build_prompt(48, np.random.default_rng(5), builder=builder)) == 48


def test_default_builder_produces_exact_lengths() -> None:
    try:
        default = PromptBuilder.default()
    except (RuntimeError, OSError) as e:  # offline CI: tokenizer.json cannot be fetched
        pytest.skip(f"tokenizer unavailable: {e}")
    rng = np.random.default_rng(11)
    for n in (1, 17, 128, 4096):
        ids = default.build(n, rng)
        assert len(ids) == n
        assert all(0 <= i for i in ids)

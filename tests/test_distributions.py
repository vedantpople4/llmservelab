from pathlib import Path

import numpy as np
import pytest

from llmserve.config.schema import Choice, Empirical, Fixed, LogNormal, Uniform
from llmserve.workload.distributions import sample


def test_fixed_is_constant() -> None:
    draws = sample(Fixed(distribution="fixed", tokens=128), np.random.default_rng(0), 100)
    assert draws.shape == (100,)
    assert np.all(draws == 128)


def test_uniform_respects_bounds() -> None:
    dist = Uniform(distribution="uniform", min=16, max=64)
    draws = sample(dist, np.random.default_rng(1), 5000)
    assert draws.min() >= 16
    assert draws.max() <= 64
    assert np.allclose(np.bincount(draws - 16) > 0, True)


def test_choice_respects_values_and_weights() -> None:
    dist = Choice(distribution="choice", values=[16, 128], weights=[9, 1])
    draws = sample(dist, np.random.default_rng(2), 10_000)
    assert set(np.unique(draws)) <= {16, 128}
    share = float(np.mean(draws == 16))
    assert 0.87 < share < 0.93  # ~90% within a few sigma at n=10k


def test_choice_without_weights_is_uniform() -> None:
    dist = Choice(distribution="choice", values=[8, 16, 32])
    draws = sample(dist, np.random.default_rng(3), 3000)
    counts = {int(v): int(np.sum(draws == v)) for v in (8, 16, 32)}
    assert max(counts.values()) - min(counts.values()) < 300


def test_lognormal_is_clipped_to_bounds() -> None:
    dist = LogNormal(distribution="lognormal", median=256.0, sigma=1.5, min=32, max=2048)
    draws = sample(dist, np.random.default_rng(4), 5000)
    assert draws.min() >= 32
    assert draws.max() <= 2048
    assert np.median(draws) > 100  # the mass sits near the median, not the floor


def test_empirical_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "lengths.txt"
    path.write_text("16\n32\n64\n")
    dist = Empirical(distribution="empirical", path=str(path))
    draws = sample(dist, np.random.default_rng(5), 200)
    assert set(int(d) for d in draws) <= {16, 32, 64}


def test_empirical_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("")
    dist = Empirical(distribution="empirical", path=str(path))
    with pytest.raises(ValueError, match="empty"):
        sample(dist, np.random.default_rng(0), 1)


def test_same_seed_same_draws() -> None:
    dist = LogNormal(distribution="lognormal", median=128.0, sigma=0.8, min=8, max=512)
    a = sample(dist, np.random.default_rng(9), 100)
    b = sample(dist, np.random.default_rng(9), 100)
    assert np.array_equal(a, b)


def test_zero_draws_is_allowed() -> None:
    assert sample(Fixed(distribution="fixed", tokens=1), np.random.default_rng(0), 0).size == 0

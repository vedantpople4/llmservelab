"""Interval estimates: t-CI coverage on known-normal data and a seeded, reproducible bootstrap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from llmserve.analysis.stats import bootstrap_percentile_diff, mean_ci


def test_t_interval_covers_the_true_mean_about_95_percent_of_the_time() -> None:
    rng = np.random.default_rng(7)
    trials = 4000
    hits = 0
    for _ in range(trials):
        est = mean_ci(rng.normal(10.0, 3.0, size=5))
        hits += est.lo <= 10.0 <= est.hi
    # Binomial sd at p = 0.95 over 4000 trials is ~0.0034; a z-interval (no t) would sit near 0.88.
    assert 0.935 <= hits / trials <= 0.965


def test_mean_ci_matches_the_textbook_formula() -> None:
    est = mean_ci([1.0, 2.0, 3.0, 4.0])
    # t(0.975, 3) = 3.1824463; s = 1.2909944; half = t·s/2
    assert est.mean == 2.5 and est.n == 4
    assert math.isclose(est.hi - est.mean, 3.1824463 * 1.2909944 / 2, rel_tol=1e-6)


def test_mean_ci_degenerate_inputs() -> None:
    one = mean_ci([4.0])
    assert one.mean == 4.0 and one.n == 1 and math.isnan(one.lo) and math.isnan(one.hi)
    none = mean_ci([])
    assert none.n == 0 and math.isnan(none.mean)
    assert mean_ci([1.0, float("nan"), 3.0]).n == 2


def test_bootstrap_is_reproducible_for_a_seed_and_varies_across_seeds() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.exponential(1.0, 300), rng.exponential(1.0, 300)
    first = bootstrap_percentile_diff(a, b, 95, resamples=2000, seed=3)
    again = bootstrap_percentile_diff(a, b, 95, resamples=2000, seed=3)
    other = bootstrap_percentile_diff(a, b, 95, resamples=2000, seed=4)
    assert first == again
    assert (first.lo, first.hi) != (other.lo, other.hi)
    assert first.diff == other.diff == float(np.percentile(a, 95) - np.percentile(b, 95))


def test_bootstrap_separates_a_real_shift_and_brackets_no_shift() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(1.0, 0.1, 2000)
    shifted = bootstrap_percentile_diff(base + 0.2, rng.normal(1.0, 0.1, 2000), 95)
    assert shifted.lo > 0.1 and shifted.hi < 0.3
    same = bootstrap_percentile_diff(base, base.copy(), 95)
    assert same.diff == 0 and same.lo < 0 < same.hi


def test_bootstrap_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError):
        bootstrap_percentile_diff([], [1.0], 95)

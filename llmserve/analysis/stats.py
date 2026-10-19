"""The two interval estimates the Phase 3 report uses (plan §5).

- **Across-rep mean:** Student-t 95% CI over the per-rep values, `mean ± t(0.975, n−1)·s/√n`.
  With one rep the bounds are NaN: one value says nothing about spread.
- **Difference of percentiles:** percentile bootstrap of `pct_q(a) − pct_q(b)`, resampling
  each sample independently with replacement. `B = 10_000` resamples and `seed = 0` by default,
  so the same inputs always give the same interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as sps

CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
_CHUNK = 500


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int


@dataclass(frozen=True)
class DiffCI:
    """`pct_q(a) − pct_q(b)` with its bootstrap interval."""

    q: float
    a: float
    b: float
    diff: float
    lo: float
    hi: float
    n_a: int
    n_b: int
    resamples: int
    seed: int


def mean_ci(values: npt.ArrayLike, confidence: float = CONFIDENCE) -> Estimate:
    x = _finite(values)
    n = len(x)
    if n == 0:
        return Estimate(math.nan, math.nan, math.nan, 0)
    mean = float(x.mean())
    if n == 1:
        return Estimate(mean, math.nan, math.nan, 1)
    half = float(sps.t.ppf(0.5 + confidence / 2, n - 1)) * float(x.std(ddof=1)) / math.sqrt(n)
    return Estimate(mean, mean - half, mean + half, n)


def bootstrap_percentile_diff(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    q: float,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> DiffCI:
    """Bootstrap CI of `percentile(a, q) − percentile(b, q)`; `q` in [0, 100]."""
    xa, xb = _finite(a), _finite(b)
    if len(xa) == 0 or len(xb) == 0:
        raise ValueError("bootstrap_percentile_diff needs two non-empty samples")
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for start in range(0, resamples, _CHUNK):
        k = min(_CHUNK, resamples - start)
        ra = np.percentile(rng.choice(xa, size=(k, len(xa))), q, axis=1)
        rb = np.percentile(rng.choice(xb, size=(k, len(xb))), q, axis=1)
        diffs[start : start + k] = ra - rb
    tail = (1 - confidence) / 2 * 100
    pa, pb = float(np.percentile(xa, q)), float(np.percentile(xb, q))
    return DiffCI(
        q=q,
        a=pa,
        b=pb,
        diff=pa - pb,
        lo=float(np.percentile(diffs, tail)),
        hi=float(np.percentile(diffs, 100 - tail)),
        n_a=len(xa),
        n_b=len(xb),
        resamples=resamples,
        seed=seed,
    )


def _finite(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64).ravel()
    kept: npt.NDArray[np.float64] = x[np.isfinite(x)]
    return kept

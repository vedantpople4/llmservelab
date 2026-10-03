"""Length distributions: sample token counts from a validated config (PRD §9).

Each config `LengthDistribution` maps to one draw here, so a workload file is reproducible from
its seed alone. Bounds are enforced by construction: samples are integers inside the configured
range, which is what the request specs and the KV-capacity model assume.

Arrival processes live in `workload/arrivals.py` (Phase 2); this module only answers "how long".
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from llmserve.config.schema import Choice, Empirical, Fixed, LengthDistribution, LogNormal, Uniform


@lru_cache(maxsize=8)
def _load_empirical(path: str) -> npt.NDArray[np.int64]:
    if not Path(path).read_text().strip():
        raise ValueError(f"{path}: empirical distribution file is empty")
    values = np.atleast_1d(np.loadtxt(path))
    if values.size == 0:
        raise ValueError(f"{path}: empirical distribution file is empty")
    return np.round(values).astype(np.int64)


def sample(dist: LengthDistribution, rng: Generator, n: int) -> npt.NDArray[np.int64]:
    """Draw `n` integer token counts from `dist`. Same rng state, same draw."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if isinstance(dist, Fixed):
        return np.full(n, dist.tokens, dtype=np.int64)
    if isinstance(dist, Uniform):
        return rng.integers(dist.min, dist.max + 1, size=n).astype(np.int64)
    if isinstance(dist, Choice):
        weights = dist.weights
        p = None if weights is None else np.asarray(weights, dtype=np.float64) / float(sum(weights))
        picks = rng.choice(np.asarray(dist.values, dtype=np.int64), size=n, p=p)
        return picks.astype(np.int64)
    if isinstance(dist, LogNormal):
        draws = rng.lognormal(mean=float(np.log(dist.median)), sigma=dist.sigma, size=n)
        return np.clip(np.round(draws), dist.min, dist.max).astype(np.int64)
    if isinstance(dist, Empirical):
        pool = _load_empirical(dist.path)
        return rng.choice(pool, size=n).astype(np.int64)
    raise TypeError(f"unsupported distribution: {type(dist).__name__}")

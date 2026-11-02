"""Output-length estimators (plan Phase 6 item 5, ADR-0014): the `output_tokens_est` a scheduler
sees, fixed when a request enqueues.

- **oracle** is the true requested length, the upper bound of the ablation.
- **none** is no estimate.
- **class_prior** is the per-class median output length of earlier runs, read from a prior file
  that `llmserve report` writes as `output_prior.json`. A class missing from the prior gets no
  estimate.
- **online** starts from the prior when one is given and moves each class's estimate toward
  every completed request's `usage` output tokens by an EWMA with weight `ONLINE_ALPHA`. A class
  with no prior and no completion yet gets no estimate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol

from llmserve.metrics.records import RequestRecord
from llmserve.workload.spec import RequestSpec

EstimatorName = Literal["oracle", "class_prior", "online", "none"]
ONLINE_ALPHA = 0.2


class OutputEstimator(Protocol):
    def estimate(self, spec: RequestSpec) -> int | None: ...

    def observe(self, spec: RequestSpec, record: RequestRecord) -> None:
        """Called with every ok completion."""


class Oracle:
    def estimate(self, spec: RequestSpec) -> int | None:
        return spec.output_tokens

    def observe(self, spec: RequestSpec, record: RequestRecord) -> None:
        pass


class NoEstimate:
    def estimate(self, spec: RequestSpec) -> int | None:
        return None

    def observe(self, spec: RequestSpec, record: RequestRecord) -> None:
        pass


class ClassPrior:
    def __init__(self, prior: dict[str, int]) -> None:
        self.prior = prior

    def estimate(self, spec: RequestSpec) -> int | None:
        return self.prior.get(spec.workload_class)

    def observe(self, spec: RequestSpec, record: RequestRecord) -> None:
        pass


class Online:
    def __init__(self, prior: dict[str, int] | None = None, alpha: float = ONLINE_ALPHA) -> None:
        self.alpha = alpha
        self.mean: dict[str, float] = {k: float(v) for k, v in (prior or {}).items()}

    def estimate(self, spec: RequestSpec) -> int | None:
        mean = self.mean.get(spec.workload_class)
        return None if mean is None else max(1, round(mean))

    def observe(self, spec: RequestSpec, record: RequestRecord) -> None:
        tokens = record.output_tokens_usage
        if tokens is None:
            return
        mean = self.mean.get(spec.workload_class)
        self.mean[spec.workload_class] = (
            float(tokens) if mean is None else mean + self.alpha * (tokens - mean)
        )


def load_prior(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    return {str(k): int(v) for k, v in data.items()}


def create(name: EstimatorName, prior: Path | None = None) -> OutputEstimator:
    loaded = load_prior(prior) if prior is not None else None
    if name == "oracle":
        return Oracle()
    if name == "none":
        return NoEstimate()
    if name == "class_prior":
        if loaded is None:
            raise ValueError("output_estimator class_prior needs gateway.output_prior")
        return ClassPrior(loaded)
    return Online(loaded)

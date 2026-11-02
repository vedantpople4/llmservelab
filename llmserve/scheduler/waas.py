"""WAAS v1, the Workload-Aware Adaptive Scheduler (plan Phase 6, ADR-0014).

**Order.** Admit the smallest `arrival_time + cost_weight · service_time`, where `service_time`
is the cost model's prefill plus decode estimate at decode batch `batch`, as SJF computes it. A
request that waits `cost_weight` seconds longer is worth one second less of predicted service:
`cost_weight = 0` is FIFO and a large weight approaches SJF. The key never changes while a
request waits, so it is computed once per request.

**Bypass bound.** Each waiting request counts the dispatches of later arrivals. When the oldest
waiting request has been bypassed `max_bypass` times, it is admitted next. Every later arrival
dispatched ahead of a younger waiting request was also dispatched ahead of the oldest one, so
the oldest always has the largest count, and bounding it bounds every request.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llmserve.scheduler.base import Request, Scheduler, SystemState
from llmserve.scheduler.estimators import CostModel
from llmserve.scheduler.shortest import DEFAULT_COST_MODEL


class WAAS(Scheduler):
    name = "waas"

    def __init__(
        self,
        cost_model: str = DEFAULT_COST_MODEL,
        batch: float = 16,
        cost_weight: float = 1.0,
        max_bypass: int = 32,
    ) -> None:
        if cost_weight < 0:
            raise ValueError(f"cost_weight must be >= 0, got {cost_weight}")
        if max_bypass < 1:
            raise ValueError(f"max_bypass must be >= 1, got {max_bypass}")
        self.model = CostModel.load(Path(cost_model))
        self.batch = batch
        self.cost_weight = cost_weight
        self.max_bypass = max_bypass
        self._keys: dict[str, tuple[float, float, str]] = {}
        self._bypassed: dict[str, int] = {}

    def key(self, r: Request) -> tuple[float, float, str]:
        k = self._keys.get(r.request_id)
        if k is None:
            service = self.model.service_time(r.prompt_tokens, r.output_tokens_est or 1, self.batch)
            k = self._keys[r.request_id] = (
                r.arrival_time + self.cost_weight * service,
                r.arrival_time,
                r.request_id,
            )
        return k

    def select_next(self, waiting: Sequence[Request], state: SystemState) -> Request | None:
        if not waiting:
            return None
        oldest = min(waiting, key=_age)
        if self._bypassed.get(oldest.request_id, 0) >= self.max_bypass:
            choice = oldest
        else:
            choice = min(waiting, key=self.key)
        if choice is not oldest:
            line = _age(choice)
            for r in waiting:
                if _age(r) < line:
                    self._bypassed[r.request_id] = self._bypassed.get(r.request_id, 0) + 1
        return choice

    def on_dispatch(self, r: Request) -> tuple[float | None, int | None]:
        score = self.key(r)[0]
        self._keys.pop(r.request_id, None)
        return score, self._bypassed.pop(r.request_id, 0)


def _age(r: Request) -> tuple[float, str]:
    return r.arrival_time, r.request_id

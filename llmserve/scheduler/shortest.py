"""Shortest-first policies (plan Phase 5 item 1). Both starve long requests under sustained load;
Phase 5 measures by how much.

- **ShortestPromptFirst** admits the shortest prompt: the request whose prefill finishes first.
- **ShortestJobFirst** admits the smallest predicted service time, prefill plus decode from the
  fitted cost model, using the gateway's output estimate. With no estimate
  (`output_estimator: none`) the decode term is zero and SJF orders by predicted prefill, which
  is SPF's order.
"""

from __future__ import annotations

from pathlib import Path

from llmserve.scheduler.base import Request, StaticKeyScheduler
from llmserve.scheduler.estimators import CostModel

DEFAULT_COST_MODEL = "configs/cost_models/m5_qwen7b_awq.json"


class ShortestPromptFirst(StaticKeyScheduler):
    name = "spf"

    def key(self, r: Request) -> float:
        return r.prompt_tokens


class ShortestJobFirst(StaticKeyScheduler):
    """`cost_model` is a `CostModel` JSON, relative to the working directory; `batch` is the
    decode batch the service time assumes, fixed so the key stays static while a request
    waits."""

    name = "sjf"

    def __init__(self, cost_model: str = DEFAULT_COST_MODEL, batch: float = 16) -> None:
        super().__init__()
        self.model = CostModel.load(Path(cost_model))
        self.batch = batch

    def key(self, r: Request) -> float:
        return self.model.service_time(r.prompt_tokens, r.output_tokens_est or 1, self.batch)

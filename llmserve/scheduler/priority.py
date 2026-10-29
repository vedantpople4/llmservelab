"""Strict class priority, FIFO within a class, with optional aging (plan Phase 5 item 2).

The most urgent class (lowest `priority`, as in vLLM) goes first. With `aging_s > 0` a request
gains one class level for every `aging_s` seconds it has waited, so a LOW request that has
waited `2 * aging_s` ties a HIGH request that just arrived and, having arrived earlier, wins.
Aging bounds the wait of a low class; strict priority (`aging_s = 0`) does not.
"""

from __future__ import annotations

from collections.abc import Sequence

from llmserve.scheduler.base import Request, Scheduler, SystemState


class PriorityScheduler(Scheduler):
    name = "priority"

    def __init__(self, aging_s: float = 0.0) -> None:
        if aging_s < 0:
            raise ValueError(f"aging_s must be >= 0, got {aging_s}")
        self.aging_s = aging_s

    def select_next(self, waiting: Sequence[Request], state: SystemState) -> Request | None:
        if not self.aging_s:
            return min(
                waiting, key=lambda r: (r.priority, r.arrival_time, r.request_id), default=None
            )
        now, aging = state.now, self.aging_s
        return min(
            waiting,
            key=lambda r: (
                r.priority - (now - r.arrival_time) / aging,
                r.arrival_time,
                r.request_id,
            ),
            default=None,
        )

from __future__ import annotations

from collections.abc import Sequence

from llmserve.scheduler.base import Request, Scheduler, SystemState


class FIFOScheduler(Scheduler):
    name = "fifo"

    def select_next(self, waiting: Sequence[Request], state: SystemState) -> Request | None:
        return min(waiting, key=lambda r: r.arrival_time, default=None)

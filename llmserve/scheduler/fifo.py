from __future__ import annotations

from llmserve.scheduler.base import Request, StaticKeyScheduler


class FIFOScheduler(StaticKeyScheduler):
    name = "fifo"

    def key(self, r: Request) -> float:
        return r.arrival_time

from __future__ import annotations

from typing import Any

from llmserve.scheduler.base import Scheduler
from llmserve.scheduler.fifo import FIFOScheduler

SCHEDULERS: dict[str, type[Scheduler]] = {
    FIFOScheduler.name: FIFOScheduler,
}


def create(name: str, params: dict[str, Any] | None = None) -> Scheduler:
    try:
        cls = SCHEDULERS[name]
    except KeyError:
        known = ", ".join(sorted(SCHEDULERS))
        raise ValueError(f"unknown scheduler {name!r} (known: {known})") from None
    return cls(**(params or {}))

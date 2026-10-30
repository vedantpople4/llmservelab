from __future__ import annotations

import inspect
from typing import Any

from llmserve.scheduler.base import Scheduler
from llmserve.scheduler.fifo import FIFOScheduler
from llmserve.scheduler.priority import PriorityScheduler
from llmserve.scheduler.shortest import ShortestJobFirst, ShortestPromptFirst

SCHEDULERS: dict[str, type[Scheduler]] = {
    cls.name: cls
    for cls in (FIFOScheduler, ShortestPromptFirst, ShortestJobFirst, PriorityScheduler)
}


def check(name: str, params: dict[str, Any]) -> None:
    """Raise ValueError unless `name` is known and accepts `params` by name; nothing is built,
    so a config validates without the files a scheduler loads."""
    cls = _lookup(name)
    try:
        inspect.signature(cls).bind(**params)
    except TypeError as e:
        raise ValueError(f"scheduler {name!r}: {e}") from None


def create(name: str, params: dict[str, Any] | None = None) -> Scheduler:
    return _lookup(name)(**(params or {}))


def _lookup(name: str) -> type[Scheduler]:
    try:
        return SCHEDULERS[name]
    except KeyError:
        known = ", ".join(sorted(SCHEDULERS))
        raise ValueError(f"unknown scheduler {name!r} (known: {known})") from None

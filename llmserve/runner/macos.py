"""macOS run guards: time spent asleep, keeping the Mac awake, and the power source.

All three are no-ops off Darwin (Linux CI, Kaggle): readings are None and nothing is spawned.
`is_darwin`, `_read_clocks`, `_pmset_batt` and `_spawn` are module-level so tests can substitute
them without a Mac.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Protocol

BATTERY_POWER = "Battery Power"
_POWER_SOURCE = re.compile(r"Now drawing from '([^']+)'")

SleepMark = tuple[int, int]  # (CLOCK_MONOTONIC ns, CLOCK_UPTIME_RAW ns)


class _Process(Protocol):
    def terminate(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


def is_darwin() -> bool:
    return sys.platform == "darwin"


def _read_clocks() -> SleepMark | None:
    # On Darwin MONOTONIC keeps counting while asleep and UPTIME_RAW stops, so the gap between
    # their deltas is time asleep.
    if sys.platform != "darwin":
        return None
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC), time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)


def sleep_mark() -> SleepMark | None:
    return _read_clocks()


def slept_s(since: SleepMark | None) -> float | None:
    """Seconds asleep since `since`; None when either reading is unavailable."""
    now = _read_clocks()
    if since is None or now is None:
        return None
    asleep_ns = (now[0] - since[0]) - (now[1] - since[1])
    return round(max(asleep_ns, 0) / 1e9, 3)  # the two reads jitter by microseconds either way


def parse_power_source(pmset_batt: str) -> str | None:
    """`'AC Power'` or `'Battery Power'` from `pmset -g batt` output; None if unrecognized."""
    match = _POWER_SOURCE.search(pmset_batt)
    return match.group(1) if match else None


def power_source() -> str | None:
    if not is_darwin():
        return None
    output = _pmset_batt()
    return None if output is None else parse_power_source(output)


def _pmset_batt() -> str | None:
    try:
        proc = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def keep_awake(progress: Callable[[str], None]) -> Callable[[], None]:
    """Hold `caffeinate -i` for this process's lifetime; returns the function that releases it.

    `-i` prevents idle sleep only; closing the lid still sleeps the Mac, which `slept_s` records.
    `-w <pid>` ends caffeinate with us even if release never runs.
    """
    if not is_darwin():
        return lambda: None
    try:
        proc = _spawn(["caffeinate", "-i", "-w", str(os.getpid())])
    except OSError as e:
        progress(f"caffeinate unavailable ({e}); the Mac may idle-sleep during the run")
        return lambda: None
    progress("caffeinate -i held for the run (a closed lid still sleeps the Mac)")

    def release() -> None:
        proc.terminate()
        proc.wait(timeout=5)

    return release


def _spawn(argv: list[str]) -> _Process:
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

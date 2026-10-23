"""macOS run guards with fake clocks, a fake `pmset` and a fake `caffeinate` spawner."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_env_check import make_cfg as env_cfg
from test_env_check import mock_client
from test_experiment import make_cfg, run_in_process

from llmserve.runner import experiment, macos
from llmserve.runner.env_check import check_server
from llmserve.runner.run import drive as real_drive

AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining\n"
BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=1)\t19%; discharging; 1:02 remaining present: true\n"
)
S = 1_000_000_000


class FakeCaffeinate:
    def __init__(self) -> None:
        self.argv: list[str] | None = None
        self.terminated = False
        self.waited = False

    def spawn(self, argv: list[str]) -> FakeCaffeinate:
        self.argv = argv
        return self

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0


def on_a_mac(monkeypatch: pytest.MonkeyPatch, *, pmset: str = AC) -> FakeCaffeinate:
    caffeinate = FakeCaffeinate()
    monkeypatch.setattr(macos, "is_darwin", lambda: True)
    monkeypatch.setattr(macos, "_pmset_batt", lambda: pmset)
    monkeypatch.setattr(macos, "_spawn", caffeinate.spawn)
    return caffeinate


def test_slept_time_is_monotonic_advance_beyond_uptime(monkeypatch: pytest.MonkeyPatch) -> None:
    readings = iter([(100 * S, 50 * S), (3800 * S, 138 * S)])  # 3700 s passed, 88 s awake
    monkeypatch.setattr(macos, "_read_clocks", lambda: next(readings))
    assert macos.slept_s(macos.sleep_mark()) == 3612.0


def test_awake_interval_reports_zero_even_with_read_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter([(100 * S, 50 * S + 16_000), (160 * S, 110 * S + 32_000)])
    monkeypatch.setattr(macos, "_read_clocks", lambda: next(readings))
    assert macos.slept_s(macos.sleep_mark()) == 0.0


def test_sleep_is_unknown_without_the_clocks() -> None:
    assert macos.sleep_mark() is None
    assert macos.slept_s(None) is None


def test_power_source_is_parsed_from_pmset() -> None:
    assert macos.parse_power_source(AC) == "AC Power"
    assert macos.parse_power_source(BATTERY) == macos.BATTERY_POWER
    assert macos.parse_power_source("pmset: command failed") is None


def test_power_source_is_unknown_off_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos, "_pmset_batt", lambda: BATTERY)
    assert macos.power_source() is None


def test_caffeinate_is_held_for_exactly_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    caffeinate = on_a_mac(monkeypatch)
    held_during_reps: list[bool] = []

    async def drive(*args: Any, **kwargs: Any) -> Any:
        held_during_reps.append(caffeinate.argv is not None and not caffeinate.terminated)
        return await real_drive(*args, **kwargs)

    monkeypatch.setattr(experiment, "drive", drive)
    run_in_process(make_cfg(), tmp_path, allow_dirty=True)
    assert caffeinate.argv == ["caffeinate", "-i", "-w", str(os.getpid())]
    assert held_during_reps == [True, True, True]  # warm-up and both reps
    assert caffeinate.terminated and caffeinate.waited


def test_caffeinate_is_released_when_the_run_crashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    caffeinate = on_a_mac(monkeypatch)

    async def drive(*args: Any, **kwargs: Any) -> Any:
        raise httpx.ConnectError("server went away")

    monkeypatch.setattr(experiment, "drive", drive)
    with pytest.raises(httpx.ConnectError):
        run_in_process(make_cfg(), tmp_path, allow_dirty=True)
    assert caffeinate.terminated


def test_missing_caffeinate_is_noted_and_the_run_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    on_a_mac(monkeypatch)

    def missing(argv: list[str]) -> Any:
        raise FileNotFoundError(2, "No such file or directory", "caffeinate")

    monkeypatch.setattr(macos, "_spawn", missing)
    lines: list[str] = []
    run_dir, _ = run_in_process(make_cfg(), tmp_path, allow_dirty=True, progress=lines.append)
    assert any(line.startswith("caffeinate unavailable") for line in lines)
    assert (run_dir / "metadata.json").is_file()


def test_metadata_attributes_sleep_to_the_rep_it_happened_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    on_a_mac(monkeypatch)
    ticks = 0
    asleep_ns = 0

    def read_clocks() -> tuple[int, int]:
        nonlocal ticks
        ticks += 1
        return ticks * S + asleep_ns, ticks * S

    drives = 0

    async def drive(*args: Any, **kwargs: Any) -> Any:
        nonlocal drives, asleep_ns
        drives += 1
        asleep_ns += {1: 100 * S, 3: 3612 * S}.get(drives, 0)  # during warm-up and rep 1
        return await real_drive(*args, **kwargs)

    monkeypatch.setattr(macos, "_read_clocks", read_clocks)
    monkeypatch.setattr(experiment, "drive", drive)
    sources = iter(["AC Power", "Battery Power"])
    monkeypatch.setattr(macos, "power_source", lambda: next(sources))

    run_dir, _ = run_in_process(make_cfg(), tmp_path, allow_dirty=True)
    md = json.loads((run_dir / "metadata.json").read_text())
    assert md["sleep"]["per_rep_s"] == [0.0, 3612.0]
    assert md["sleep"]["total_s"] == 3712.0
    assert md["power_source"] == {"start": "AC Power", "end": "Battery Power"}


def test_metadata_records_nulls_off_darwin(tmp_path: Path) -> None:
    run_dir, _ = run_in_process(make_cfg(), tmp_path, allow_dirty=True)
    md = json.loads((run_dir / "metadata.json").read_text())
    assert md["sleep"]["total_s"] is None
    assert md["sleep"]["per_rep_s"] == [None, None]
    assert md["power_source"] == {"start": None, "end": None}


def test_env_check_warns_on_battery_without_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    on_a_mac(monkeypatch, pmset=BATTERY)

    async def run() -> tuple[str, ...]:
        async with mock_client() as client:
            return (await check_server(env_cfg(), client=client)).checks

    assert "WARNING on battery power: runs will not be reportable" in asyncio.run(run())
    on_a_mac(monkeypatch, pmset=AC)
    assert "power AC Power" in asyncio.run(run())


def test_env_check_says_nothing_about_power_off_darwin() -> None:
    async def run() -> tuple[str, ...]:
        async with mock_client() as client:
            return (await check_server(env_cfg(), client=client)).checks

    assert not any("power" in line for line in asyncio.run(run()))

"""The powermetrics sampler against a fake `powermetrics`: a child that writes NUL-separated
plists until SIGTERM, like the real one under sudo."""

from __future__ import annotations

import plistlib
import sys
import time
from pathlib import Path

import pytest

from llmserve.metrics import power

SAMPLE = {
    "elapsed_ns": 500_000_000,
    "thermal_pressure": "Nominal",
    "processor": {
        "cpu_power": 1945.4,
        "gpu_power": 717.3,
        "ane_power": 0.0,
        "combined_power": 2662.7,
    },
    "gpu": {"freq_hz": 457.2, "idle_ratio": 0.25},
}


def fake_powermetrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, body: str = "", fail: str = ""
) -> None:
    """Point `power.ARGV` at a Python child that prints `SAMPLE` every 20 ms (or `body`
    instead of each sample), or with `fail` writes it to stderr and exits 1."""
    sample = tmp_path / "sample.plist"
    sample.write_bytes(plistlib.dumps(SAMPLE))
    script = tmp_path / "powermetrics.py"
    script.write_text(
        "import sys, time\n"
        f"fail = {fail!r}\n"
        "if fail:\n"
        "    sys.stderr.write(fail + '\\n'); sys.exit(1)\n"
        f"data = {body!r}.encode() or open({str(sample)!r}, 'rb').read()\n"
        "while True:\n"
        "    sys.stdout.buffer.write(data + b'\\0'); sys.stdout.flush(); time.sleep(0.02)\n"
    )
    monkeypatch.setattr(power, "ARGV", [sys.executable, str(script)])
    monkeypatch.setattr(power, "FIRST_SAMPLE_TIMEOUT_S", 2.0)


def test_parse_reads_power_frequency_activity_and_pressure() -> None:
    s = power.parse_sample(plistlib.dumps(SAMPLE), t_ns=7)
    assert s == power.PowerSample(
        t_ns=7,
        elapsed_ns=500_000_000,
        gpu_active_ratio=0.75,
        gpu_freq_mhz=457.2,
        cpu_power_mw=1945.4,
        gpu_power_mw=717.3,
        ane_power_mw=0.0,
        combined_power_mw=2662.7,
        thermal_pressure="Nominal",
    )


def test_samples_until_stopped_and_leaves_no_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_powermetrics(monkeypatch, tmp_path)
    sampler = power.PowerSampler(time.perf_counter_ns)
    assert sampler.start()
    proc = sampler._proc
    time.sleep(0.2)
    samples = sampler.stop()
    assert len(samples) >= 3
    assert [s.t_ns for s in samples] == sorted(s.t_ns for s in samples)
    assert proc is not None and proc.poll() is not None
    assert sampler.error is None


def test_no_sudo_rule_is_a_reason_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_powermetrics(monkeypatch, tmp_path, fail="sudo: a password is required")
    sampler = power.PowerSampler(time.perf_counter_ns)
    assert not sampler.start()
    assert sampler.error == "sudo: a password is required"
    assert sampler.stop() == []


def test_a_changed_plist_format_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_powermetrics(monkeypatch, tmp_path, body=plistlib.dumps({"gpu": {}}).decode())
    sampler = power.PowerSampler(time.perf_counter_ns)
    assert not sampler.start()
    assert sampler.error is not None and sampler.error.startswith("unparseable sample: KeyError")
    assert sampler.bad_samples > 0

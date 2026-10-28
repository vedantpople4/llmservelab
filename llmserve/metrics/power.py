"""Mac power and GPU sampling through `powermetrics` (plan Phase 4 item 1), the counterpart of
the NVML thread in `metrics/gpu.py`.

`powermetrics` needs root. `docs/runbooks/mac.md` installs a sudoers rule that allows exactly
`ARGV` and nothing else (no `-o`, so the rule cannot write files as root), which is why the
rate is fixed at 2 Hz here rather than configurable. Without the rule `sudo -n` exits at once,
`start()` returns False with the reason in `.error`, and the run goes on without power samples.

`powermetrics --format plist` writes one property list per sample, NUL-separated. A reader
thread stamps each on the run clock as it arrives; the values describe the `elapsed_ns` before
that. GPU active ratio is `1 - gpu.idle_ratio`: it means only "the GPU was busy", and reads
high during memory-bound decode too.

The child runs as root, but sudo relays SIGTERM from its invoking user, so `stop()` terminates
it. SIGINT is not enough: sudo leaves it to the terminal's process group.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Any

import pyarrow as pa

ARGV = [
    "sudo",
    "-n",
    "/usr/bin/powermetrics",
    "--samplers",
    "cpu_power,gpu_power,thermal",
    "--format",
    "plist",
    "--sample-rate",
    "500",
    "--buffer-size",
    "0",
]
HZ = 2.0
FIRST_SAMPLE_TIMEOUT_S = 5.0

POWER_SCHEMA = pa.schema(
    [
        ("t_ns", pa.int64()),
        ("elapsed_ns", pa.int64()),
        ("gpu_active_ratio", pa.float64()),
        ("gpu_freq_mhz", pa.float64()),
        ("cpu_power_mw", pa.float64()),
        ("gpu_power_mw", pa.float64()),
        ("ane_power_mw", pa.float64()),
        ("combined_power_mw", pa.float64()),
        ("thermal_pressure", pa.string()),
    ]
)


@dataclass(frozen=True)
class PowerSample:
    t_ns: int
    elapsed_ns: int
    gpu_active_ratio: float
    gpu_freq_mhz: float
    cpu_power_mw: float
    gpu_power_mw: float
    ane_power_mw: float
    combined_power_mw: float
    thermal_pressure: str


def parse_sample(data: bytes, *, t_ns: int) -> PowerSample:
    """One plist to a row; raises KeyError or ValueError when a field is missing."""
    plist: dict[str, Any] = plistlib.loads(data)
    processor = plist["processor"]
    gpu = plist["gpu"]
    return PowerSample(
        t_ns=t_ns,
        elapsed_ns=int(plist["elapsed_ns"]),
        gpu_active_ratio=1.0 - float(gpu["idle_ratio"]),
        gpu_freq_mhz=float(gpu["freq_hz"]),  # named Hz, reported in MHz
        cpu_power_mw=float(processor["cpu_power"]),
        gpu_power_mw=float(processor["gpu_power"]),
        ane_power_mw=float(processor["ane_power"]),
        combined_power_mw=float(processor["combined_power"]),
        thermal_pressure=str(plist["thermal_pressure"]),
    )


def _spawn(argv: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class PowerSampler:
    """Run `powermetrics` for one repetition and collect its samples."""

    def __init__(self, clock: Callable[[], int]) -> None:
        self._clock = clock
        self._samples: list[PowerSample] = []
        self._ready = threading.Event()  # first sample, or the child's output ended
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.bad_samples = 0

    def start(self) -> bool:
        """Spawn and wait for the first sample; False, with `.error` set, when none arrives."""
        try:
            self._proc = _spawn(ARGV)
        except OSError as e:
            self.error = f"cannot run {ARGV[0]}: {e}"
            return False
        assert self._proc.stdout is not None
        self._thread = threading.Thread(
            target=self._read, args=(self._proc.stdout,), name="power-sampler", daemon=True
        )
        self._thread.start()
        self._ready.wait(FIRST_SAMPLE_TIMEOUT_S)
        if self._samples:
            return True
        self._halt()
        self.error = self.error or "no sample within 5 s"
        return False

    def _read(self, stream: IO[bytes]) -> None:
        buffer = b""
        while chunk := os.read(stream.fileno(), 65536):
            buffer += chunk
            *complete, buffer = buffer.split(b"\0")
            for data in complete:
                if not data.strip():
                    continue
                try:
                    self._samples.append(parse_sample(data, t_ns=self._clock()))
                except (KeyError, ValueError, TypeError) as e:
                    self.bad_samples += 1
                    self.error = f"unparseable sample: {type(e).__name__}: {e}"
                    continue
                self._ready.set()
        self._ready.set()

    def _halt(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        stderr = b"" if self._proc.stderr is None else self._proc.stderr.read()
        stderr_text = stderr.decode(errors="replace").strip()
        if stderr_text and not self._samples:
            self.error = stderr_text.splitlines()[-1]
        self._proc = None

    def stop(self) -> list[PowerSample]:
        """Terminate `powermetrics` and return the samples in arrival order."""
        self._halt()
        return list(self._samples)

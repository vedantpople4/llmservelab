"""GPU sampling through NVML (plan Phase 2 item 7).

NVML calls block, so sampling runs on a dedicated thread instead of an asyncio task. The thread
appends every tick to a bounded ring buffer that is flushed to `rep-NN/gpu.parquet` when the
repetition ends. No NVML (dev machine, CI) means `start()` returns False and nothing is sampled —
"cannot tell", recorded in run metadata, never a failure.

`_read_sample`/`_gpu_count` are module-level functions so tests can substitute a fake GPU without
a GPU.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import pyarrow as pa

RING_MAXLEN = 1_000_000  # 10 Hz x 2 GPUs x ~14 h of run: a rep can never overrun the buffer

GPU_SCHEMA = pa.schema(
    [
        ("t_ns", pa.int64()),
        ("gpu_idx", pa.int32()),
        ("util_pct", pa.float64()),
        ("mem_used", pa.float64()),
        ("mem_total", pa.float64()),
        ("power_w", pa.float64()),
        ("temp_c", pa.float64()),
        ("sm_clock", pa.float64()),
        ("mem_clock", pa.float64()),
    ]
)


@dataclass(frozen=True)
class GpuSample:
    t_ns: int
    gpu_idx: int
    util_pct: float
    mem_used: float
    mem_total: float
    power_w: float
    temp_c: float
    sm_clock: float
    mem_clock: float


def _gpu_count() -> int:
    """Visible GPU count; 0 when NVML or the driver is unavailable."""
    try:
        import pynvml
    except ImportError:
        return 0
    try:
        pynvml.nvmlInit()
        return int(pynvml.nvmlDeviceGetCount())
    except Exception:  # NVML errors mean "cannot tell", not "no GPU"
        return 0
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _read_sample(gpu_idx: int, t_ns: int) -> GpuSample | None:
    """One timestamped sample of one device; None when any NVML call fails."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GpuSample(
            t_ns=t_ns,
            gpu_idx=gpu_idx,
            util_pct=float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
            mem_used=float(mem.used),
            mem_total=float(mem.total),
            power_w=float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
            temp_c=float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
            sm_clock=float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            mem_clock=float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
        )
    except Exception:  # a failed read skips this tick; the next one may succeed
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


class GpuSampler:
    """Sample every visible GPU at `hz` until `stop()` flushes the ring.

    The clock is injectable (ADR-0006: one monotonic clock per run) and returns nanoseconds.
    """

    def __init__(self, clock: Callable[[], int], *, hz: float) -> None:
        self._clock = clock
        self._hz = hz
        self._ring: deque[GpuSample] = deque(maxlen=RING_MAXLEN)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Begin sampling; False when disabled (`hz == 0`) or NVML is unavailable."""
        if self._hz <= 0:
            return False
        count = _gpu_count()
        if count == 0:
            return False
        self._thread = threading.Thread(
            target=self._run, args=(count,), name="gpu-sampler", daemon=True
        )
        self._thread.start()
        return True

    def _run(self, count: int) -> None:
        interval = 1.0 / self._hz
        next_t = time.monotonic() + interval
        while not self._stop.is_set():
            t_ns = self._clock()
            for idx in range(count):
                sample = _read_sample(idx, t_ns)
                if sample is not None:
                    self._ring.append(sample)
            if next_t < time.monotonic() - interval:
                next_t = time.monotonic() + interval  # fell behind; resync rather than burst
            self._stop.wait(max(0.0, next_t - time.monotonic()))
            next_t += interval

    def stop(self) -> list[GpuSample]:
        """Stop sampling and return the collected rows (in tick order)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        return list(self._ring)

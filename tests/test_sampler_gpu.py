"""NVML GPU sampling (plan Phase 2 item 7).

The NVML calls are injected at the module level (`_gpu_count`/`_read_sample`), so these tests
exercise the real thread, ring buffer and parquet writer on any machine, GPU or not.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from llmserve.metrics import gpu as gpu_mod
from llmserve.metrics.gpu import GPU_SCHEMA, GpuSample, GpuSampler
from llmserve.runner import storage


def fake_sample(gpu_idx: int, t_ns: int) -> GpuSample:
    return GpuSample(
        t_ns=t_ns,
        gpu_idx=gpu_idx,
        util_pct=42.0,
        mem_used=1.0,
        mem_total=2.0,
        power_w=50.0,
        temp_c=40.0,
        sm_clock=1590.0,
        mem_clock=5001.0,
    )


def test_samples_both_gpus_until_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_mod, "_gpu_count", lambda: 2)
    monkeypatch.setattr(gpu_mod, "_read_sample", fake_sample)
    sampler = GpuSampler(time.perf_counter_ns, hz=20)
    assert sampler.start() is True
    time.sleep(0.15)
    samples = sampler.stop()

    assert len(samples) >= 2  # first tick is immediate, then every 50 ms
    assert sorted({s.gpu_idx for s in samples}) == [0, 1]  # every visible device is sampled
    assert all(s.util_pct == 42.0 and s.mem_total == 2.0 for s in samples)
    assert [s.t_ns for s in samples] == sorted(s.t_ns for s in samples)  # tick order


def test_start_refuses_when_disabled_or_nvml_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    assert GpuSampler(time.perf_counter_ns, hz=0).start() is False
    monkeypatch.setattr(gpu_mod, "_gpu_count", lambda: 0)
    assert GpuSampler(time.perf_counter_ns, hz=10).start() is False
    assert GpuSampler(time.perf_counter_ns, hz=10).stop() == []  # never started: harmless flush


def test_ring_buffer_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_mod, "_gpu_count", lambda: 1)
    monkeypatch.setattr(gpu_mod, "_read_sample", fake_sample)
    monkeypatch.setattr(gpu_mod, "RING_MAXLEN", 3)
    sampler = GpuSampler(time.perf_counter_ns, hz=1000)
    assert sampler.start() is True
    time.sleep(0.05)
    samples = sampler.stop()
    assert 1 <= len(samples) <= 3  # overflow drops the oldest, never the newest


def test_gpu_parquet_round_trip(tmp_path: Path) -> None:
    samples = [fake_sample(0, 1000), fake_sample(1, 2000)]
    path = tmp_path / "gpu.parquet"
    storage.write_gpu(path, samples)
    table = pq.read_table(path)
    assert table.column_names == GPU_SCHEMA.names
    assert table.to_pylist() == [asdict(s) for s in samples]


def test_empty_gpu_parquet_keeps_the_schema(tmp_path: Path) -> None:
    path = tmp_path / "gpu.parquet"
    storage.write_gpu(path, [])
    table = pq.read_table(path)
    assert table.num_rows == 0
    assert table.schema == GPU_SCHEMA

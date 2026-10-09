"""End-to-end benchmark runs against the in-process mock (plan Phase 2 items 8-10).

Covers the whole pipeline without a socket: warm-up discarded, N repetitions frozen to parquet,
per-rep and cross-rep summaries, metadata field coverage (PRD §16), same-seed workload
byte-identity (PRD §29), open-loop steady-window trimming, and the dirty-tree gate. The real-
socket CLI path lives in `test_integration_socket.py`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from llmserve import cli
from llmserve.client.capabilities import capabilities
from llmserve.config.loader import config_hash, load_config
from llmserve.config.schema import ExperimentConfig
from llmserve.metrics.gpu import GPU_SCHEMA, GpuSample
from llmserve.metrics.records import CHUNK_SCHEMA, REQUEST_SCHEMA
from llmserve.mock import DelayModel, MockEngine, create_app
from llmserve.runner import experiment
from llmserve.runner.experiment import DirtyTreeError, run_experiment
from llmserve.workload.generator import load_workload, materialize, save_workload
from llmserve.workload.prompts import PromptBuilder

# PRD §16 field coverage: every key must be present (null when the host cannot provide it).
METADATA_KEYS = (
    "run_id",
    "experiment",
    "description",
    "schema_version",
    "started_utc",
    "finished_utc",
    "config_hash",
    "uv_lock_hash",
    "git_sha",
    "git_dirty",
    "hostname",
    "platform",
    "python_version",
    "seed",
    "repetitions",
    "child_seed_keys",
    "server",
    "backend_version",
    "tokenizer_revision",
    "vllm_launch_args",
    "max_model_len",
    "gpu_memory_utilization",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu",
    "load",
    "gateway",
    "measurement",
    "warmup",
    "clock_anchor_ns",
    "monotonic_clock",
    "samplers",
)


class _StubTokenizer:
    """Character-level stand-in so these tests never touch the network."""

    vocab_size = 1000

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 997 + 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids)


def make_builder() -> PromptBuilder:
    corpus = np.random.default_rng(7).integers(1, 1000, size=5000).astype(np.int64)
    return PromptBuilder(_StubTokenizer(), corpus)


def fake_gpu_sample(gpu_idx: int, t_ns: int) -> GpuSample:
    """A stand-in NVML reading, injected in place of `metrics.gpu._read_sample`."""
    return GpuSample(
        t_ns=t_ns,
        gpu_idx=gpu_idx,
        util_pct=33.0,
        mem_used=1.0,
        mem_total=2.0,
        power_w=25.0,
        temp_c=39.0,
        sm_clock=1590.0,
        mem_clock=5001.0,
    )


def with_fake_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """One visible GPU at 33% utilization, and a mock told it has GPU metrics."""
    from dataclasses import replace

    from llmserve.metrics import gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "_gpu_count", lambda: 1)
    monkeypatch.setattr(gpu_mod, "_read_sample", fake_gpu_sample)
    monkeypatch.setattr(
        experiment, "capabilities", lambda kind: replace(capabilities(kind), gpu_metrics=True)
    )


def make_cfg(*, open_loop: bool = False) -> ExperimentConfig:
    load: dict[str, Any] = (
        {"mode": "closed", "concurrency": 4, "requests": 40}
        if not open_loop
        else {"mode": "open", "arrival": {"process": "poisson", "rate": 30}, "duration_s": 2.0}
    )
    measurement: dict[str, Any] = {"warmup": {"requests": 8, "settle_timeout_s": 1.0}}
    if open_loop:
        measurement.update({"warmup_s": 0.5, "cooldown_s": 0.5})
    return ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": "bench-it",
            "description": "in-process benchmark pipeline test",
            "seed": 42,
            "repetitions": 2 if not open_loop else 1,
            "measurement": measurement,
            "server": {"kind": "mock", "endpoint": "http://test/v1", "model": "mock"},
            "load": load,
            "workload": {
                "classes": [
                    {
                        "name": "c",
                        "prompt": {"distribution": "fixed", "tokens": 16},
                        "output": {"distribution": "fixed", "tokens": 4},
                    }
                ]
            },
        }
    )


def run_in_process(cfg: ExperimentConfig, tmp_path: Path, **kwargs: Any) -> Any:
    builder = kwargs.pop("builder", None) or make_builder()
    app = create_app(MockEngine(delay=DelayModel.instant()))

    async def run() -> Any:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_experiment(
                cfg, out_root=tmp_path / "results", client=client, builder=builder, **kwargs
            )

    return asyncio.run(run())


def test_closed_benchmark_end_to_end(tmp_path: Path) -> None:
    cfg = make_cfg()
    builder = make_builder()
    # PRD §29: the run's rep-0 workload file is exactly what generate-workload produces.
    expected = tmp_path / "expected.parquet"
    save_workload(materialize(cfg, rep=0, builder=builder), expected)

    run_dir, cross = run_in_process(
        cfg, tmp_path, builder=builder, allow_dirty=True, endpoint="http://test/v1"
    )

    # --- run directory layout (plan §4) ---
    assert (run_dir / "config.resolved.yaml").is_file()
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "workload.parquet").read_bytes() == expected.read_bytes()
    for rep in range(2):
        rep_path = run_dir / f"rep-{rep:02d}"
        assert (rep_path / "workload.parquet").is_file()
        assert (rep_path / "requests.parquet").is_file()
        assert (rep_path / "chunks.parquet").is_file()
        assert (rep_path / "summary.json").is_file()

    # --- resolved config round-trips to the same hash ---
    reloaded = ExperimentConfig.model_validate(
        yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    )
    assert config_hash(reloaded) == config_hash(cfg)

    # --- requests/chunks follow the recorded schemas ---
    run_id = run_dir.name
    table = pq.read_table(run_dir / "rep-00" / "requests.parquet")
    assert table.column_names == REQUEST_SCHEMA.names
    assert table.num_rows == 40
    assert set(table.column("rep").to_pylist()) == {0}
    assert set(table.column("run_id").to_pylist()) == {run_id}
    chunks = pq.read_table(run_dir / "rep-01" / "chunks.parquet")
    assert chunks.column_names == CHUNK_SCHEMA.names
    assert chunks.num_rows == sum(
        pq.read_table(run_dir / "rep-01" / "requests.parquet").column("n_chunks").to_pylist()
    )

    # --- workload files round-trip per rep (different seed child, ADR-0004) ---
    assert load_workload(run_dir / "rep-01" / "workload.parquet") == materialize(
        cfg, rep=1, builder=builder
    )

    # --- metadata (PRD §16 coverage + the run's own facts) ---
    md = json.loads((run_dir / "metadata.json").read_text())
    for key in METADATA_KEYS:
        assert key in md, key
    assert md["experiment"] == "bench-it"
    assert md["seed"] == 42
    assert md["repetitions"] == 2
    assert md["child_seed_keys"] == [[42, 0], [42, 1]]
    assert md["config_hash"] == config_hash(cfg)
    assert isinstance(md["git_dirty"], bool) and md["git_sha"]
    assert md["clock_anchor_ns"] > 0
    assert md["backend_version"] == {"source": "/version", "value": "0.1.0"}
    assert md["warmup"] == {
        "performed": True,
        "requests": 8,
        "ok": 8,
        "gpu_util_pct": None,  # no NVML on this host
        "server_idle": None,  # the mock gains /metrics with the simulator (item 4)
    }
    assert md["samplers"]["active"] == []

    # --- summaries ---
    assert cross["repetitions"] == 2
    assert cross["invalid"] is False
    assert cross["success_rate"]["mean"] == 1.0
    for rep_summary in cross["per_rep"]:
        assert rep_summary["n_requests"] == 40  # warm-up never lands in a rep
        assert rep_summary["window"]["mode"] == "closed"
        ttft = rep_summary["latency"]["ttft_s"]
        assert 20 <= ttft["count"] < 40  # ramp dispatches are outside the steady window
        assert rep_summary["throughput"] is not None
        assert rep_summary["throughput"]["req_s"] > 0
    assert json.loads((run_dir / "summary.json").read_text())["repetitions"] == 2


def test_gpu_sampler_runs_and_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake GPU plus a mock told it has GPU metrics: the sampler must produce gpu.parquet for
    # every rep and say so in metadata — no NVML and no backend honesty required.
    with_fake_gpu(monkeypatch)

    run_dir, _ = run_in_process(make_cfg(), tmp_path, allow_dirty=True)

    for rep in range(2):
        table = pq.read_table(run_dir / f"rep-{rep:02d}" / "gpu.parquet")
        assert table.column_names == GPU_SCHEMA.names
        assert table.num_rows >= 1
        assert set(table.column("util_pct").to_pylist()) == {33.0}
        events = (run_dir / f"rep-{rep:02d}" / "events.jsonl").read_text()
        assert events == ""  # a healthy sampler produces no events
        assert not (run_dir / f"rep-{rep:02d}" / "server.parquet").exists()  # no /metrics yet
    md = json.loads((run_dir / "metadata.json").read_text())
    assert md["samplers"] == {
        "active": ["gpu"],  # the mock still has no /metrics endpoint to poll
        "gpu_hz": 10,
        "server_hz": 2,
        "dcgm": "auto",
        "errors": {},
    }


def test_open_benchmark_trims_the_steady_window(tmp_path: Path) -> None:
    cfg = make_cfg(open_loop=True)
    run_dir, cross = run_in_process(cfg, tmp_path, allow_dirty=True)

    rep_summary = cross["per_rep"][0]
    window = rep_summary["window"]
    assert window["mode"] == "open"
    assert rep_summary["n_requests"] > 0
    assert 0 < rep_summary["n_measured"] < rep_summary["n_requests"]  # trims bit on both ends
    assert rep_summary["success_rate"] == 1.0
    assert cross["invalid"] is False
    lag = rep_summary["latency"]["client_lag_s"]
    assert lag["count"] == rep_summary["n_measured"]
    assert lag["p99"] is not None and lag["p99"] < 0.05  # harness overhead vs the mock
    assert (run_dir / "metadata.json").is_file()


def test_dirty_tree_refuses_without_allow_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment, "git_state", lambda: ("0123456789abcdef", True))
    with pytest.raises(DirtyTreeError, match="dirty"):
        asyncio.run(run_experiment(make_cfg(), out_root=tmp_path, builder=make_builder()))
    assert list(tmp_path.iterdir()) == []  # nothing written before the gate


def test_clean_tree_writes_with_the_recorded_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment, "git_state", lambda: ("0123456789abcdef", False))
    run_dir, _ = run_in_process(make_cfg(), tmp_path)
    md = json.loads((run_dir / "metadata.json").read_text())
    assert md["git_sha"] == "0123456789abcdef"
    assert md["git_dirty"] is False
    assert run_dir.name.endswith("-0123456")  # run_id carries the git sha prefix


def test_run_config_alias_and_benchmark_flag_wiring(tmp_path: Path) -> None:
    # `run --config` is the PRD's alias of `benchmark`; a missing config exits 2, proving the
    # flags reach `_benchmark` without spending time on a real run.
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--config", str(missing), "-o", str(tmp_path)])
    assert exc.value.code == 2


def test_config_loader_is_used_by_benchmark(tmp_path: Path) -> None:
    # The CLI path loads YAML (with workload.ref resolution) rather than a hand-built config.
    cfg = make_cfg()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.canonical(), sort_keys=False))
    assert load_config(cfg_path) == cfg

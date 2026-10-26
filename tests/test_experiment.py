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
from tcp import served_over_tcp

from llmserve import cli
from llmserve.client.capabilities import capabilities
from llmserve.config.loader import config_hash, load_config
from llmserve.config.schema import ExperimentConfig
from llmserve.metrics.gpu import GPU_SCHEMA, GpuSample
from llmserve.metrics.records import CHUNK_SCHEMA, REQUEST_SCHEMA
from llmserve.metrics.server import SERVER_SCHEMA
from llmserve.mock import CbParams, ContinuousBatchEngine, DelayModel, MockEngine, create_app
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
    "sleep",
    "power_source",
    "thermal_probe",
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
    engine = kwargs.pop("engine", None) or MockEngine(delay=DelayModel.instant())
    app = create_app(engine)

    async def run() -> Any:
        transport = httpx.ASGITransport(app=app)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://test") as client,
            httpx.AsyncClient(transport=transport, base_url="http://test") as metrics_client,
        ):
            return await run_experiment(
                cfg,
                out_root=tmp_path / "results",
                client=client,
                metrics_client=metrics_client,
                builder=builder,
                **kwargs,
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
        "server_idle": True,  # the mock's /metrics reported an empty engine after warm-up
    }
    probe = md["thermal_probe"]
    assert (probe["requests"], probe["prompt_tokens"], probe["output_tokens"]) == (16, 128, 32)
    for when in ("before", "after"):
        assert probe[when]["n"] == len(probe[when]["tpot_s"]) == 16
        assert probe[when]["tpot_p50_s"] == float(np.median(probe[when]["tpot_s"]))
    # The mock has no GPU but does serve /metrics: exactly one sampler ran, none failed.
    assert md["samplers"] == {
        "active": ["server"],
        "gpu_hz": 10,
        "server_hz": 2,
        "dcgm": "auto",
        "errors": {},
        "server_failed_scrapes": 0,
    }
    # events.jsonl exists for every rep even when nothing happened (plan §4 layout).
    for rep in range(2):
        rep_path = run_dir / f"rep-{rep:02d}"
        assert (rep_path / "events.jsonl").is_file()
        assert not (rep_path / "gpu.parquet").exists()  # never wanted (mock has no GPU)
        server = pq.read_table(rep_path / "server.parquet")
        assert server.column_names == SERVER_SCHEMA.names
        assert server.num_rows >= 1  # the validating scrape alone guarantees a row

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
    md = json.loads((run_dir / "metadata.json").read_text())
    assert md["samplers"] == {
        "active": ["gpu", "server"],  # both are fed: a fake GPU and the mock's /metrics
        "gpu_hz": 10,
        "server_hz": 2,
        "dcgm": "auto",
        "errors": {},
        "server_failed_scrapes": 0,
    }


def test_progress_lines_describe_the_run(tmp_path: Path) -> None:
    # Every stage narrates itself: run header + interpreter context (which Python, which
    # sitecustomize), the sampler plan, the warm-up settle verdict, per-rep status with
    # sampler row counts — the log a GPU run leaves behind when nobody watches it.
    lines: list[str] = []
    run_dir, _ = run_in_process(
        make_cfg(),
        tmp_path,
        allow_dirty=True,
        endpoint="http://test/v1",
        progress=lines.append,
    )

    joined = "\n".join(lines)
    assert f"run {run_dir.name}: [bench-it] mock → http://test/v1 · closed load" in joined
    assert "2 rep(s) · seed 42 · git " in joined
    assert "python " in joined and " @" in joined  # which interpreter ran the harness
    plan = "samplers: gpu off (mock has no gpu_metrics) · server 2 Hz (http://test/metrics)"
    assert plan in joined
    assert "warm-up: 8 requests (discarded)" in joined
    assert "warm-up done: 8/8 ok · server idle True · gpu n/a (no NVML)" in joined
    assert "rep 1/2: 40/40 ok · samplers gpu 0 / server " in joined
    assert "rep 2/2: 40/40 ok" in joined
    assert "0 event(s)" in joined


def test_simulator_run_with_both_samplers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The whole Phase 2 measurement path against the continuous-batching simulator: every rep
    # flushes gpu.parquet and server.parquet, warm-up sees an idle engine, nothing fails.
    with_fake_gpu(monkeypatch)
    engine = ContinuousBatchEngine(
        params=CbParams(alpha_s=0.001, beta_s=0.0, gamma_s=0.0, delta_s=0.0)
    )

    run_dir, cross = run_in_process(
        make_cfg(), tmp_path, engine=engine, allow_dirty=True, endpoint="http://test/v1"
    )

    assert cross["success_rate"]["mean"] == 1.0  # the simulator served every request to length
    assert cross["invalid"] is False
    md = json.loads((run_dir / "metadata.json").read_text())
    assert md["samplers"]["active"] == ["gpu", "server"]
    assert md["samplers"]["errors"] == {}
    assert md["warmup"]["server_idle"] is True  # the simulator drained before the settle
    for rep in range(2):
        rep_path = run_dir / f"rep-{rep:02d}"
        gpu = pq.read_table(rep_path / "gpu.parquet")
        server = pq.read_table(rep_path / "server.parquet")
        assert gpu.num_rows >= 1 and server.num_rows >= 1
        assert server.column_names == SERVER_SCHEMA.names
        assert (rep_path / "events.jsonl").read_text() == ""


def test_metrics_scrapes_do_not_queue_behind_benchmark_streams(tmp_path: Path) -> None:
    # A benchmark client whose pool is always full (one connection, four workers) stands in for
    # session 1's overloaded points. Sharing it, every scrape waits behind open streams until it
    # hits PoolTimeout; the sampler must scrape through its own connections instead.
    raw = make_cfg().canonical()
    raw.update(repetitions=1)
    raw["measurement"].update(warmup={"requests": 0}, samplers={"gpu_hz": 0, "server_hz": 50})
    cfg = ExperimentConfig.model_validate(raw)
    app = create_app(MockEngine(delay=DelayModel()))

    async def run(base: str) -> Path:
        limits = httpx.Limits(max_connections=1)
        timeout = httpx.Timeout(30.0, pool=0.05)
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            run_dir, _ = await run_experiment(
                cfg,
                out_root=tmp_path / "results",
                endpoint=f"{base}/v1",
                client=client,
                builder=make_builder(),
                allow_dirty=True,
            )
        return run_dir

    with served_over_tcp(app) as base:
        run_dir = asyncio.run(run(base))

    md = json.loads((run_dir / "metadata.json").read_text())
    assert md["samplers"]["errors"] == {}
    assert md["samplers"]["server_failed_scrapes"] == 0
    assert pq.read_table(run_dir / "rep-00" / "server.parquet").num_rows >= 20


class _InFlight:
    """ASGI wrapper counting concurrent `/v1/completions` requests as the server sees them."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.now = 0
        self.peak = 0

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["path"] != "/v1/completions":
            await self.app(scope, receive, send)
            return
        self.now += 1
        self.peak = max(self.peak, self.now)
        try:
            await self.app(scope, receive, send)
        finally:
            self.now -= 1


def _slow_run_over_tcp(
    tmp_path: Path, *, concurrency: int, request_timeout_s: float = 600
) -> tuple[dict[str, Any], int]:
    """Run with the runner's own clients against a TCP mock where each measured request spends
    ~1 s in prefill (probes stay short); returns the cross-rep summary and the server's peak
    in-flight count."""
    raw = make_cfg().canonical()
    raw.update(repetitions=1)
    raw["load"].update(concurrency=concurrency, requests=concurrency)
    raw["workload"]["classes"][0]["prompt"]["tokens"] = 2048
    raw["measurement"].update(
        warmup={"requests": 0},
        samplers={"gpu_hz": 0, "server_hz": 0},
        request_timeout_s=request_timeout_s,
    )
    cfg = ExperimentConfig.model_validate(raw)
    delay = DelayModel(
        ttft_base_s=0.0, ttft_per_prompt_token_s=5e-4, token_base_s=0.001, token_in_flight_s=0.0
    )
    server = _InFlight(create_app(MockEngine(delay=delay)))
    with served_over_tcp(server) as base:
        _, cross = asyncio.run(
            run_experiment(
                cfg,
                out_root=tmp_path / "results",
                endpoint=f"{base}/v1",
                builder=make_builder(),
                allow_dirty=True,
            )
        )
    return cross, server.peak


def test_the_runners_client_never_caps_in_flight_requests(tmp_path: Path) -> None:
    # httpx's default pool holds 100 connections, which silently capped every load above 100.
    cross, peak = _slow_run_over_tcp(tmp_path, concurrency=150)
    assert cross["success_rate"]["mean"] == 1.0
    assert peak == 150


def test_the_runners_client_still_times_requests_out(tmp_path: Path) -> None:
    cross, _ = _slow_run_over_tcp(tmp_path, concurrency=2, request_timeout_s=0.2)
    assert cross["success_rate"]["mean"] == 0.0


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


def _probes_reading(monkeypatch: pytest.MonkeyPatch, p50s: list[float]) -> None:
    readings = iter(p50s)

    async def probe(cfg: ExperimentConfig, when: str, **kwargs: Any) -> dict[str, Any]:
        p50 = next(readings)
        return {"tpot_s": [p50] * 16, "tpot_p50_s": p50, "n": 16}

    monkeypatch.setattr(experiment, "_thermal_probe", probe)
    monkeypatch.setattr(experiment, "COOLDOWN_PAUSE_S", 0.0)


def test_the_first_rep_waits_for_a_cool_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _probes_reading(monkeypatch, [0.091, 0.067, 0.050, 0.052])  # E06 began at 67-91 ms
    lines: list[str] = []
    run_dir, _ = run_in_process(make_cfg(), tmp_path, allow_dirty=True, progress=lines.append)
    probe = json.loads((run_dir / "metadata.json").read_text())["thermal_probe"]
    assert probe["before"]["tpot_p50_s"] == 0.050
    assert probe["cooldown"] == {
        "limit_tpot_s": experiment.COOL_PROBE_TPOT_S,
        "attempts": 3,
        "waited_s": probe["cooldown"]["waited_s"],
        "cool": True,
    }
    assert sum(line.startswith("Mac still warm") for line in lines) == 2


def test_a_mac_that_never_cools_runs_anyway_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _probes_reading(monkeypatch, [0.080, 0.081])
    monkeypatch.setattr(experiment, "COOLDOWN_TIMEOUT_S", 0.0)
    run_dir, _ = run_in_process(make_cfg(), tmp_path, allow_dirty=True)
    probe = json.loads((run_dir / "metadata.json").read_text())["thermal_probe"]
    assert (probe["cooldown"]["attempts"], probe["cooldown"]["cool"]) == (1, False)

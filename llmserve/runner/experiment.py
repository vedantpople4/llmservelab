"""One command from config to a complete run directory (plan Phase 2 item 8).

Flow: git gate → warm-up (a separate small workload, discarded, then settle until the GPU is
idle) → N repetitions of materialize → drive → steady window → parquet → per-rep summary →
cross-rep summary → metadata. One `RunClock` for the whole process, so timestamps are
comparable across warm-up and every repetition (ADR-0006).

The warm-up uses `spawn_key=(repetitions,)` — a seed child outside the measured range — so its
prompts differ from every measured request (prefix caches cannot make the first rep look fast).
The server-metrics half of the settle ("running == waiting == 0") lands with the samplers in
Phase 2 item 7; until then metadata records `server_idle: null` rather than pretending.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llmserve.analysis.aggregate import aggregate, summarize, window_of
from llmserve.client.capabilities import Capabilities, capabilities
from llmserve.config.loader import config_hash
from llmserve.config.schema import ExperimentConfig, Load
from llmserve.runner import storage
from llmserve.runner.clock import RunClock
from llmserve.runner.env_check import GPU_IDLE_UTIL_PCT, base_url, settled_gpu_utilization
from llmserve.runner.metadata import build_metadata, git_state
from llmserve.runner.run import drive
from llmserve.workload.generator import materialize, save_workload
from llmserve.workload.prompts import PromptBuilder


class DirtyTreeError(RuntimeError):
    """The working tree has uncommitted changes and `--allow-dirty` was not passed."""


async def run_experiment(
    cfg: ExperimentConfig,
    *,
    out_root: Path,
    endpoint: str | None = None,
    allow_dirty: bool = False,
    client: httpx.AsyncClient | None = None,
    builder: PromptBuilder | None = None,
    progress: Callable[[str], None] = lambda _msg: None,
) -> tuple[Path, dict[str, Any]]:
    """Run `cfg` end to end under `out_root`; returns the run directory and cross-rep summary.

    `client` and `builder` are injectable for tests; the CLI uses a real HTTP client and the
    pinned tokenizer. `endpoint` overrides `server.endpoint` (the mock's port is assigned at
    runtime). Paper runs refuse a dirty tree unless `allow_dirty`.
    """
    sha, dirty = git_state()
    if dirty and not allow_dirty:
        raise DirtyTreeError(
            f"working tree is dirty (git {sha or 'unknown'}); commit, or pass --allow-dirty"
        )

    builder = builder or PromptBuilder.default()
    caps = capabilities(cfg.server.kind)
    token_counter = None if caps.token_id_prompts else _token_counter(builder)
    target = endpoint or cfg.server.endpoint
    clock = RunClock()
    started = datetime.now(UTC)
    cfg_hash = config_hash(cfg)
    run_id = storage.make_run_id(cfg_hash, sha, started)
    run_dir = out_root / cfg.experiment / run_id
    run_dir.mkdir(parents=True)  # a same-second same-config rerun collides loudly, not silently
    storage.write_config(cfg, run_dir / "config.resolved.yaml")

    own_client = client is None
    if client is None:
        client = httpx.AsyncClient()
    try:
        warmup = await _warmup(
            cfg,
            client=client,
            caps=caps,
            clock=clock,
            target=target,
            builder=builder,
            token_counter=token_counter,
            progress=progress,
        )
        summaries: list[dict[str, Any]] = []
        for rep in range(cfg.repetitions):
            workload = materialize(cfg, rep=rep, builder=builder)
            rep_path = storage.rep_dir(run_dir, rep)
            rep_path.mkdir()
            save_workload(workload, rep_path / "workload.parquet")
            if rep == 0:
                save_workload(workload, run_dir / "workload.parquet")

            t0_ns = clock()
            records = await drive(
                cfg,
                workload,
                clock=clock,
                client=client,
                caps=caps,
                endpoint=target,
                token_counter=token_counter,
            )
            window = window_of(records, cfg, t0_ns=t0_ns)
            summary = summarize(records, rep=rep, window=window)
            storage.write_requests(rep_path / "requests.parquet", records, run_id=run_id, rep=rep)
            storage.write_chunks(rep_path / "chunks.parquet", records)
            storage.write_json(summary, rep_path / "summary.json")
            summaries.append(summary)
            note = " (INVALID: >1% failures)" if summary["invalid"] else ""
            progress(
                f"rep {rep + 1}/{cfg.repetitions}: "
                f"{summary['n_ok']}/{summary['n_requests']} ok{note}"
            )

        cross = aggregate(summaries)
        storage.write_json(cross, run_dir / "summary.json")
        storage.write_json(
            build_metadata(
                cfg=cfg,
                run_id=run_id,
                config_hash=cfg_hash,
                endpoint=target,
                git_sha=sha,
                git_dirty=dirty,
                clock_anchor_ns=clock.wall_anchor_ns,
                started_utc=started.isoformat(),
                finished_utc=datetime.now(UTC).isoformat(),
                warmup=warmup,
                backend_version=await _backend_version(client, target, caps),
            ),
            run_dir / "metadata.json",
        )
        return run_dir, cross
    finally:
        if own_client:
            await client.aclose()


async def _warmup(
    cfg: ExperimentConfig,
    *,
    client: httpx.AsyncClient,
    caps: Capabilities,
    clock: RunClock,
    target: str,
    builder: PromptBuilder,
    token_counter: Callable[[str], int] | None,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """Run a small closed-loop workload, discard it, then wait for the GPU to go idle."""
    requests_n = cfg.measurement.warmup.requests
    if requests_n == 0:
        return {"performed": False, "requests": 0, "ok": 0, "gpu_util_pct": None}
    progress(f"warm-up: {requests_n} requests (discarded)")
    warm_cfg = cfg.model_copy(
        update={
            "load": Load.model_validate(
                {
                    "mode": "closed",
                    "concurrency": max(1, min(cfg.load.concurrency or 4, requests_n)),
                    "requests": requests_n,
                }
            )
        }
    )
    workload = materialize(warm_cfg, rep=cfg.repetitions, builder=builder)
    records = await drive(
        warm_cfg,
        workload,
        clock=clock,
        client=client,
        caps=caps,
        endpoint=target,
        token_counter=token_counter,
    )
    util = _settle(cfg.measurement.warmup.settle_timeout_s)
    return {
        "performed": True,
        "requests": requests_n,
        "ok": sum(1 for r in records if r.ok),
        "gpu_util_pct": util,
        "server_idle": None,  # needs the /metrics scraper (Phase 2 item 7)
        "note": "server-metrics idle wait lands with the samplers",
    }


def _settle(timeout_s: float) -> float | None:
    """Wait until NVML reports < 5% GPU utilization or the settle timeout expires (item 8).

    Returns the last utilization seen (None without NVML — nothing to wait for).
    """
    deadline = time.monotonic() + timeout_s
    while True:
        util = settled_gpu_utilization()  # ~3 s per call with NVML, instant without
        if util is None or util <= GPU_IDLE_UTIL_PCT or time.monotonic() >= deadline:
            return util
        time.sleep(1.0)


def _token_counter(builder: PromptBuilder) -> Callable[[str], int]:
    """Text→tokens for backends that cannot take token IDs (ADR-009); unused for mock/vLLM."""

    def count(text: str) -> int:
        return len(builder.tokenizer.encode(text))

    return count


async def _backend_version(
    client: httpx.AsyncClient, target: str, caps: Capabilities
) -> dict[str, Any]:
    """`GET /version` for metadata; best-effort, because reachability is already proven."""
    if not caps.version_endpoint:
        return {"source": None, "value": None}
    try:
        resp = await client.get(f"{base_url(target)}/version")
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return {"source": "/version", "value": None}
    value = data.get("version") if isinstance(data, dict) else None
    return {"source": "/version", "value": value}

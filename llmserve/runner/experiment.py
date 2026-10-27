"""One command from config to a complete run directory (plan Phase 2 items 7-10).

Flow: git gate → warm-up (a separate small workload, discarded, then settle until the GPU is
idle and the server reports no in-flight work) → thermal probe, repeated until cool → N
repetitions of materialize → drive → steady window → parquet → per-rep summary → thermal probe →
cross-rep summary → metadata. One `RunClock` for the whole process, so timestamps are comparable
across warm-up and every repetition (ADR-0006).

The warm-up uses `spawn_key=(repetitions,)` — a seed child outside the measured range — so its
prompts differ from every measured request (prefix caches cannot make the first rep look fast).

The thermal probe is a fixed closed-loop batch (C=1, 128-token prompts, 32 output tokens) run just
before the first rep and just after the last. Its TPOT p50 depends on the machine's state only,
so `Run.thermal_drift` compares the two probes; comparing the reps would confound heat with
per-rep workload samples that differ. Probe requests are recorded only in `metadata.thermal_probe`.
The "before" probe doubles as a cool-down gate: it repeats, with pauses, until the Mac probes
cool, because heat carries from one experiment into the next (session 4's E06 started at 67-91 ms
right after E05). The power source is polled through the whole run, not only at its ends.

The samplers (NVML thread at `gpu_hz`, `/metrics` scraper at `server_hz`) run inside every
repetition when the config asks for them and the backend supports them. A sampler that cannot
start or dies mid-run deactivates itself and is recorded in `metadata.samplers.errors` and in
that rep's `events.jsonl`; a run is never failed for a missing metric (ADR-0010).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from llmserve.analysis.aggregate import aggregate, summarize, window_of
from llmserve.client.capabilities import Capabilities, capabilities
from llmserve.config import schema
from llmserve.config.loader import config_hash
from llmserve.config.schema import ExperimentConfig, Load
from llmserve.metrics import latency
from llmserve.metrics.gpu import GpuSampler
from llmserve.metrics.server import ServerSampler, describe, lookup, parse_prometheus
from llmserve.runner import macos, storage
from llmserve.runner.clock import RunClock
from llmserve.runner.env_check import GPU_IDLE_UTIL_PCT, base_url, settled_gpu_utilization
from llmserve.runner.metadata import build_metadata, git_state
from llmserve.runner.run import drive
from llmserve.workload.generator import materialize, save_workload
from llmserve.workload.prompts import PromptBuilder

PROBE_REQUESTS = 16  # with 4, one slow request moved the median past the drift threshold
PROBE_PROMPT_TOKENS = 128
PROBE_OUTPUT_TOKENS = 32
COOL_PROBE_TPOT_S = 0.058  # a cool M5 probes at 45-55 ms (ADR-0013)
COOLDOWN_TIMEOUT_S = 1200.0
COOLDOWN_PAUSE_S = 60.0
POWER_POLL_S = 15.0
PROBE_SETTLE_S = 30.0  # E03's probe ran 57 ms for ~10 requests after its warm-up batch, then 48


class DirtyTreeError(RuntimeError):
    """The working tree has uncommitted changes and `--allow-dirty` was not passed."""


async def run_experiment(
    cfg: ExperimentConfig,
    *,
    out_root: Path,
    endpoint: str | None = None,
    allow_dirty: bool = False,
    client: httpx.AsyncClient | None = None,
    metrics_client: httpx.AsyncClient | None = None,
    builder: PromptBuilder | None = None,
    progress: Callable[[str], None] = lambda _msg: None,
) -> tuple[Path, dict[str, Any]]:
    """Run `cfg` end to end under `out_root`; returns the run directory and cross-rep summary.

    `client` (benchmark requests), `metrics_client` (the `/metrics` sampler) and `builder` are
    injectable for tests; the CLI uses real HTTP clients and the pinned tokenizer. `endpoint`
    overrides `server.endpoint` (the mock's port is assigned at runtime). Paper runs refuse a
    dirty tree unless `allow_dirty`.
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

    progress(
        f"run {run_id}: [{cfg.experiment}] {cfg.server.kind} → {target} · "
        f"{cfg.load.mode} load · {cfg.repetitions} rep(s) · seed {cfg.seed} · "
        f"git {sha or 'unknown'}{' (dirty)' if dirty else ''}"
    )
    for line in _startup_lines():
        progress(line)

    owned: list[httpx.AsyncClient] = []
    if client is None:
        # The load sets concurrency; httpx's default 100-connection pool would silently cap it.
        # Per-request timeouts (`request_timeout_s`) still bound each request.
        client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)
        )
        owned.append(client)
    if metrics_client is None:
        # Its own pool: on a shared one, scrapes queue behind open streams and hit PoolTimeout.
        metrics_client = httpx.AsyncClient()
        owned.append(metrics_client)
    run_mark = macos.sleep_mark()
    power_start = macos.power_source()
    if power_start == macos.BATTERY_POWER:
        progress("WARNING on battery power: this run will not be reportable")
    macos.keep_awake(progress)
    power_seen: set[str] = set()
    power_watch = asyncio.create_task(macos.watch_power(power_seen, POWER_POLL_S))
    try:
        samplers = _Samplers(
            cfg,
            caps=caps,
            clock=clock,
            client=metrics_client,
            target=target,
            kind=cfg.server.kind,
            progress=progress,
        )
        progress(samplers.plan())
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
        probe_args: dict[str, Any] = {
            "client": client,
            "caps": caps,
            "clock": clock,
            "target": target,
            "builder": builder,
            "token_counter": token_counter,
            "progress": progress,
        }
        probe_before, cooldown = await _cool_probe(cfg, **probe_args)
        summaries: list[dict[str, Any]] = []
        slept_per_rep: list[float | None] = []
        for rep in range(cfg.repetitions):
            workload = materialize(cfg, rep=rep, builder=builder)
            rep_path = storage.rep_dir(run_dir, rep)
            rep_path.mkdir()
            save_workload(workload, rep_path / "workload.parquet")
            if rep == 0:
                save_workload(workload, run_dir / "workload.parquet")

            rep_mark = macos.sleep_mark()
            t0_ns = clock()
            gpu, server, events = await samplers.start()
            try:
                records = await drive(
                    cfg,
                    workload,
                    clock=clock,
                    client=client,
                    caps=caps,
                    endpoint=target,
                    token_counter=token_counter,
                )
            finally:
                counts = await samplers.stop(gpu, server, events, rep_path)  # flush on a crash too
            slept = macos.slept_s(rep_mark)
            slept_per_rep.append(slept)
            window = window_of(records, cfg, t0_ns=t0_ns)
            summary = summarize(records, rep=rep, window=window)
            storage.write_requests(rep_path / "requests.parquet", records, run_id=run_id, rep=rep)
            storage.write_chunks(rep_path / "chunks.parquet", records)
            storage.write_json(summary, rep_path / "summary.json")
            summaries.append(summary)
            note = " (INVALID: >1% failures)" if summary["invalid"] else ""
            if slept is not None and slept >= 1:
                note += f" (Mac slept {slept:.0f} s)"
            progress(
                f"rep {rep + 1}/{cfg.repetitions}: "
                f"{summary['n_ok']}/{summary['n_requests']} ok{note} · "
                f"samplers gpu {counts['gpu']} / server {counts['server']} rows · "
                f"{counts['events']} event(s)"
            )

        probe_after = await _thermal_probe(cfg, "after", **probe_args)
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
                samplers={
                    "active": samplers.active,  # started at least once; wanted-but-absent -> errors
                    "gpu_hz": cfg.measurement.samplers.gpu_hz,
                    "server_hz": cfg.measurement.samplers.server_hz,
                    "dcgm": cfg.measurement.samplers.dcgm,
                    "errors": samplers.errors,
                    "server_failed_scrapes": samplers.server_failures,
                },
                sleep={
                    "total_s": macos.slept_s(run_mark),
                    "per_rep_s": slept_per_rep,
                    "method": "delta CLOCK_MONOTONIC - delta CLOCK_UPTIME_RAW (Darwin only)",
                },
                power_source={
                    "start": power_start,
                    "end": macos.power_source(),
                    "seen": sorted(power_seen),
                },
                thermal_probe={
                    "requests": PROBE_REQUESTS,
                    "prompt_tokens": PROBE_PROMPT_TOKENS,
                    "output_tokens": PROBE_OUTPUT_TOKENS,
                    "settle_s": PROBE_SETTLE_S,
                    "before": probe_before,
                    "after": probe_after,
                    "cooldown": cooldown,
                },
            ),
            run_dir / "metadata.json",
        )
        return run_dir, cross
    finally:
        power_watch.cancel()
        for owned_client in owned:
            await owned_client.aclose()


class _Samplers:
    """The run's sampler lifecycle: start per repetition, flush per repetition, remember why
    one stopped.

    A sampler runs only when the config asks for it (rate > 0) and the backend can feed it
    (`capabilities`). After the first failure it stays off for the remaining repetitions, with
    the reason recorded once in `errors` and timestamped in that repetition's `events.jsonl`.
    Skipped `/metrics` ticks (transient scrape failures) are not failures; they are counted in
    `server_failures`.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        caps: Capabilities,
        clock: RunClock,
        client: httpx.AsyncClient,
        target: str,
        kind: str,
        progress: Callable[[str], None],
    ) -> None:
        sampler_cfg = cfg.measurement.samplers
        self.gpu_hz = sampler_cfg.gpu_hz
        self.server_hz = sampler_cfg.server_hz
        self._gpu_on = sampler_cfg.gpu_hz > 0 and caps.gpu_metrics
        self._server_on = sampler_cfg.server_hz > 0 and caps.metrics_endpoint
        self._gpu_capable = caps.gpu_metrics
        self._metrics_capable = caps.metrics_endpoint
        self._kind = kind
        self._clock = clock
        self._client = client
        self._base = base_url(target)
        self._progress = progress
        self.active: list[str] = []
        self.errors: dict[str, str] = {}
        self.server_failures = 0

    def plan(self) -> str:
        """One line saying what will sample, or why not — printed before the warm-up."""
        if self.gpu_hz <= 0:
            gpu = "off (gpu_hz=0)"
        elif not self._gpu_capable:
            gpu = f"off ({self._kind} has no gpu_metrics)"
        else:
            gpu = f"{self.gpu_hz:g} Hz (NVML)"
        if self.server_hz <= 0:
            server = "off (server_hz=0)"
        elif not self._metrics_capable:
            server = f"off ({self._kind} has no /metrics)"
        else:
            server = f"{self.server_hz:g} Hz ({self._base}/metrics)"
        return f"samplers: gpu {gpu} · server {server}"

    async def start(
        self,
    ) -> tuple[GpuSampler | None, ServerSampler | None, list[dict[str, Any]]]:
        """`(gpu, server, events)` for the coming repetition."""
        events: list[dict[str, Any]] = []
        gpu: GpuSampler | None = None
        if self._gpu_on:
            gpu_candidate = GpuSampler(self._clock, hz=self.gpu_hz)
            if gpu_candidate.start():
                gpu = gpu_candidate
                if "gpu" not in self.active:
                    self.active.append("gpu")
            else:
                self._gpu_on = False
                self._record("gpu", "NVML unavailable (no GPU samples collected)", events)
        server: ServerSampler | None = None
        if self._server_on:
            server_candidate = ServerSampler(
                self._clock, hz=self.server_hz, base_url=self._base, client=self._client
            )
            try:
                await server_candidate.start()  # its first scrape validates the series names
            except Exception as e:  # missing series or an unreachable /metrics
                self._server_on = False
                self._record("server", describe(e), events)
            else:
                server = server_candidate
                if "server" not in self.active:
                    self.active.append("server")
        return gpu, server, events

    async def stop(
        self,
        gpu: GpuSampler | None,
        server: ServerSampler | None,
        events: list[dict[str, Any]],
        rep_path: Path,
    ) -> dict[str, int]:
        """Flush both samplers into the repetition directory; returns `{"gpu", "server", "events"}`
        row counts for the repetition's progress line."""
        counts = {"gpu": 0, "server": 0, "events": 0}
        if gpu is not None:
            gpu_samples = gpu.stop()
            storage.write_gpu(rep_path / "gpu.parquet", gpu_samples)
            counts["gpu"] = len(gpu_samples)
            if not gpu_samples:
                self._gpu_on = False
                self._record("gpu", "NVML returned no samples during the run", events)
        if server is not None:
            server_samples = await server.stop()
            storage.write_server(rep_path / "server.parquet", server_samples)
            counts["server"] = len(server_samples)
            self.server_failures += server.failures
            if server.error:
                self._server_on = False
                self._record("server", server.error, events)
        storage.write_events(rep_path / "events.jsonl", events)
        counts["events"] = len(events)  # sampler failures land here during the flush
        return counts

    def _record(self, name: str, message: str, events: list[dict[str, Any]]) -> None:
        self.errors[name] = message
        self._progress(f"sampler {name} stopped for the rest of the run: {message}")
        events.append(
            {"t_ns": self._clock(), "kind": "sampler_error", "sampler": name, "error": message}
        )


def _startup_lines() -> list[str]:
    """Interpreter context for the run log: which Python ran the harness, and which
    `sitecustomize` module Python tried to import at startup (a failing one — e.g. a missing
    dependency — prints `Error in sitecustomize` before any of our code runs)."""
    pythonpath = os.environ.get("PYTHONPATH")
    header = f"python {platform.python_version()} @ {sys.executable}"
    lines = [header + (f" · PYTHONPATH={pythonpath}" if pythonpath else "")]
    spec = importlib.util.find_spec("sitecustomize")
    if spec is not None and spec.origin:
        lines.append(f"sitecustomize: {spec.origin}")
    return lines


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
    """Run a small closed-loop workload, discard it, then wait for the server and GPU to settle."""
    requests_n = cfg.measurement.warmup.requests
    if requests_n == 0:
        return {
            "performed": False,
            "requests": 0,
            "ok": 0,
            "gpu_util_pct": None,
            "server_idle": None,
        }
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
    timeout_s = cfg.measurement.warmup.settle_timeout_s
    server_idle = await _server_idle(
        client,
        target,
        caps,
        timeout_s=timeout_s,
        hz=cfg.measurement.samplers.server_hz,
    )
    util = _settle(timeout_s)
    ok_n = sum(1 for r in records if r.ok)
    gpu_txt = f"{util:.0f}%" if util is not None else "n/a (no NVML)"
    progress(f"warm-up done: {ok_n}/{requests_n} ok · server idle {server_idle} · gpu {gpu_txt}")
    return {
        "performed": True,
        "requests": requests_n,
        "ok": ok_n,
        "gpu_util_pct": util,
        "server_idle": server_idle,
    }


async def _cool_probe(
    cfg: ExperimentConfig, **probe_args: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Repeat the "before" probe until it is cool or the cool-down times out; returns the last
    probe and how the wait went. A probe with no TPOT cannot judge heat and ends the wait."""
    progress: Callable[[str], None] = probe_args["progress"]
    start = time.monotonic()
    attempts = 0
    while True:
        probe = await _thermal_probe(cfg, "before", **probe_args)
        attempts += 1
        p50 = probe["tpot_p50_s"]
        cool = None if p50 is None else p50 <= COOL_PROBE_TPOT_S
        waited_s = time.monotonic() - start
        if cool is not False or waited_s >= COOLDOWN_TIMEOUT_S:
            return probe, {
                "limit_tpot_s": COOL_PROBE_TPOT_S,
                "attempts": attempts,
                "waited_s": round(waited_s, 1),
                "cool": cool,
            }
        progress(
            f"Mac still warm (limit {COOL_PROBE_TPOT_S * 1e3:.0f} ms); probing again in "
            f"{COOLDOWN_PAUSE_S:.0f} s"
        )
        await asyncio.sleep(COOLDOWN_PAUSE_S)


async def _thermal_probe(
    cfg: ExperimentConfig,
    when: Literal["before", "after"],
    *,
    client: httpx.AsyncClient,
    caps: Capabilities,
    clock: RunClock,
    target: str,
    builder: PromptBuilder,
    token_counter: Callable[[str], int] | None,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """Pause idle, then send the fixed probe batch; returns each request's TPOT and their
    median. The pause lets the tail of the previous load (warm-up or the last rep) pass, which
    takes seconds; heat that throttles a run lasts minutes and still shows."""
    await asyncio.sleep(PROBE_SETTLE_S)
    probe_cfg = cfg.model_copy(
        update={
            "load": Load(mode="closed", concurrency=1, requests=PROBE_REQUESTS),
            "workload": schema.Workload.model_validate(
                {
                    "classes": [
                        {
                            "name": "thermal_probe",
                            "prompt": {"distribution": "fixed", "tokens": PROBE_PROMPT_TOKENS},
                            "output": {"distribution": "fixed", "tokens": PROBE_OUTPUT_TOKENS},
                        }
                    ]
                }
            ),
            # The experiment's scheduler would make probes incomparable across experiments.
            "gateway": schema.Gateway(),
        }
    )
    # Seed children past the warm-up's: same shapes before and after, but no shared prefixes.
    rep = cfg.repetitions + (1 if when == "before" else 2)
    workload = materialize(probe_cfg, rep=rep, builder=builder)
    records = await drive(
        probe_cfg,
        workload,
        clock=clock,
        client=client,
        caps=caps,
        endpoint=target,
        token_counter=token_counter,
    )
    tpots = [t for t in map(latency.tpot, records) if t is not None]
    p50 = statistics.median(tpots) if tpots else None
    shown = f"{p50 * 1e3:.1f} ms" if p50 is not None else "n/a"
    progress(f"thermal probe {when}: TPOT p50 {shown} ({len(tpots)}/{PROBE_REQUESTS} requests)")
    return {"tpot_s": tpots, "tpot_p50_s": p50, "n": len(tpots)}


async def _server_idle(
    client: httpx.AsyncClient,
    target: str,
    caps: Capabilities,
    *,
    timeout_s: float,
    hz: float,
) -> bool | None:
    """Scrape `/metrics` until `running == waiting == 0` or the settle timeout expires.

    None when the backend has no usable `/metrics`; False when it stayed busy. Uses the scrape
    rate only as a poll interval — settling is a correctness gate, not a sampler.
    """
    if not caps.metrics_endpoint:
        return None
    base = base_url(target)
    interval = 1.0 / hz if hz > 0 else 0.5
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            resp = await client.get(f"{base}/metrics")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        series = parse_prometheus(resp.text)
        running, waiting = lookup(series, "running"), lookup(series, "waiting")
        if running is None or waiting is None:
            return None
        if running <= 0 and waiting <= 0:
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(interval)


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

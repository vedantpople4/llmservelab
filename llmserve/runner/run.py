"""The load driver: fires a materialized workload at the server and records every request
(plan §Phase 2 item 5). Two shapes:

- **closed loop** (`load.mode: closed`): `concurrency` worker tasks pull specs from one shared
  iterator; each worker awaits its request before pulling the next, so at most `concurrency`
  requests are ever in flight. Arrival is the moment a worker takes a spec — there is no
  schedule, so `client_lag_ns` stays None.
- **open loop** (`load.mode: open`): a single arrival loop sleeps until each spec's scheduled
  offset (`t0 + arrival_offset_s`) and fires a send task without waiting for it, so the arrival
  process paces the run, not the server's responses. `t_arrival` is the *scheduled* time, so
  TTFT includes any lateness, and `client_lag_ns` records that lateness separately — the
  harness-overhead metric (against the mock it must stay under 5 ms P99).

`load.requests` caps how many specs are taken (earliest first under open loop);
`load.duration_s` stops the closed loop issuing new work and drops open-loop arrivals at or
after the deadline.

Direct mode: the driver calls the streaming client itself. `max_in_flight` bounding and
scheduling arrive with the gateway (plan item 6); until then an open-loop burst is limited
only by the arrival process — which is the point of open loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable

import httpx

from llmserve.client.capabilities import Capabilities
from llmserve.client.openai_stream import stream_completion
from llmserve.config.schema import ExperimentConfig
from llmserve.metrics.records import RequestRecord
from llmserve.runner.clock import Clock
from llmserve.workload.spec import RequestSpec, Workload


async def drive(
    cfg: ExperimentConfig,
    workload: Workload,
    *,
    clock: Clock,
    client: httpx.AsyncClient,
    caps: Capabilities,
    endpoint: str | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> list[RequestRecord]:
    """Run `workload` under `cfg.load`; returns one record per request, in arrival order."""
    target = endpoint or cfg.server.endpoint
    if cfg.load.mode == "closed":
        records = await _drive_closed(
            cfg,
            workload,
            clock=clock,
            client=client,
            caps=caps,
            endpoint=target,
            token_counter=token_counter,
        )
    else:
        records = await _drive_open(
            cfg,
            workload,
            clock=clock,
            client=client,
            caps=caps,
            endpoint=target,
            token_counter=token_counter,
        )
    records.sort(key=lambda r: (r.t_arrival, r.request_id))
    return records


async def _send(
    cfg: ExperimentConfig,
    client: httpx.AsyncClient,
    caps: Capabilities,
    clock: Clock,
    endpoint: str,
    token_counter: Callable[[str], int] | None,
    spec: RequestSpec,
    t_arrival: int,
) -> RequestRecord:
    return await stream_completion(
        client,
        spec,
        clock,
        endpoint=endpoint,
        model=cfg.server.model,
        t_arrival=t_arrival,
        caps=caps,
        timeout_s=cfg.measurement.request_timeout_s,
        token_counter=token_counter,
    )


async def _drive_closed(
    cfg: ExperimentConfig,
    workload: Workload,
    *,
    clock: Clock,
    client: httpx.AsyncClient,
    caps: Capabilities,
    endpoint: str,
    token_counter: Callable[[str], int] | None,
) -> list[RequestRecord]:
    load = cfg.load
    specs = workload[: load.requests] if load.requests is not None else workload
    shared = iter(specs)
    deadline_ns = None if load.duration_s is None else int(load.duration_s * 1e9)
    records: list[RequestRecord] = []

    async def worker() -> None:
        while True:
            if deadline_ns is not None and clock() >= deadline_ns:
                return
            spec = next(shared, None)
            if spec is None:
                return
            records.append(
                await _send(cfg, client, caps, clock, endpoint, token_counter, spec, clock())
            )

    workers = load.concurrency
    if workers is None:  # unreachable: the schema requires concurrency in closed mode
        raise ValueError("closed-loop load needs `concurrency`")
    await asyncio.gather(*(worker() for _ in range(workers)))
    return records


async def _drive_open(
    cfg: ExperimentConfig,
    workload: Workload,
    *,
    clock: Clock,
    client: httpx.AsyncClient,
    caps: Capabilities,
    endpoint: str,
    token_counter: Callable[[str], int] | None,
) -> list[RequestRecord]:
    load = cfg.load
    specs = sorted(workload, key=lambda s: s.arrival_offset_s)
    if load.requests is not None:
        specs = specs[: load.requests]
    if load.duration_s is not None:
        specs = [s for s in specs if s.arrival_offset_s < load.duration_s]
    t0_ns = clock()

    async def fire(spec: RequestSpec) -> RequestRecord:
        scheduled_ns = t0_ns + int(round(spec.arrival_offset_s * 1e9))
        lag_ns = clock() - scheduled_ns  # actual send start − schedule; >= 0 by construction
        rec = await _send(cfg, client, caps, clock, endpoint, token_counter, spec, scheduled_ns)
        return dataclasses.replace(rec, client_lag_ns=lag_ns)

    tasks: list[asyncio.Task[RequestRecord]] = []
    for spec in specs:
        scheduled_ns = t0_ns + int(round(spec.arrival_offset_s * 1e9))
        now_ns = clock()
        if scheduled_ns > now_ns:
            await asyncio.sleep((scheduled_ns - now_ns) / 1e9)
        tasks.append(asyncio.create_task(fire(spec)))
    return list(await asyncio.gather(*tasks))

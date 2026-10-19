"""Fit the continuous-batching mock's step costs to measured runs (plan Phase 3 item 6).

The aim is qualitative: a calibrated mock should make laptop experiments point the same direction
as real ones (which policy wins, where saturation starts). It is not a claim that the mock
reproduces a GPU's latencies; the observed-vs-predicted table says how far off it is.

The engine's step time is `alpha + beta*prefill_tokens + gamma*decode_seqs + delta*sum(context)`,
where context is each participating sequence's footprint (prompt + tokens generated so far). Two
observations of closed-loop runs are linear in `(alpha, beta, gamma, delta)`:

- **Prefill (concurrency-1 runs).** An idle engine prefills a prompt of `p` tokens in
  `k = ceil(p / min(prefill_chunk, max_batched_tokens))` chunked steps, each paying alpha and
  the context term on the full prompt, so the engine's TTFT is `k·alpha + beta·p + delta·k·p`
  exactly. Measured TTFT adds a fixed per-request overhead `o` (HTTP, tokenization, waking the
  engine) that is fitted as a fifth unknown so it does not inflate alpha; the engine has no
  such term. With every prompt within one chunk, beta and delta are not separable from TTFT
  alone; decode separates them.
- **Decode (every run).** A pure decode step of batch `B` costs `alpha + gamma·B + delta·S`,
  where `S` is the batch's resident context. `S` is approximated by `p + n/2` for the request
  itself plus `(B − 1)` times the run's mean `p + n/2` — mean footprint over a request's decode.
  The observation is each request's *median* inter-token latency, not its TPOT: in a closed loop
  other requests' prefills land inside a request's decode span, and TPOT averages those slower
  mixed steps in (by roughly `(B − 1)·beta·p / (n − 1)`), which would bias gamma upwards. The
  median ITL is a pure decode step as long as fewer than half the steps are mixed (`n > 2B`).

Rows are weighted by `1/observed`, so the least squares minimises relative error and the short
decode steps count as much as long prefills. Identification needs a concurrency-1 run, two
prompt lengths among concurrency-1 requests (TTFT slope, and the context slope of decode at
`B = 1`) and two concurrency levels (gamma). A coefficient that fits negative is reported in
`warnings`, fixed at 0, and the rest refitted (the most negative first, one at a time).

Per-step overhead the engine does not model (timer overshoot, streaming each token) does land in
alpha; that is the honest place for it, since a real engine pays it every step too.

The predicted column comes from the real `ContinuousBatchEngine` with the fitted params, replaying
each repetition's requests (same order, lengths and concurrency) under a virtual clock: the
engine's `asyncio.sleep` steps advance simulated time instead of waiting, so replay is exact to
the engine's step rule and costs CPU only, however slow the measured system was.
"""

from __future__ import annotations

import asyncio
import math
import selectors
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from llmserve.analysis.runs import Run
from llmserve.metrics import latency
from llmserve.metrics.records import RequestRecord
from llmserve.mock.engine import CbParams, ContinuousBatchEngine

COEFFICIENTS = ("alpha_s", "beta_s", "gamma_s", "delta_s")
_UNKNOWNS = (*COEFFICIENTS, "request_overhead_s")


class CalibrationError(ValueError):
    """The runs cannot identify the step costs (the message says what is missing)."""


@dataclass(frozen=True)
class Point:
    """One (run, prompt length, output length) group of measured, successful requests."""

    run: str
    concurrency: int
    prompt_tokens: int
    output_tokens: int
    requests: int
    ttft_s: float
    """Median TTFT."""
    tpot_s: float
    """Median TPOT."""
    itl_s: float
    """Median over requests of each request's median inter-token latency."""
    batch_context_tokens: float
    """`S`: the decode batch's approximate resident context (module docstring)."""


@dataclass(frozen=True)
class Calibration:
    params: CbParams
    request_overhead_s: float
    """Fixed per-request TTFT outside the engine's steps (HTTP, tokenization, the first wake-up).
    Fitted so it does not leak into alpha. The engine has no such term; a mock served over HTTP
    pays its own transport overhead instead, so it is not written into `params`."""
    table: pd.DataFrame
    """Per point: observed vs predicted median TTFT and TPOT, and `pred/obs − 1` for each.
    Predicted TTFT is the engine replay plus `request_overhead_s`."""
    warnings: tuple[str, ...]


def fit_cb_params(
    runs: Sequence[Run],
    *,
    kv_capacity_tokens: int,
    max_batched_tokens: int | None = None,
    prefill_chunk: int | None = None,
) -> Calibration:
    """Fit alpha..delta to closed-loop `runs`, then replay them on the fitted engine."""
    defaults = CbParams()
    shape = replace(
        defaults,
        kv_capacity_tokens=kv_capacity_tokens,
        max_num_batched_tokens=max_batched_tokens or defaults.max_num_batched_tokens,
        prefill_chunk=prefill_chunk or defaults.prefill_chunk,
    )
    points = [p for run in runs for p in points_of(run)]
    fitted, warnings = fit_step_costs(points, shape)
    overhead = fitted["request_overhead_s"]
    params = replace(
        shape,
        alpha_s=fitted["alpha_s"],
        beta_s=fitted["beta_s"],
        gamma_s=fitted["gamma_s"],
        delta_s=fitted["delta_s"],
    )
    predicted = [p for run in runs for p in points_of(run, replay=params)]
    return Calibration(
        params=params,
        request_overhead_s=overhead,
        table=_table(points, predicted, overhead),
        warnings=warnings,
    )


def points_of(run: Run, *, replay: CbParams | None = None) -> list[Point]:
    """`run`'s measured points; with `replay`, the same points as the engine would produce them."""
    load = run.config.load
    if load.mode != "closed" or load.concurrency is None:
        raise CalibrationError(f"{run.run_id}: calibration needs closed-loop runs")
    concurrency = load.concurrency
    groups: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for rep_dir in run.rep_dirs():
        rep = int(rep_dir.name.removeprefix("rep-"))
        measured = [r for r in run.measured(rep) if r.ok]
        if replay is None:
            observed = {r.request_id: _observe(r) for r in measured}
        else:
            observed = _replay(run.records(rep), concurrency, replay)
        for r in measured:
            obs = observed[r.request_id]
            if obs is not None:
                groups.setdefault((r.prompt_tokens_req, r.output_tokens_req), []).append(obs)
    if not groups:
        return []
    weights = {key: len(rows) for key, rows in groups.items()}
    mean_context = sum((p + n / 2) * w for (p, n), w in weights.items()) / sum(weights.values())
    return [
        Point(
            run=run.run_id,
            concurrency=concurrency,
            prompt_tokens=p,
            output_tokens=n,
            requests=len(rows),
            ttft_s=statistics.median(row[0] for row in rows),
            tpot_s=statistics.median(row[1] for row in rows),
            itl_s=statistics.median(row[2] for row in rows),
            batch_context_tokens=p + n / 2 + (concurrency - 1) * mean_context,
        )
        for (p, n), rows in sorted(groups.items())
    ]


def fit_step_costs(
    points: Sequence[Point], shape: CbParams
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Relative least squares for alpha..delta and the request overhead (module docstring)."""
    _check_identifiable(points)
    step_slice = min(shape.prefill_chunk, shape.max_num_batched_tokens)
    rows: list[list[float]] = []
    targets: list[float] = []
    for pt in points:
        if pt.concurrency == 1:
            k = math.ceil(pt.prompt_tokens / step_slice)
            rows.append([k, pt.prompt_tokens, 0.0, k * pt.prompt_tokens, 1.0])
            targets.append(pt.ttft_s)
        rows.append([1.0, 0.0, pt.concurrency, pt.batch_context_tokens, 0.0])
        targets.append(pt.itl_s)
    y = np.asarray(targets)
    design = np.asarray(rows) / y[:, None]
    if np.linalg.matrix_rank(design) < len(_UNKNOWNS):
        raise CalibrationError("the runs do not vary enough to separate alpha, beta, gamma, delta")
    free = list(range(len(_UNKNOWNS)))
    coef = np.zeros(len(_UNKNOWNS))
    warnings: list[str] = []
    while free:
        solution, *_ = np.linalg.lstsq(design[:, free], np.ones_like(y), rcond=None)
        worst = int(np.argmin(solution))
        if solution[worst] >= 0:
            coef[free] = solution
            break
        warnings.append(
            f"{_UNKNOWNS[free[worst]]} fitted negative ({solution[worst]:.3g}); set to 0"
        )
        del free[worst]
    return dict(zip(_UNKNOWNS, (float(c) for c in coef), strict=True)), tuple(warnings)


def _check_identifiable(points: Sequence[Point]) -> None:
    single = [p for p in points if p.concurrency == 1]
    if not single:
        raise CalibrationError("no concurrency-1 run: alpha and beta need TTFT without queueing")
    if len({p.prompt_tokens for p in single}) < 2:
        raise CalibrationError(
            "concurrency-1 runs cover fewer than 2 prompt lengths: beta and delta need a slope"
        )
    if len({p.concurrency for p in points}) < 2:
        raise CalibrationError("fewer than 2 concurrency levels: gamma needs a batch-size slope")


def _observe(r: RequestRecord) -> tuple[float, float, float] | None:
    ttft, tpot, gaps = latency.ttft(r), latency.tpot(r), latency.inter_token_latencies(r)
    if ttft is None or tpot is None or not gaps:
        return None
    return ttft, tpot, statistics.median(gaps)


def _table(
    observed: Sequence[Point], predicted: Sequence[Point], request_overhead_s: float
) -> pd.DataFrame:
    pred = {(p.run, p.prompt_tokens, p.output_tokens): p for p in predicted}
    rows = []
    for o in observed:
        p = pred[(o.run, o.prompt_tokens, o.output_tokens)]
        ttft_pred = p.ttft_s + request_overhead_s
        rows.append(
            {
                "run": o.run,
                "concurrency": o.concurrency,
                "prompt_tokens": o.prompt_tokens,
                "output_tokens": o.output_tokens,
                "requests": o.requests,
                "ttft_obs_s": o.ttft_s,
                "ttft_pred_s": ttft_pred,
                "ttft_rel_err": ttft_pred / o.ttft_s - 1,
                "tpot_obs_s": o.tpot_s,
                "tpot_pred_s": p.tpot_s,
                "tpot_rel_err": p.tpot_s / o.tpot_s - 1,
            }
        )
    return pd.DataFrame(rows)


def _replay(
    records: Sequence[RequestRecord], concurrency: int, params: CbParams
) -> dict[str, tuple[float, float, float] | None]:
    """Run one repetition's requests through the engine closed-loop, in arrival order."""
    order = sorted(records, key=lambda r: r.t_arrival)

    async def drive() -> dict[str, tuple[float, float, float] | None]:
        loop = asyncio.get_running_loop()
        engine = ContinuousBatchEngine(params=params)
        pending = iter(order)
        out: dict[str, tuple[float, float, float] | None] = {}

        async def worker() -> None:
            for r in pending:
                if engine.check_request(r.prompt_tokens_req, r.output_tokens_req) is not None:
                    out[r.request_id] = None  # the mock server would answer 400
                    continue
                start = loop.time()
                stamps = [
                    loop.time()
                    async for _ in engine.stream_tokens(r.prompt_tokens_req, r.output_tokens_req)
                ]
                if len(stamps) < 2:
                    out[r.request_id] = None
                    continue
                gaps = np.diff(stamps)
                out[r.request_id] = (
                    stamps[0] - start,
                    (stamps[-1] - stamps[0]) / (len(stamps) - 1),
                    float(np.median(gaps)),
                )

        try:
            await asyncio.gather(*(worker() for _ in range(concurrency)))
        finally:
            await engine.aclose()
        return out

    loop = _VirtualClockLoop()
    try:
        return loop.run_until_complete(drive())
    finally:
        loop.close()


class _VirtualClockSelector(selectors.SelectSelector):
    """When nothing is ready, jump the clock to the next timer instead of sleeping."""

    def __init__(self) -> None:
        super().__init__()
        self.now = 0.0

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        ready = super().select(0)
        if not ready and timeout is None:
            raise RuntimeError("replay stalled: no timer and nothing ready")
        if not ready and timeout:
            self.now += timeout
        return ready


class _VirtualClockLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        self._clock = _VirtualClockSelector()
        super().__init__(self._clock)

    def time(self) -> float:
        return self._clock.now

"""Phase 3 figures (plan Phase 3 item 3). Each function takes tidy inputs and returns a Figure.

Figures are built on `matplotlib.figure.Figure` directly (Agg, no pyplot state), so building
one has no global side effects and saving it is deterministic. Colors are the Okabe–Ito
palette (colorblind-safe); every axis names its unit; every title says what the figure shows,
and the caller prepends any banner (the report's development-results warning) to it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import ParamSpec

import matplotlib

matplotlib.use("Agg")

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullLocator

from llmserve.scheduler.estimators import PrefillFit

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000")

LATENCY_SERIES = (
    ("latency.ttft_s.p50", "TTFT P50"),
    ("latency.ttft_s.p99", "TTFT P99"),
)
TPOT_SERIES = (("latency.tpot_s.p50", "TPOT P50"),)
THROUGHPUT_SERIES = (("throughput.gen_toks_s", "generated tokens/s"),)

_P = ParamSpec("_P")


def _styled(build: Callable[_P, Figure]) -> Callable[_P, Figure]:
    """Build under the shared style: text and axes read rcParams when they are created."""

    @functools.wraps(build)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Figure:
        with matplotlib.rc_context(
            {
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        ):
            return build(*args, **kwargs)

    return wrapper


@_styled
def latency_vs_concurrency(table: pd.DataFrame, *, title: str) -> Figure:
    """TTFT P50/P99 (s) and TPOT P50 (ms) vs C from a `sweep_table`, with across-rep 95% CIs."""
    fig = _figure(ncols=2, width=11)
    ttft_ax, tpot_ax = fig.axes
    _sweep_lines(ttft_ax, table, LATENCY_SERIES, scale=1.0)
    _sweep_lines(tpot_ax, table, TPOT_SERIES, scale=1e3, color_offset=2)
    ttft_ax.set_ylabel("TTFT (s)")
    tpot_ax.set_ylabel("TPOT (ms)")
    for ax in fig.axes:
        _log2_x(ax, table["value"])
        ax.set_xlabel("concurrency C (closed loop)")
        ax.legend(loc="upper left")
    fig.suptitle(title)
    return fig


@_styled
def throughput_vs_concurrency(table: pd.DataFrame, *, title: str) -> Figure:
    """Generated tok/s vs C from a `sweep_table`, with across-rep 95% CIs."""
    fig = _figure()
    (ax,) = fig.axes
    _sweep_lines(ax, table, THROUGHPUT_SERIES, scale=1.0)
    _log2_x(ax, table["value"])
    ax.set_xlabel("concurrency C (closed loop)")
    ax.set_ylabel("generated tokens/s")
    ax.legend(loc="upper left")
    ax.set_title(title)
    return fig


@_styled
def prompt_length_ttft(points: pd.DataFrame, fit: PrefillFit | None, *, title: str) -> Figure:
    """Per-request TTFT vs prompt length at C = 1 (`prompt_tokens`, `ttft_s`), plus the fit."""
    fig = _figure()
    (ax,) = fig.axes
    ax.scatter(
        points["prompt_tokens"],
        points["ttft_s"],
        s=10,
        alpha=0.4,
        color=PALETTE[0],
        label="request",
    )
    if fit is not None and len(points):
        xs = np.linspace(0, float(points["prompt_tokens"].max()), 200)
        ax.plot(
            xs,
            [fit.predict(x) for x in xs],
            color=PALETTE[3],
            linewidth=2,
            label=f"a + b·p + c·p²  (R² = {fit.r2:.3f})",
        )
    ax.set_xlabel("prompt length p (tokens)")
    ax.set_ylabel("TTFT (s)")
    ax.legend(loc="upper left")
    ax.set_title(title)
    return fig


@_styled
def ttft_cdf(samples: Mapping[str, npt.ArrayLike], *, title: str) -> Figure:
    """Empirical TTFT CDFs, one line per named sample; P95 and P99 marked as horizontal guides."""
    fig = _figure()
    (ax,) = fig.axes
    for i, (name, values) in enumerate(samples.items()):
        x = np.sort(np.asarray(values, dtype=np.float64))
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=PALETTE[i % len(PALETTE)], label=f"{name} (n={len(x)})")
    for q in (0.95, 0.99):
        ax.axhline(q, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("fraction of requests")
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    ax.set_title(title)
    return fig


@_styled
def bursty_timeseries(
    requests: pd.DataFrame, server: pd.DataFrame, *, title: str, bin_s: float = 1.0
) -> Figure:
    """One rep of an open-loop run over time: offered arrivals/s, server queue depth
    (`waiting`), TTFT by arrival time, KV-cache usage. `requests` is `Run.requests()` and
    `server` is `Run.server()`, both already filtered to that rep; time 0 is its first arrival."""
    fig = _figure(nrows=4, height=9, sharex=True)
    arrivals_ax, queue_ax, ttft_ax, kv_ax = fig.axes
    t0 = float(requests["t_arrival"].min()) if len(requests) else 0.0
    arrive_s = (requests["t_arrival"].astype(float) - t0) / 1e9
    if len(arrive_s):
        edges = np.arange(0, float(arrive_s.max()) + bin_s, bin_s)
        counts, _ = np.histogram(arrive_s, bins=edges)
        arrivals_ax.stairs(counts / bin_s, edges, color=PALETTE[0])
    arrivals_ax.set_ylabel("arrivals/s")
    if len(server):
        server_s = server["t_s"].astype(float) - t0 / 1e9
        queue_ax.plot(server_s, server["waiting"], color=PALETTE[1])
        kv_ax.plot(server_s, server["kv_usage"] * 100, color=PALETTE[2])
    queue_ax.set_ylabel("waiting (requests)")
    ok = requests["ok"].to_numpy(dtype=bool) if len(requests) else np.zeros(0, dtype=bool)
    ttft_ax.scatter(arrive_s[ok], requests["ttft_s"][ok], s=6, alpha=0.5, color=PALETTE[3])
    ttft_ax.set_ylabel("TTFT (s)")
    kv_ax.set_ylabel("KV cache used (%)")
    kv_ax.set_xlabel("time since first arrival (s)")
    fig.suptitle(title)
    return fig


def _figure(
    *, nrows: int = 1, ncols: int = 1, width: float = 7, height: float = 4.5, sharex: bool = False
) -> Figure:
    fig = Figure(figsize=(width, height), layout="constrained")
    fig.subplots(nrows, ncols, sharex=sharex, squeeze=False)
    return fig


def _sweep_lines(
    ax: Axes,
    table: pd.DataFrame,
    series: Sequence[tuple[str, str]],
    *,
    scale: float,
    color_offset: int = 0,
) -> None:
    for i, (metric, label) in enumerate(series):
        rows = table[table["metric"] == metric].sort_values("value")
        if rows.empty:
            continue
        color = PALETTE[(i + color_offset) % len(PALETTE)]
        x = rows["value"].to_numpy(dtype=float)
        ax.plot(x, rows["mean"] * scale, marker="o", color=color, label=label)
        ax.fill_between(
            x, rows["lo"] * scale, rows["hi"] * scale, color=color, alpha=0.2, linewidth=0
        )


def _log2_x(ax: Axes, values: pd.Series) -> None:
    ticks = sorted({float(v) for v in values if v > 0})
    if not ticks:
        return
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])

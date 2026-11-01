"""Report figures (plan Phase 3 item 3 and Phase 4). Each takes tidy inputs and returns a Figure.

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
PHASE_LABELS = {
    "gateway_wait_s": "gateway wait",
    "server_wait_s": "server wait",
    "prefill_s": "prefill (unloaded)",
    "decode_s": "decode",
}

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
def time_breakdown(table: pd.DataFrame, *, title: str) -> Figure:
    """Mean seconds per phase as horizontal stacked bars, absolute and as a share of E2E.
    `table` has a `label` column (one bar each, top to bottom) and the `breakdown.PHASES`."""
    fig = _figure(ncols=2, width=12, height=2 + 0.45 * len(table))
    seconds_ax, share_ax = fig.axes
    labels = table["label"].tolist()[::-1]
    values = table[list(PHASE_LABELS)].to_numpy(dtype=float)[::-1]
    totals = values.sum(axis=1, keepdims=True)
    shares = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0) * 100
    for ax, data in ((seconds_ax, values), (share_ax, shares)):
        left = np.zeros(len(labels))
        for i, name in enumerate(PHASE_LABELS.values()):
            ax.barh(labels, data[:, i], left=left, color=PALETTE[i], label=name)
            left += data[:, i]
    seconds_ax.set_xlabel("mean seconds per request")
    share_ax.set_xlabel("share of mean E2E (%)")
    share_ax.set_xlim(0, 100)
    share_ax.set_yticklabels([])
    handles, names = seconds_ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=len(PHASE_LABELS))
    fig.suptitle(title)
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


@_styled
def profile_vs_rate(table: pd.DataFrame, *, title: str) -> Figure:
    """GPU active share (%) and energy per output token (J) vs offered rate, one point per run.
    `table` has `rate_rps`, `gpu_active` (0 to 1) and `joules_per_token`."""
    fig = _figure(ncols=2, width=11)
    gpu_ax, energy_ax = fig.axes
    rows = table.sort_values("rate_rps")
    gpu_ax.plot(rows["rate_rps"], rows["gpu_active"] * 100, marker="o", color=PALETTE[0])
    energy_ax.plot(rows["rate_rps"], rows["joules_per_token"], marker="o", color=PALETTE[1])
    gpu_ax.set_ylabel("GPU active (%)")
    gpu_ax.set_ylim(0, 100)
    energy_ax.set_ylabel("energy per output token (J)")
    energy_ax.set_ylim(bottom=0)
    for ax in fig.axes:
        ax.set_xlabel("offered rate (req/s, Poisson)")
    fig.suptitle(title)
    return fig


@_styled
def profile_timeline(server: pd.DataFrame, power: pd.DataFrame, *, title: str) -> Figure:
    """One rep over time: vLLM's KV usage and waiting queue, GPU frequency, and combined power
    with samples under thermal pressure above Nominal shaded. Both frames are `Run.server()` /
    `Run.power()` filtered to that rep; time 0 is the first sample of either."""
    fig = _figure(nrows=4, height=9, sharex=True)
    kv_ax, queue_ax, freq_ax, power_ax = fig.axes
    starts = [float(f["t_s"].min()) for f in (server, power) if len(f)]
    t0 = min(starts) if starts else 0.0
    if len(server):
        server_s = server["t_s"].astype(float) - t0
        kv_ax.plot(server_s, server["kv_usage"] * 100, color=PALETTE[2])
        queue_ax.plot(server_s, server["waiting"], color=PALETTE[1])
    if len(power):
        power_s = (power["t_s"].astype(float) - t0).to_numpy()
        freq_ax.plot(power_s, power["gpu_freq_mhz"], color=PALETTE[0], linewidth=0.6)
        power_ax.plot(power_s, power["combined_power_mw"] / 1e3, color=PALETTE[3], linewidth=0.6)
        power_ax.fill_between(
            power_s,
            0,
            1,
            where=(power["thermal_pressure"] != "Nominal").to_numpy(),
            transform=power_ax.get_xaxis_transform(),
            color=PALETTE[4],
            alpha=0.15,
            linewidth=0,
            label="thermal pressure above Nominal",
        )
        power_ax.legend(loc="upper right")
    kv_ax.set_ylabel("KV cache used (%)")
    queue_ax.set_ylabel("waiting (requests)")
    freq_ax.set_ylabel("GPU frequency (MHz)")
    power_ax.set_ylabel("CPU + GPU + ANE (W)")
    power_ax.set_xlabel("time since first sample (s)")
    fig.suptitle(title)
    return fig


@_styled
def kv_vs_ttft(frame: pd.DataFrame, *, title: str) -> Figure:
    """TTFT (s) against vLLM's KV usage at dispatch (%), one point per request, by class.
    `frame` is `profile.kv_at_dispatch`."""
    fig = _figure()
    (ax,) = fig.axes
    for i, (name, rows) in enumerate(frame.groupby("workload_class", sort=True)):
        ax.scatter(
            rows["kv_usage"] * 100,
            rows["ttft_s"],
            s=10,
            alpha=0.6,
            color=PALETTE[i % len(PALETTE)],
            label=str(name),
        )
    ax.set_xlabel("KV cache used at dispatch (%)")
    ax.set_ylabel("TTFT (s)")
    ax.set_xlim(0, 100)
    if len(frame):
        ax.legend(loc="upper left")
    fig.suptitle(title)
    return fig


@_styled
def inflight_cap(table: pd.DataFrame, *, title: str) -> Figure:
    """Generated tok/s and P95 TTFT per class against the gateway's in-flight cap.
    `table` is `policies.inflight_cap_table`."""
    fig = _figure(ncols=2, width=11)
    tput_ax, ttft_ax = fig.axes
    tput_ax.plot(table["cap"], table["gen_toks_s"], marker="o", color=PALETTE[0])
    tput_ax.set_ylabel("generated tokens/s")
    tput_ax.set_ylim(bottom=0)
    classes = sorted(c.removeprefix("ttft_p95_") for c in table if c.startswith("ttft_p95_"))
    for i, cls in enumerate(classes):
        ttft_ax.plot(
            table["cap"],
            table[f"ttft_p95_{cls}"],
            marker="o",
            color=PALETTE[i % len(PALETTE)],
            label=cls,
        )
    ttft_ax.set_ylabel("TTFT P95 (s)")
    ttft_ax.set_ylim(bottom=0)
    if classes:
        ttft_ax.legend(loc="upper right")
    for ax in fig.axes:
        _log2_x(ax, table["cap"])
        ax.set_xlabel("gateway max_in_flight")
    fig.suptitle(title)
    return fig


@_styled
def policy_ttft_bars(table: pd.DataFrame, *, title: str) -> Figure:
    """TTFT P95 per class, one bar per policy, one panel per ρ. `table` is
    `policies.class_ttft`; its row order sets the policy order."""
    rhos = sorted(table["rho"].unique())
    fig = _figure(ncols=max(1, len(rhos)), width=4 + 3.5 * len(rhos))
    policies = list(dict.fromkeys(table["policy"]))
    classes = sorted(table["workload_class"].unique())
    width = 0.8 / max(1, len(policies))
    for ax, rho in zip(fig.axes, rhos, strict=False):
        at = table[table["rho"] == rho]
        for i, policy in enumerate(policies):
            rows = at[at["policy"] == policy].set_index("workload_class")
            heights = [rows["p95"].get(c, np.nan) for c in classes]
            x = np.arange(len(classes)) + (i - (len(policies) - 1) / 2) * width
            ax.bar(x, heights, width=width, color=PALETTE[i % len(PALETTE)], label=policy)
        ax.set_xticks(np.arange(len(classes)), classes)
        ax.set_title(f"ρ = {rho:g}")
        ax.set_ylabel("TTFT P95 (s)")
    handles, names = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="outside lower center", ncols=len(policies))
    fig.suptitle(title)
    return fig


@_styled
def slowdown_by_decile(table: pd.DataFrame, *, title: str) -> Figure:
    """Slowdown P95 (solid) and P50 (dotted) by request-size decile, one line per policy, one
    panel per ρ, log y. `table` is `policies.slowdown_deciles`."""
    rhos = sorted(table["rho"].unique())
    fig = _figure(ncols=max(1, len(rhos)), width=4 + 3.5 * len(rhos))
    policies = list(dict.fromkeys(table["policy"]))
    for ax, rho in zip(fig.axes, rhos, strict=False):
        at = table[table["rho"] == rho]
        for i, policy in enumerate(policies):
            rows = at[at["policy"] == policy].sort_values("decile")
            color = PALETTE[i % len(PALETTE)]
            ax.plot(rows["decile"], rows["p95"], marker="o", color=color, label=policy)
            ax.plot(rows["decile"], rows["p50"], linestyle=":", color=color)
        ax.set_yscale("log")
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("size decile (unloaded service time, 1 = smallest)")
        ax.set_ylabel("slowdown (E2E ÷ unloaded service time)")
        ax.set_title(f"ρ = {rho:g}")
    handles, names = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="outside lower center", ncols=len(policies))
    fig.suptitle(title)
    return fig

"""Figures: the right series on labelled axes, and they save to PNG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import StepPatch

from llmserve.analysis import plots
from llmserve.scheduler.estimators import PrefillFit


def sweep_table() -> pd.DataFrame:
    rows = []
    for c in (1, 2, 4, 8):
        for metric, base in (
            ("throughput.gen_toks_s", 100.0 * c),
            ("latency.ttft_s.p50", 0.1 * c),
            ("latency.ttft_s.p99", 0.3 * c),
            ("latency.tpot_s.p50", 0.02),
        ):
            rows.append(
                {
                    "label": f"c{c}",
                    "value": float(c),
                    "metric": metric,
                    "mean": base,
                    "lo": base * 0.9,
                    "hi": base * 1.1,
                    "n": 5,
                }
            )
    return pd.DataFrame(rows)


def values(data: object) -> list[float]:
    return [float(v) for v in np.asarray(data, dtype=float)]


def label(artist: Artist) -> str:
    return str(artist.get_label())


def saves(fig: Figure, tmp_path: Path) -> None:
    out = tmp_path / "fig.png"
    fig.savefig(out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_latency_vs_concurrency(tmp_path: Path) -> None:
    fig = plots.latency_vs_concurrency(sweep_table(), title="Latency")
    ttft, tpot = fig.axes
    assert [line.get_label() for line in ttft.get_lines()] == ["TTFT P50", "TTFT P99"]
    assert [line.get_label() for line in tpot.get_lines()] == ["TPOT P50"]
    assert ttft.get_ylabel() == "TTFT (s)" and tpot.get_ylabel() == "TPOT (ms)"
    assert ttft.get_xscale() == "log" and "concurrency" in ttft.get_xlabel()
    assert values(tpot.get_lines()[0].get_ydata()) == [20.0] * 4  # seconds → ms
    assert sum(isinstance(c, PolyCollection) for c in ttft.collections) == 2  # CI bands
    assert fig.get_suptitle() == "Latency"
    saves(fig, tmp_path)


def test_throughput_vs_concurrency(tmp_path: Path) -> None:
    fig = plots.throughput_vs_concurrency(sweep_table(), title="Throughput")
    (ax,) = fig.axes
    (line,) = ax.get_lines()
    assert values(line.get_xdata()) == [1, 2, 4, 8]
    assert ax.get_ylabel() == "generated tokens/s" and ax.get_title() == "Throughput"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "4", "8"]
    saves(fig, tmp_path)


def test_prompt_length_ttft_draws_points_and_the_fit(tmp_path: Path) -> None:
    points = pd.DataFrame({"prompt_tokens": [128.0, 512.0, 2048.0], "ttft_s": [0.1, 0.3, 1.1]})
    fit = PrefillFit(a=0.05, b=5e-4, c=0.0, r2=0.99, n=3)
    fig = plots.prompt_length_ttft(points, fit, title="Prefill")
    (ax,) = fig.axes
    assert len(ax.collections) == 1 and len(ax.get_lines()) == 1
    assert "R² = 0.990" in label(ax.get_lines()[0])
    assert ax.get_xlabel() == "prompt length p (tokens)"
    no_fit = plots.prompt_length_ttft(points, None, title="Prefill")
    assert no_fit.axes[0].get_lines() == []
    saves(fig, tmp_path)


def test_ttft_cdf_one_step_per_nonempty_sample(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    fig = plots.ttft_cdf(
        {"mixed": rng.exponential(1, 50), "homogeneous": rng.exponential(1, 30), "none": []},
        title="CDF",
    )
    (ax,) = fig.axes
    steps = [line for line in ax.get_lines() if not label(line).startswith("_")]
    assert [label(line) for line in steps] == ["mixed (n=50)", "homogeneous (n=30)"]
    assert values(steps[0].get_ydata())[-1] == 1.0
    assert len(values(steps[1].get_xdata())) == 30
    saves(fig, tmp_path)


def test_bursty_timeseries_panels(tmp_path: Path) -> None:
    t = np.array([0.0, 0.5, 1.2, 1.3, 1.4, 2.5]) * 1e9 + 7e9
    requests = pd.DataFrame(
        {
            "t_arrival": t,
            "ok": [True, True, True, False, True, True],
            "ttft_s": [0.1, 0.1, 0.4, np.nan, 0.6, 0.2],
        }
    )
    server = pd.DataFrame(
        {"t_s": [7.0, 8.0, 9.0], "waiting": [0, 3, 0], "kv_usage": [0.1, 0.5, 0.1]}
    )
    fig = plots.bursty_timeseries(requests, server, title="Bursty")
    arrivals, queue, ttft, kv = fig.axes
    (stairs,) = arrivals.patches
    assert isinstance(stairs, StepPatch)
    assert values(stairs.get_data()[0]) == [2, 3, 1]  # 1 s bins
    assert values(queue.get_lines()[0].get_xdata()) == [0.0, 1.0, 2.0]  # from first arrival
    assert values(kv.get_lines()[0].get_ydata()) == [10.0, 50.0, 10.0]
    assert len(np.asarray(ttft.collections[0].get_offsets())) == 5  # failed request not plotted
    assert kv.get_ylabel() == "KV cache used (%)"
    saves(fig, tmp_path)
    empty = plots.bursty_timeseries(requests, pd.DataFrame(), title="Bursty")
    assert empty.axes[1].get_lines() == []


def test_time_breakdown_stacks_phases_to_e2e_and_to_100_percent(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "label": ["e01 c", "e02 c"],
            "gateway_wait_s": [0.0, 1.0],
            "server_wait_s": [0.0, 2.0],
            "prefill_s": [0.5, 0.5],
            "decode_s": [1.5, 4.5],
        }
    )
    fig = plots.time_breakdown(table, title="Time")
    seconds, share = fig.axes
    for ax, totals in ((seconds, [8.0, 2.0]), (share, [100.0, 100.0])):
        last = ax.containers[-1]  # bars are drawn bottom-up: e02 first, then e01
        assert [p.get_x() + p.get_width() for p in last] == totals
    assert [t.get_text() for t in fig.legends[0].get_texts()] == list(plots.PHASE_LABELS.values())
    saves(fig, tmp_path)

"""The engine and power profile: vLLM's split from its totals, energy from power samples, and KV
usage at each request's dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import RunFactory, experiment_config
from test_analysis_suites import write_manifest

from llmserve.analysis.profile import kv_at_dispatch, profile
from llmserve.analysis.report import build_report
from llmserve.analysis.runs import Run
from llmserve.metrics.power import PowerSample
from llmserve.metrics.server import ServerSample
from llmserve.runner import storage

S = 1_000_000_000


def server_sample(t_ns: int, *, kv: float = 0.0, **totals: float) -> ServerSample:
    return ServerSample(
        t_ns=t_ns,
        running=1.0,
        waiting=0.0,
        kv_usage=kv,
        preemptions_total=0.0,
        prompt_tokens_total=0.0,
        generation_tokens_total=0.0,
        **totals,
    )


def timing(finished: float, queue: float, prefill: float, decode: float) -> dict[str, float]:
    return {
        "finished_total": finished,
        "queue_time_s_total": queue,
        "prefill_time_s_total": prefill,
        "decode_time_s_total": decode,
        "steps_total": 10 * finished,
        "step_tokens_total": 40 * finished,
    }


def power_sample(t_s: float, watts: float, pressure: str = "Nominal") -> PowerSample:
    return PowerSample(
        t_ns=int(t_s * S),
        elapsed_ns=S // 2,
        gpu_active_ratio=0.5,
        gpu_freq_mhz=1000.0,
        cpu_power_mw=0.0,
        gpu_power_mw=watts * 1e3,
        ane_power_mw=0.0,
        combined_power_mw=watts * 1e3,
        thermal_pressure=pressure,
    )


def test_engine_split_pools_each_reps_change_in_the_totals(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config())
    rep0, rep1 = run.rep_dirs()
    storage.write_server(
        rep0 / "server.parquet",
        [
            server_sample(1 * S, **timing(100, 50, 70, 900)),
            server_sample(2 * S, **timing(102, 52, 71, 910)),
        ],
    )
    storage.write_server(
        rep1 / "server.parquet",
        [
            server_sample(3 * S, **timing(102, 52, 71, 910)),
            server_sample(4 * S, **timing(104, 58, 75, 920)),
        ],
    )
    engine = profile(run).engine
    assert engine is not None
    assert (engine.finished, engine.queue_s, engine.prefill_s, engine.decode_s) == (
        4,
        2.0,
        1.25,
        5.0,
    )
    assert engine.tokens_per_step == 4.0


def test_engine_split_is_none_without_vllms_timing_totals(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config(repetitions=1))
    (rep,) = run.rep_dirs()
    storage.write_server(rep / "server.parquet", [server_sample(1 * S), server_sample(2 * S)])
    p = profile(run)
    assert p.engine is None
    assert p.power is None  # no power.parquet off a Mac


def test_energy_is_power_times_elapsed_over_generated_tokens(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config(repetitions=1))
    (rep,) = run.rep_dirs()
    samples = [power_sample(1, 10.0), power_sample(1.5, 20.0, "Heavy"), power_sample(2, 30.0)]
    storage.write_power(rep / "power.parquet", samples)
    power = profile(run).power
    assert power is not None
    tokens = sum(r.output_tokens_usage or 0 for r in run.records(0) if r.ok)
    assert tokens > 0
    assert power.power_w == pytest.approx(20.0)
    assert power.joules_per_token == pytest.approx(30.0 / tokens)  # 1.5 s at a mean 20 W
    assert power.throttled_share == pytest.approx(1 / 3)
    assert (power.gpu_active, power.gpu_freq_mhz) == (0.5, 1000.0)


def test_kv_at_dispatch_takes_the_last_sample_at_or_before_dispatch(
    run_factory: RunFactory,
) -> None:
    run = run_factory(experiment_config(repetitions=1))
    (rep,) = run.rep_dirs()
    measured = sorted(run.measured(0), key=lambda r: r.t_dispatch or 0)
    dispatches = [r.t_dispatch or 0 for r in measured]
    first, split = dispatches[0], dispatches[len(dispatches) // 2]
    storage.write_server(
        rep / "server.parquet",
        [server_sample(first + 1, kv=0.2), server_sample(split, kv=0.7)],
    )
    frame = kv_at_dispatch(run)
    expected = [0.7 if t >= split else 0.2 for t in dispatches if t >= first + 1]
    assert sorted(frame["kv_usage"]) == sorted(expected)  # the first request predates any sample
    assert len(frame) == len(measured) - 1
    assert frame["ttft_s"].notna().all()


def test_kv_at_dispatch_is_empty_without_server_samples(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config(repetitions=1))
    (run.rep_dirs()[0] / "server.parquet").unlink()
    assert kv_at_dispatch(run).empty


def _profiled(run: Run, *, kv: float) -> Run:
    rep = run.rep_dirs()[0]
    storage.write_server(
        rep / "server.parquet",
        [
            server_sample(0, kv=kv, **timing(0, 0, 0, 0)),
            server_sample(10 * S, kv=kv, **timing(10, 5, 8, 50)),
        ],
    )
    storage.write_power(rep / "power.parquet", [power_sample(1, 8.0), power_sample(2, 9.0)])
    return run


def test_report_profile_section_tabulates_and_plots_both_suites(
    run_factory: RunFactory, tmp_path: Path
) -> None:
    open_load: dict[str, Any] = {
        "mode": "open",
        "arrival": {"process": "poisson", "rate": 20},
        "duration_s": 1.0,
    }
    root = tmp_path / "results"
    p128 = _profiled(
        run_factory(
            experiment_config(experiment="profile-p128-r20", repetitions=1, load=open_load)
        ),
        kv=0.1,
    )
    mixed = _profiled(
        run_factory(
            experiment_config(experiment="profile-mixed-r20", repetitions=1, load=open_load)
        ),
        kv=0.5,
    )
    write_manifest(root, "profile-p128", "load.arrival.rate", [("r20", 20, p128, "done")])
    write_manifest(root, "profile-mixed", "none", [("r20", None, mixed, "done")])

    out = tmp_path / "out"
    md = build_report(root, out, dev=True).read_text()
    section = md.split("## Engine and power profile", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| profile-")]
    assert [row.split(" | ")[:4] for row in rows] == [
        ["| profile-p128", "r20", "20", "10"],
        ["| profile-mixed", "r20", "20", "10"],
    ]
    cells = [c.strip() for c in rows[0].strip("|").split("|")]
    assert cells[4:8] == ["0.50", "0.80", "5.0", "4.0"]  # queue, prefill, decode, tokens/step
    assert cells[9] == "10%"  # peak KV
    for name in ("profile_vs_rate", "profile_timeline", "kv_vs_ttft"):
        assert (out / "figures" / f"{name}.png").is_file()
        assert f"(figures/{name}.png)" in section


def test_report_profile_section_without_suites_names_the_commands(
    run_factory: RunFactory, tmp_path: Path
) -> None:
    run_factory(experiment_config(experiment="e01", repetitions=1))
    md = build_report(tmp_path / "results", tmp_path / "out", dev=True).read_text()
    section = md.split("## Engine and power profile", 1)[1].split("\n## ", 1)[0]
    assert "`llmserve suite configs/suites/profile_p128.yaml`" in section
    assert "`llmserve suite configs/suites/profile_mixed.yaml`" in section

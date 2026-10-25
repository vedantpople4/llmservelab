"""Reading runs back from disk: the analysis side must see what the runner summarized."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from conftest import RunFactory, experiment_config

from llmserve.analysis.aggregate import summarize, window_of
from llmserve.analysis.runs import Run, find_runs, latest_run


def test_rebuilt_records_reproduce_every_stored_summary(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config())
    for rep, stored in enumerate(run.summary["per_rep"]):
        records = run.records(rep)
        again = summarize(records, rep=rep, window=window_of(records, run.config, t0_ns=0))
        assert again == stored


def test_open_loop_window_is_recovered_from_the_summary(run_factory: RunFactory) -> None:
    cfg = experiment_config(
        repetitions=1,
        load={"mode": "open", "arrival": {"process": "poisson", "rate": 40}, "duration_s": 1.5},
        measurement={"warmup": {"requests": 0}, "warmup_s": 0.4, "cooldown_s": 0.4},
    )
    run = run_factory(cfg)
    frame = run.requests()
    rep_summary = run.summary["per_rep"][0]
    assert 0 < int(frame["in_window"].sum()) < len(frame)  # the trim bit on both ends
    assert int(frame["in_window"].sum()) == rep_summary["n_measured"]
    assert len(run.measured(0)) == rep_summary["n_measured"]


def test_requests_frame_has_latency_in_seconds(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config())
    frame = run.requests()
    assert sorted(frame["rep"].unique()) == [0, 1]
    assert len(frame) == 2 * 12
    assert frame["ok"].all()
    measured = frame[frame["in_window"] & (frame["rep"] == 0)]
    p50 = run.summary["per_rep"][0]["latency"]["ttft_s"]["p50"]
    assert math.isclose(float(measured["ttft_s"].median()), p50, rel_tol=1e-9)


def test_mock_runs_are_never_reportable(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config())
    assert run.config.server.kind == "mock"
    assert run.reportable is False


def _written_run(
    root: Path,
    *,
    dirty: bool = False,
    invalid: bool = False,
    tpots: tuple[float, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> Run:
    run_dir = root / "e01" / "20261019T000000Z-aaaaaaaa-bbbbbbb"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text(
        "schema_version: 1\nexperiment: e01\nseed: 1\n"
        "server: {kind: vllm-metal, endpoint: 'http://x/v1', model: m}\n"
        "load: {mode: closed, concurrency: 1, requests: 1}\n"
        "workload: {classes: [{name: c, prompt: {distribution: fixed, tokens: 1},"
        " output: {distribution: fixed, tokens: 1}}]}\n"
    )
    per_rep = [{"latency": {"tpot_s": {"p50": t}}} for t in tpots]
    (run_dir / "metadata.json").write_text(json.dumps({"git_dirty": dirty, **(metadata or {})}))
    (run_dir / "summary.json").write_text(json.dumps({"invalid": invalid, "per_rep": per_rep}))
    return Run(run_dir)


def test_reportable_needs_vllm_metal_a_clean_tree_and_a_valid_summary(tmp_path: Path) -> None:
    cases = ((False, False, True), (True, False, False), (False, True, False))
    for dirty, invalid, expected in cases:
        assert _written_run(tmp_path, dirty=dirty, invalid=invalid).reportable is expected


def _probes(before: float, after: float) -> dict[str, Any]:
    return {
        "thermal_probe": {
            "before": {"tpot_s": [before] * 4, "tpot_p50_s": before, "n": 4},
            "after": {"tpot_s": [after] * 4, "tpot_p50_s": after, "n": 4},
        }
    }


def test_more_than_ten_percent_probe_drift_is_unreportable(tmp_path: Path) -> None:
    run = _written_run(tmp_path, tpots=(0.020, 0.020, 0.020), metadata=_probes(0.020, 0.0222))
    assert math.isclose(run.thermal_drift or 0.0, 0.11)
    assert run.unreportable_reasons == ["thermal drift +11% (probe after vs before, TPOT p50)"]


def test_a_run_that_starts_on_a_slowed_machine_is_unreportable(tmp_path: Path) -> None:
    run = _written_run(tmp_path, metadata=_probes(0.0795, 0.0566))
    assert run.unreportable_reasons == ["thermal drift -29% (probe after vs before, TPOT p50)"]


def test_probe_drift_under_ten_percent_is_reportable(tmp_path: Path) -> None:
    run = _written_run(tmp_path, metadata=_probes(0.020, 0.0218))
    assert math.isclose(run.thermal_drift or 0.0, 0.09)
    assert run.reportable is True


def test_reps_that_differ_in_tpot_are_not_drift_when_the_probes_agree(tmp_path: Path) -> None:
    # Session 1's bursty run: each rep sampled a different long-context mix.
    run = _written_run(tmp_path, tpots=(0.301, 0.355, 0.408), metadata=_probes(0.020, 0.0204))
    assert math.isclose(run.thermal_drift or 0.0, 0.02)
    assert run.reportable is True


def test_a_run_without_probes_has_no_drift_evidence(tmp_path: Path) -> None:
    run = _written_run(tmp_path, tpots=(0.020, 0.030))
    assert run.thermal_drift is None
    assert run.reportable is True


def test_a_run_that_slept_more_than_a_second_is_unreportable(tmp_path: Path) -> None:
    run = _written_run(tmp_path, metadata={"sleep": {"total_s": 3612.4, "per_rep_s": [3612.4]}})
    assert run.unreportable_reasons == ["Mac slept 3612 s during the run"]
    assert _written_run(tmp_path, metadata={"sleep": {"total_s": 0.8}}).reportable is True


def test_a_run_that_started_or_ended_on_battery_is_unreportable(tmp_path: Path) -> None:
    ac, batt = "AC Power", "Battery Power"
    for start, end in ((batt, ac), (ac, batt), (batt, batt)):
        run = _written_run(tmp_path, metadata={"power_source": {"start": start, "end": end}})
        assert run.unreportable_reasons == ["ran on battery power"]
    on_ac = _written_run(tmp_path, metadata={"power_source": {"start": ac, "end": ac}})
    assert on_ac.reportable is True


def test_unknown_sleep_and_power_are_not_reasons(tmp_path: Path) -> None:
    unknown = {"sleep": {"total_s": None}, "power_source": {"start": None, "end": None}}
    assert _written_run(tmp_path, metadata=unknown).reportable is True
    assert _written_run(tmp_path).reportable is True  # a run from before these fields existed


def test_rep_metric_reads_one_value_per_rep(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config(repetitions=3))
    values = run.rep_metric("latency.ttft_s.p95")
    assert len(values) == 3
    assert values == [r["latency"]["ttft_s"]["p95"] for r in run.summary["per_rep"]]
    assert run.rep_metric("latency.no_such_metric.p95") == []


def test_find_and_latest_run(run_factory: RunFactory, tmp_path: Path) -> None:
    first = run_factory(experiment_config(experiment="alpha", seed=1))
    second = run_factory(experiment_config(experiment="alpha", seed=2))
    run_factory(experiment_config(experiment="beta"))
    root = tmp_path / "results"
    assert [r.experiment for r in find_runs(root)].count("alpha") == 2
    assert {r.path for r in find_runs(root, "alpha")} == {first.path, second.path}
    latest = latest_run(root, "alpha")
    assert latest is not None and latest.path == max(first.path, second.path, key=lambda p: p.name)
    assert latest_run(root, "missing") is None

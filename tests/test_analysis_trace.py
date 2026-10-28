"""`llmserve trace`: one request's timeline from the files a run stored."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import RunFactory, experiment_config

from llmserve.analysis.trace import trace
from llmserve.cli import main
from llmserve.scheduler.estimators import CostModel, DecodeFit, PrefillFit

PREFILL = PrefillFit(a=0.0, b=0.0, c=0.0, r2=1.0, n=3)


def events_line(path: Path, **event: object) -> None:
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def test_timeline_lists_every_step_in_order(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config())
    text = trace(run, "r000003", rep=1, prefill=PREFILL)
    assert text.startswith(f"run {run.run_id} · rep 1 · request r000003 · class c · ok")
    steps = [line.split()[1] for line in text.splitlines()[4:]]
    assert steps == ["arrival", "dispatch", "first", "last", "complete"]
    assert "gateway wait" in text and "unloaded prefill 0.0 ms" in text
    assert "vLLM running" in text  # the mock's /metrics, sampled before the first request


def test_events_for_the_request_or_during_it_are_shown(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config(repetitions=1))
    (r,) = [r for r in run.records(0) if r.request_id == "r000002"]
    assert r.t_complete is not None
    path = run.path / "rep-00" / "events.jsonl"
    events_line(path, t_ns=r.t_arrival, kind="decision", request_id="r000002")
    events_line(path, t_ns=r.t_arrival, kind="decision", request_id="r000009")
    events_line(path, t_ns=r.t_complete, kind="sampler_error", sampler="server")
    events_line(path, t_ns=r.t_complete + 10**12, kind="sampler_error", sampler="gpu")
    text = trace(run, "r000002")
    shown = text.split("events:\n", 1)[1]
    assert '"request_id": "r000002"' in shown
    assert '"sampler": "server"' in shown
    assert "r000009" not in shown and '"gpu"' not in shown


def test_unknown_request_is_a_key_error(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config())
    with pytest.raises(KeyError, match="no request 'nope' in rep 0"):
        trace(run, "nope")


def test_cli_trace(
    run_factory: RunFactory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_factory(experiment_config())
    model = tmp_path / "cost_model.json"
    CostModel(prefill=PREFILL, decode=DecodeFit(d0=0.0, d1=0.0, r2=1.0, n=2)).save(model)
    with pytest.raises(SystemExit) as ok:
        main(["trace", str(run.path), "--request", "r000000", "--cost-model", str(model)])
    assert ok.value.code == 0
    assert "server wait" in capsys.readouterr().out
    with pytest.raises(SystemExit) as missing:
        main(["trace", str(run.path), "--request", "nope"])
    assert missing.value.code == 2
    assert "no request 'nope'" in capsys.readouterr().err

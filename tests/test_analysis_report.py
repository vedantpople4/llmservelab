"""`build_report` end to end on real run directories (in-process mock) and fixture manifests."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import experiment_config
from matplotlib.figure import Figure
from test_analysis_suites import closed, write_manifest
from test_experiment import run_in_process

from llmserve.analysis.report import DEV_BANNER, build_report, describe_run
from llmserve.analysis.runs import Run
from llmserve.mock import DelayModel, MockEngine
from llmserve.scheduler.estimators import CostModel

DELAY = DelayModel(
    ttft_base_s=0.004, ttft_per_prompt_token_s=5e-5, token_base_s=0.001, token_in_flight_s=2e-4
)
MISSING = ("concurrency-mixed", "output-length", "capacity-mixed", "capacity-interactive")


def _classes(*names: str) -> dict[str, Any]:
    shapes = {
        "interactive": {"distribution": "uniform", "min": 16, "max": 64},
        "long_context": {"distribution": "fixed", "tokens": 512},
    }
    return {
        "classes": [
            {
                "name": n,
                "prompt": shapes[n],
                "output": {"distribution": "fixed", "tokens": 4},
            }
            for n in names
        ]
    }


def _open(arrival: dict[str, Any], duration_s: float) -> dict[str, Any]:
    return {"mode": "open", "arrival": arrival, "duration_s": duration_s}


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("report")
    root = base / "results"

    def run(experiment: str, **overrides: Any) -> Run:
        overrides.setdefault("repetitions", 1)
        cfg = experiment_config(experiment=experiment, **overrides)
        engine = MockEngine(delay=DELAY)
        run_dir, _ = run_in_process(cfg, base, allow_dirty=True, engine=engine)
        return Run(run_dir)

    conc = [
        (
            f"c{c:03d}",
            float(c),
            run(f"concurrency-p128-c{c:03d}", repetitions=2, load=closed(c)),
            "done",
        )
        for c in (1, 2, 4)
    ]
    write_manifest(
        root, "concurrency-p128", "load.concurrency", [*conc, ("c008", 8, None, "failed")]
    )
    prompt = []
    for tokens in (32, 128, 512):
        wl = {
            "classes": [
                {
                    "name": "c",
                    "prompt": {"distribution": "fixed", "tokens": tokens},
                    "output": {"distribution": "fixed", "tokens": 2},
                }
            ]
        }
        r = run(f"prompt-length-p{tokens:05d}", load=closed(1, requests=6), workload=wl)
        prompt.append((f"p{tokens:05d}", float(tokens), r, "done"))
    write_manifest(root, "prompt-length", "workload.classes.0.prompt.tokens", prompt)
    poisson = {"process": "poisson", "rate": 30}
    window = {"warmup": {"requests": 0}, "warmup_s": 0.2, "cooldown_s": 0.2}
    mixed = run(
        "h2-mixed",
        load=_open(poisson, 1.5),
        measurement=window,
        workload=_classes("interactive", "long_context"),
    )
    homo = run(
        "h2-interactive",
        load=_open(poisson, 1.5),
        measurement=window,
        workload=_classes("interactive"),
    )
    write_manifest(root, "h2-mixed", "none", [("rho085", 0.85, mixed, "done")])
    write_manifest(root, "h2-interactive", "none", [("rho085", 0.85, homo, "done")])
    phases = [
        {"duration_s": 0.4, "rate": 10},
        {"duration_s": 0.4, "rate": 60},
        {"duration_s": 0.4, "rate": 10},
    ]
    bursty = run("bursty", load=_open({"process": "bursty", "phases": phases}, 1.2))
    write_manifest(root, "bursty", "none", [("burst", 0, bursty, "done")])
    cap = run(
        "capacity-p128-r020",
        load=_open({"process": "poisson", "rate": 20}, 1.0),
        measurement=window,
    )
    write_manifest(root, "capacity-p128", "load.arrival.rate", [("r020", 20, cap, "done")])
    return root


def _runs(root: Path) -> list[Run]:
    return [Run(p.parent) for p in sorted(root.glob("*/*/summary.json"))]


@pytest.fixture
def saved_titles(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    titles: list[str] = []
    original = Figure.savefig

    def record(self: Figure, *args: Any, **kwargs: Any) -> None:
        texts = [self.get_suptitle(), *(ax.get_title() for ax in self.axes)]
        titles.append("\n".join(t for t in texts if t))
        original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    yield titles


def test_mock_runs_are_excluded_by_default(results: Path, tmp_path: Path) -> None:
    md = build_report(results, tmp_path / "out").read_text()
    assert DEV_BANNER not in md
    excluded = md.split("## Excluded runs")[1].split("## Provenance")[0]
    for run in _runs(results):
        assert f"`{run.run_id}`" in excluded
    assert "server.kind is mock, not vllm" in excluded
    for hid in ("H1", "H2", "H3"):
        assert f"| {hid} |" in md and "**not-run**" in md.split(f"| {hid} |")[1].split("\n")[0]
    assert "`concurrency-p128` has no reportable runs" in md
    assert not list((tmp_path / "out" / "figures").glob("*.png"))
    assert not (tmp_path / "out" / "cost_model.json").exists()


def test_missing_suites_name_the_command_to_run(results: Path, tmp_path: Path) -> None:
    md = build_report(results, tmp_path / "out", dev=True).read_text()
    for suite in MISSING:
        command = f"llmserve suite configs/suites/{suite.replace('-', '_')}.yaml"
        assert f"Not yet run. Run it with `{command}`" in md


def test_dev_mode_banners_the_report_and_every_figure(
    results: Path, tmp_path: Path, saved_titles: list[str]
) -> None:
    out = tmp_path / "out"
    md = build_report(results, out, dev=True).read_text()
    assert md.splitlines()[2] == f"> **{DEV_BANNER}**"
    figures = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert figures == [
        "bursty_timeseries.png",
        "h2_ttft_cdf.png",
        "latency_vs_concurrency_concurrency-p128.png",
        "prompt_length_ttft_prompt-length.png",
        "throughput_vs_concurrency_concurrency-p128.png",
        "ttft_cdf_workloads.png",
    ]
    assert len(saved_titles) == len(figures)
    assert all(DEV_BANNER in t for t in saved_titles)
    for name in figures:
        assert f"(figures/{name})" in md
    assert "**not-run**" not in md.split("## Capacity")[0]  # every hypothesis had its inputs
    assert "every rate sustained" in md  # one capacity point: not bracketed, says so
    assert "None: development mode includes every run" in md


def test_dev_mode_writes_a_loadable_cost_model(results: Path, tmp_path: Path) -> None:
    build_report(results, tmp_path / "out", dev=True)
    model = CostModel.load(tmp_path / "out" / "cost_model.json")
    assert model.prefill.n == 18  # 3 prompt lengths × 6 requests
    assert model.decode.n == 6  # 3 concurrencies × 2 reps


def test_provenance_lists_every_figure_run(results: Path, tmp_path: Path) -> None:
    md = build_report(results, tmp_path / "out", dev=True).read_text()
    provenance = md.split("## Provenance")[1]
    for run in _runs(results):
        assert f"`{run.run_id}`" in provenance
        assert f"`{run.metadata['config_hash'][:12]}`" in provenance
    assert "`configs/suites/concurrency_p128.yaml`" in provenance


def test_rebuild_is_byte_identical_and_drops_stale_outputs(results: Path, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    build_report(results, a, dev=True)
    build_report(results, b, dev=True)
    assert (a / "baseline.md").read_bytes() == (b / "baseline.md").read_bytes()
    assert (a / "cost_model.json").read_bytes() == (b / "cost_model.json").read_bytes()
    for png in sorted((a / "figures").glob("*.png")):
        assert png.read_bytes() == (b / "figures" / png.name).read_bytes(), png.name
    (a / "figures" / "stale.png").write_bytes(b"old")
    build_report(results, a, dev=False)
    assert not list((a / "figures").glob("*.png"))
    assert not (a / "cost_model.json").exists()


def test_reportable_runs_are_kept_and_others_excluded(results: Path, tmp_path: Path) -> None:
    root = tmp_path / "results"
    shutil.copytree(results, root)
    for path in sorted(root.glob("concurrency-p128-*/*")):
        config = path / "config.resolved.yaml"
        config.write_text(config.read_text().replace("kind: mock", "kind: vllm"))
        metadata = json.loads((path / "metadata.json").read_text())
        (path / "metadata.json").write_text(json.dumps({**metadata, "git_dirty": False}))
    for run in _runs(root):
        assert run.reportable == (not run.unreportable_reasons)
    out = tmp_path / "out"
    md = build_report(root, out).read_text()
    h1_row = md.split("| H1 |")[1].split("\n")[0]
    assert "**not-run**" not in h1_row
    assert (out / "figures" / "latency_vs_concurrency_concurrency-p128.png").is_file()
    excluded = md.split("## Excluded runs")[1].split("## Provenance")[0]
    assert "concurrency-p128 |" not in excluded
    assert "| h2-mixed |" in excluded


def test_describe_run_summarizes_one_run(results: Path) -> None:
    run = next(r for r in _runs(results) if r.experiment == "concurrency-p128-c002")
    text = describe_run(run)
    total = sum(s["n_requests"] for s in run.summary["per_rep"])
    assert run.run_id in text
    assert f"reps 2; requests {total}, ok {total}, failed 0 (none)" in text
    assert "valid: yes" in text
    assert "reportable: no (server.kind is mock, not vllm" in text
    for label in ("TTFT", "TPOT", "E2E", "gen tok/s"):
        assert label in text


def test_baseline_section_lists_e01_to_e07(run_factory: Any, tmp_path: Path) -> None:
    run_factory(experiment_config(experiment="e01", repetitions=2))
    root = tmp_path / "results"

    dev = build_report(root, tmp_path / "dev", dev=True).read_text()
    (row,) = [line for line in dev.splitlines() if line.startswith("| e01 | closed")]
    assert row.startswith("| e01 | closed C=2 | c | 2 | yes |")
    assert "| e01 | `" in dev  # provenance row for the table
    assert (
        "| e02 | — | — | — | — | Not yet run: `llmserve benchmark configs/baseline/e02.yaml`" in dev
    )

    strict = build_report(root, tmp_path / "strict").read_text()
    assert "| e01 | — | — | — | — | Not yet run" in strict
    assert "| baseline | e01 |" in strict
    assert "server.kind is mock, not vllm" in strict

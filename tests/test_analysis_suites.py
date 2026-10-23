"""Suite manifests read back into runs, and the tidy sweep table built from them."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

from conftest import RunFactory, experiment_config

from llmserve.analysis.runs import Run
from llmserve.analysis.suites import load_suite_runs, sweep_table

Point = tuple[str, float | None, Run | None, str]  # label, value, run, status


def write_manifest(root: Path, suite: str, axis: str | None, points: Sequence[Point]) -> None:
    """A manifest in the suite runner's format; `run=None` writes a point with no run dir."""
    (root / "_suites").mkdir(parents=True, exist_ok=True)
    rows = []
    for label, value, run, status in points:
        rows.append(
            {
                "label": label,
                "value": value,
                "experiment": run.experiment if run else f"{suite}-{label}",
                "config_hash": run.metadata["config_hash"] if run else None,
                "run_dir": str(run.path.relative_to(root)) if run else None,
                "status": status,
                "error": None if status in ("done", "skipped") else "boom",
            }
        )
    manifest = {
        "suite": suite,
        "suite_file": f"configs/suites/{suite.replace('-', '_')}.yaml",
        "axis": axis,
        "points": rows,
    }
    (root / "_suites" / f"{suite}.json").write_text(json.dumps(manifest, indent=2))


def closed(c: int, **overrides: object) -> dict[str, object]:
    return {"mode": "closed", "concurrency": c, "requests": 8, **overrides}


def test_only_usable_points_load_sorted_by_value(run_factory: RunFactory, tmp_path: Path) -> None:
    root = tmp_path / "results"
    c4 = run_factory(experiment_config(experiment="s-c004", load=closed(4)))
    c1 = run_factory(experiment_config(experiment="s-c001", load=closed(1)))
    c2 = run_factory(experiment_config(experiment="s-c002", load=closed(2)))
    write_manifest(
        root,
        "sweep",
        "load.concurrency",
        [
            ("c004", 4, c4, "done"),
            ("c001", 1, c1, "skipped"),
            ("c002", 2, c2, "failed"),
            ("c008", 8, None, "pending"),
        ],
    )
    suite = load_suite_runs(root, "sweep")
    assert suite is not None
    assert suite.axis == "load.concurrency"
    assert suite.suite_file == "configs/suites/sweep.yaml"
    assert [(p.label, p.value, p.run.path) for p in suite.points] == [
        ("c001", 1.0, c1.path),
        ("c004", 4.0, c4.path),
    ]


def test_a_done_point_whose_run_dir_is_gone_is_skipped(
    run_factory: RunFactory, tmp_path: Path
) -> None:
    root = tmp_path / "results"
    run = run_factory(experiment_config(experiment="gone"))
    write_manifest(root, "s", "load.concurrency", [("c002", 2, run, "done")])
    (run.path / "summary.json").unlink()
    suite = load_suite_runs(root, "s")
    assert suite is not None and suite.points == ()


def test_no_manifest_means_none(tmp_path: Path) -> None:
    assert load_suite_runs(tmp_path, "concurrency-p128") is None


def test_sweep_table_holds_across_rep_mean_and_ci(run_factory: RunFactory, tmp_path: Path) -> None:
    root = tmp_path / "results"
    run = run_factory(experiment_config(experiment="t-c002", repetitions=3))
    write_manifest(root, "t", "load.concurrency", [("c002", 2, run, "done")])
    suite = load_suite_runs(root, "t")
    assert suite is not None
    table = sweep_table(suite, ["throughput.gen_toks_s", "latency.no_such.p50"])
    row = table[table["metric"] == "throughput.gen_toks_s"].iloc[0]
    per_rep = run.rep_metric("throughput.gen_toks_s")
    assert row["n"] == 3
    assert math.isclose(row["mean"], statistics.fmean(per_rep))
    assert row["lo"] <= row["mean"] <= row["hi"]
    missing = table[table["metric"] == "latency.no_such.p50"].iloc[0]
    assert missing["n"] == 0 and math.isnan(missing["mean"])


def test_a_suite_of_named_points_loads_without_values(
    run_factory: RunFactory, tmp_path: Path
) -> None:
    root = tmp_path / "results"
    fifo = run_factory(experiment_config(experiment="bursty-fifo"))
    write_manifest(root, "bursty", None, [("fifo", None, fifo, "done")])
    suite = load_suite_runs(root, "bursty")
    assert suite is not None
    assert [(p.label, p.value, p.run.path) for p in suite.points] == [("fifo", None, fifo.path)]

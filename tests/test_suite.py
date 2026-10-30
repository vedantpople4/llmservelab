"""Suites: planning (sweep, `set`, runtime overrides, list-index paths, missing runtime values),
resumable sequential runs against the in-process mock, the manifest contract, and every
committed suite file."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from conftest import experiment_config
from test_experiment import make_builder

from llmserve.config.loader import config_hash
from llmserve.config.schema import ExperimentConfig, Fixed
from llmserve.mock import DelayModel, MockEngine, create_app
from llmserve.runner.experiment import run_experiment
from llmserve.runner.suite import (
    MissingRuntimeValue,
    RunFn,
    Suite,
    estimate_minutes,
    load_suite,
    manifest_path,
    parse_set,
    plan_suite,
    run_suite,
    validate_suite,
)

REPO = Path(__file__).resolve().parents[1]
CONTRACT_SUITES = {
    "concurrency-p128",
    "concurrency-mixed",
    "prompt-length",
    "output-length",
    "capacity-p128",
    "capacity-mixed",
    "capacity-mixed-long",
    "capacity-interactive",
    "bursty",
    "h2-mixed",
    "h2-interactive",
    "profile-p128",
    "profile-mixed",
    "inflight-cap",
    "capacity-long-context-heavy",
    "matrix-mixed",
    "matrix-interactive",
    "matrix-long-context-heavy",
    "xcheck-vllm-priority",
    "dev-concurrency",
    "dev-prompt-length",
}

SWEEP: dict[str, Any] = {
    "suite": "s",
    "base": "base.yaml",
    "set": {"repetitions": 1},
    "sweep": {
        "field": "load.concurrency",
        "values": [1, 2, 4],
        "label": "c{value:03d}",
        "set_per_value": {4: {"load.requests": 16}},
    },
}


def write_suite(tmp_path: Path, spec: dict[str, Any], **base: Any) -> Path:
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(experiment_config(**base).canonical()))
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def in_process() -> RunFn:
    """`run_experiment` against one in-process mock, as `test_experiment.run_in_process` does."""
    app = create_app(MockEngine(delay=DelayModel.instant()))
    builder = make_builder()

    async def run(cfg: ExperimentConfig, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
        transport = httpx.ASGITransport(app=app)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://test") as client,
            httpx.AsyncClient(transport=transport, base_url="http://test") as metrics_client,
        ):
            run_dir, cross = await run_experiment(
                cfg, client=client, metrics_client=metrics_client, builder=builder, **kwargs
            )
        # The instant mock's wall-clock TPOT wanders; pin the probes so drift cannot force a rerun.
        metadata = json.loads((run_dir / "metadata.json").read_text())
        metadata["thermal_probe"]["after"] = metadata["thermal_probe"]["before"]
        (run_dir / "metadata.json").write_text(json.dumps(metadata))
        return run_dir, cross

    return run


def run(suite: Suite, out_root: Path, runner: RunFn | None = None, **kwargs: Any) -> Any:
    return asyncio.run(
        run_suite(
            suite,
            out_root=out_root,
            allow_dirty=True,
            progress=lambda _msg: None,
            run=runner or in_process(),
            **kwargs,
        )
    )


def read_manifest(out_root: Path, name: str = "s") -> dict[str, Any]:
    data: dict[str, Any] = json.loads(manifest_path(out_root, name).read_text())
    return data


def statuses(manifest: dict[str, Any]) -> dict[str, str]:
    return {p["label"]: p["status"] for p in manifest["points"]}


def run_dirs(out_root: Path) -> set[Path]:
    return {p for p in out_root.glob("s-*/*") if p.is_dir()}


# --- planning ---


def test_plan_applies_sweep_set_and_runtime_overrides(tmp_path: Path) -> None:
    suite = load_suite(write_suite(tmp_path, SWEEP))

    points = plan_suite(suite)

    assert [p.experiment for p in points] == ["s-c001", "s-c002", "s-c004"]
    assert [p.value for p in points] == [1, 2, 4]
    assert [p.config.load.concurrency for p in points] == [1, 2, 4]
    assert [p.config.load.requests for p in points] == [12, 12, 16]  # set_per_value wins for 4
    assert {p.config.repetitions for p in points} == {1}
    assert all(p.config_hash == config_hash(p.config) for p in points)
    assert len({p.config_hash for p in points}) == 3

    reseeded = plan_suite(suite, overrides={"seed": 7, "load.requests": 20})
    assert [p.experiment for p in reseeded] == [p.experiment for p in points]
    assert {p.config.seed for p in reseeded} == {7}
    assert [p.config.load.requests for p in reseeded] == [20, 20, 20]  # runtime applies last
    assert not {p.config_hash for p in reseeded} & {p.config_hash for p in points}


def test_list_index_paths_and_dotted_labels(tmp_path: Path) -> None:
    spec = {
        "suite": "s",
        "base": "base.yaml",
        "set": {"load": {"mode": "open", "arrival": {"process": "poisson"}, "duration_s": 2}},
        "sweep": {
            "field": "workload.classes.0.prompt.tokens",
            "values": [16, 64],
            "label": "p{value}",
        },
    }
    points = plan_suite(load_suite(write_suite(tmp_path, spec)), overrides={"load.arrival.rate": 5})
    assert [p.config.workload.classes[0].prompt for p in points] == [
        Fixed(distribution="fixed", tokens=16),
        Fixed(distribution="fixed", tokens=64),
    ]
    assert {p.config.load.concurrency for p in points} == {None}  # `load` replaced whole

    rates = {
        **spec,
        "sweep": {"field": "load.arrival.rate", "values": [0.25, 2.5], "label": "r{value}"},
    }
    labels = [p.label for p in plan_suite(load_suite(write_suite(tmp_path, rates)))]
    assert labels == ["r0p25", "r2p5"]


def test_missing_capacity_names_the_set_to_pass(tmp_path: Path) -> None:
    spec = {
        "suite": "h2",
        "base": "base.yaml",
        "set": {"load": {"mode": "open", "arrival": {"process": "poisson", "rho": 0.85}}},
        "points": [{"label": "rho085", "set": {"load.duration_s": 2}}],
    }
    path = write_suite(tmp_path, spec)
    suite = load_suite(path)

    with pytest.raises(MissingRuntimeValue, match=r"--set load\.capacity_rps=") as exc:
        plan_suite(suite)
    assert exc.value.experiment == "h2-rho085"

    (point,) = plan_suite(suite, overrides={"load.capacity_rps": 4.2})
    assert point.value is None
    assert point.config.load.capacity_rps == 4.2
    assert "needs --set load.capacity_rps=" in validate_suite(path)


def test_invalid_points_are_reported_with_their_label(tmp_path: Path) -> None:
    typo = {**SWEEP, "set": {"load.concurency": 3}}
    with pytest.raises(ValueError, match="point c001"):
        plan_suite(load_suite(write_suite(tmp_path, typo)))
    both = {**SWEEP, "points": [{"label": "x"}]}
    with pytest.raises(ValueError, match="exactly one of `sweep` or `points`"):
        load_suite(write_suite(tmp_path, both))
    with pytest.raises(ValueError, match="dotted.path=value"):
        parse_set(["seed"])
    assert parse_set(["load.capacity_rps=4.2", "gateway.enabled=true"]) == {
        "load.capacity_rps": 4.2,
        "gateway.enabled": True,
    }


# --- running ---


def test_run_writes_the_manifest_and_a_rerun_skips_everything(tmp_path: Path) -> None:
    suite = load_suite(write_suite(tmp_path, SWEEP))
    out = tmp_path / "results"
    points = plan_suite(suite)

    returned = run(suite, out)

    manifest = read_manifest(out)
    assert set(manifest) == {"suite", "suite_file", "axis", "points"}
    assert manifest["suite"] == "s"
    assert manifest["suite_file"] == str(suite.path)
    assert manifest["axis"] == "load.concurrency"
    for entry, point in zip(manifest["points"], points, strict=True):
        assert set(entry) == {
            "label",
            "value",
            "experiment",
            "config_hash",
            "run_dir",
            "status",
            "error",
        }
        assert entry["label"] == point.label
        assert entry["value"] == point.value
        assert entry["experiment"] == point.experiment
        assert entry["config_hash"] == point.config_hash
        assert entry["status"] == "done"
        assert entry["error"] is None
        run_dir = out / entry["run_dir"]
        assert run_dir.parent.name == point.experiment
        assert json.loads((run_dir / "metadata.json").read_text())["config_hash"] == (
            point.config_hash
        )
    text = manifest_path(out, "s").read_text()
    assert text == json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    assert returned.model_dump(mode="json") == manifest
    before = run_dirs(out)

    again = run(suite, out)

    assert {p.status for p in again.points} == {"skipped"}
    assert [p.run_dir for p in again.points] == [p["run_dir"] for p in manifest["points"]]
    assert run_dirs(out) == before  # nothing re-ran


def test_a_changed_config_reruns_only_that_point(tmp_path: Path) -> None:
    suite = load_suite(write_suite(tmp_path, SWEEP))
    out = tmp_path / "results"
    run(suite, out, only=["c001"])

    rerun = run(suite, out, only=["c001"], overrides={"seed": 7})

    assert statuses(rerun.model_dump()) == {"c001": "done", "c002": "pending", "c004": "pending"}
    assert len(list((out / "s-c001").iterdir())) == 2


def test_only_runs_a_subset_and_leaves_the_rest_pending(tmp_path: Path) -> None:
    suite = load_suite(write_suite(tmp_path, SWEEP))
    out = tmp_path / "results"

    run(suite, out, only=["c002"])

    manifest = read_manifest(out)
    assert statuses(manifest) == {"c001": "pending", "c002": "done", "c004": "pending"}
    pending = [p for p in manifest["points"] if p["status"] == "pending"]
    assert all(p["run_dir"] is None for p in pending)
    assert not (out / "s-c001").exists()

    run(suite, out, only=["c001"])  # unselected points keep their entries
    assert statuses(read_manifest(out)) == {"c001": "done", "c002": "done", "c004": "pending"}

    with pytest.raises(ValueError, match="unknown point"):
        run(suite, out, only=["c003"])


def test_a_failing_point_is_recorded_and_the_suite_continues(tmp_path: Path) -> None:
    suite = load_suite(write_suite(tmp_path, SWEEP))
    out = tmp_path / "results"
    good = in_process()

    async def flaky(cfg: ExperimentConfig, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
        if cfg.experiment == "s-c002":
            raise RuntimeError("server went away")
        return await good(cfg, **kwargs)

    run(suite, out, runner=flaky)

    manifest = read_manifest(out)
    assert statuses(manifest) == {"c001": "done", "c002": "failed", "c004": "done"}
    failed = manifest["points"][1]
    assert failed["error"] == "RuntimeError: server went away"
    assert failed["run_dir"] is None

    retry = run(suite, out, runner=good)
    assert statuses(retry.model_dump()) == {"c001": "skipped", "c002": "done", "c004": "skipped"}


def test_a_point_the_mac_was_unfit_for_runs_again(tmp_path: Path) -> None:
    suite = load_suite(write_suite(tmp_path, SWEEP))
    out = tmp_path / "results"
    first = run(suite, out, runner=in_process())
    on_battery = out / first.points[1].run_dir
    metadata = json.loads((on_battery / "metadata.json").read_text())
    metadata["power_source"] = {"start": "AC Power", "end": "Battery Power"}
    (on_battery / "metadata.json").write_text(json.dumps(metadata))
    # Backdated: the rerun can land in the same second and collide with this directory.
    on_battery.rename(on_battery.with_name("20000101T000000Z" + on_battery.name[16:]))

    again = run(suite, out, runner=in_process())
    assert statuses(again.model_dump()) == {"c001": "skipped", "c002": "done", "c004": "skipped"}


def test_the_manifest_is_valid_json_between_points(tmp_path: Path) -> None:
    suite = load_suite(write_suite(tmp_path, SWEEP))
    out = tmp_path / "results"
    good = in_process()
    seen: list[dict[str, str]] = []

    async def watching(cfg: ExperimentConfig, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
        seen.append(statuses(read_manifest(out)))  # what a dropped session would leave behind
        return await good(cfg, **kwargs)

    run(suite, out, runner=watching)

    assert seen == [
        {"c001": "pending", "c002": "pending", "c004": "pending"},
        {"c001": "done", "c002": "pending", "c004": "pending"},
        {"c001": "done", "c002": "done", "c004": "pending"},
    ]
    assert not list(manifest_path(out, "s").parent.glob("*.tmp"))


# --- estimates and committed suites ---


def test_estimate_grows_with_overload_and_needs_a_latency_model(tmp_path: Path) -> None:
    spec = {
        "suite": "cap",
        "base": "base.yaml",
        "set": {
            "load": {"mode": "open", "arrival": {"process": "poisson"}, "duration_s": 120},
            "workload.classes.0.prompt.tokens": 128,
            "workload.classes.0.output.tokens": 128,
        },
        "sweep": {"field": "load.arrival.rate", "values": [1, 100], "label": "r{value}"},
    }
    vllm = {"kind": "vllm", "endpoint": "http://localhost:8000/v1", "model": "m"}
    low, high = plan_suite(load_suite(write_suite(tmp_path, spec, server=vllm)))
    low_min, high_min = estimate_minutes(low), estimate_minutes(high)
    assert low_min is not None and high_min is not None
    assert low_min >= 2  # never shorter than the arrival window
    assert high_min > 2 * low_min  # 12,000 arrivals drain long after the window closes

    (mock,) = plan_suite(
        load_suite(write_suite(tmp_path, {**SWEEP, "sweep": None, "points": [{"label": "x"}]}))
    )
    assert estimate_minutes(mock) is None


def committed_suites() -> list[Path]:
    return sorted((REPO / "configs" / "suites").rglob("*.yaml"))


@pytest.mark.parametrize("path", committed_suites(), ids=lambda p: p.stem)
def test_committed_suite_validates(path: Path) -> None:
    assert validate_suite(path).startswith("suite [")


def test_committed_suites_cover_the_contract() -> None:
    names = [load_suite(p).name for p in committed_suites()]
    assert sorted(names) == sorted(CONTRACT_SUITES)

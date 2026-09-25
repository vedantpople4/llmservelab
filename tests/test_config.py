import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from llmserve.cli import main
from llmserve.config.loader import config_hash, load_config
from llmserve.config.schema import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIGS = sorted(
    p for p in (ROOT / "configs").rglob("*.yaml") if p.parent.name != "workloads"
)

BASE: dict[str, Any] = {
    "schema_version": 1,
    "experiment": "t",
    "seed": 1,
    "server": {"kind": "mock", "endpoint": "http://localhost:8001/v1", "model": "m"},
    "load": {"mode": "closed", "concurrency": 4, "requests": 10},
    "workload": {
        "classes": [
            {
                "name": "a",
                "prompt": {"distribution": "fixed", "tokens": 128},
                "output": {"distribution": "fixed", "tokens": 32},
            }
        ]
    },
}


def with_(**overrides: Any) -> dict[str, Any]:
    """BASE with dotted-path overrides, e.g. with_(**{"load.mode": "open"})."""
    cfg = copy.deepcopy(BASE)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            node = node[key]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return cfg


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.name)
def test_committed_configs_validate(path: Path) -> None:
    load_config(path)


def test_defaults_are_filled() -> None:
    cfg = ExperimentConfig.model_validate(BASE)
    assert cfg.repetitions == 5
    assert cfg.gateway.enabled is False
    assert cfg.gateway.scheduler.name == "fifo"
    assert cfg.server.prefix_caching is False
    assert cfg.workload.classes[0].priority == "medium"


VALID_OPEN = {
    "load": {
        "mode": "open",
        "arrival": {"process": "poisson", "rho": 0.9},
        "capacity_rps": 12.0,
        "duration_s": 60,
    }
}


def test_open_loop_with_rho() -> None:
    cfg = ExperimentConfig.model_validate({**BASE, **VALID_OPEN})
    assert cfg.load.arrival is not None and cfg.load.arrival.process == "poisson"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"load.concurrency": None}, "needs `concurrency`"),
        ({"load.arrival": {"process": "poisson", "rate": 5}}, "cannot have `arrival`"),
        ({"load.mode": "open"}, "needs `arrival`"),
        ({"load.requests": None}, "set `requests`, `duration_s`"),
        (
            {
                "load.mode": "open",
                "load.concurrency": None,
                "load.arrival": {"process": "poisson", "rho": 0.9},
            },
            "needs `capacity_rps`",
        ),
        (
            {
                "load.mode": "open",
                "load.concurrency": None,
                "load.arrival": {"process": "poisson", "rate": 5, "rho": 0.9},
            },
            "exactly one",
        ),
        ({"workload.classes": []}, "at least 1"),
        ({"server.kind": "tgi"}, "server.kind"),
        ({"server.scheduling_policy": "priority"}, "vLLM feature"),
        ({"gateway": {"scheduler": {"name": "fifo2"}}}, "unknown scheduler"),
        ({"server.typo": True}, "Extra inputs"),
        ({"experiment": "Bad Name"}, "pattern"),
    ],
)
def test_invalid_configs(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(with_(**overrides))


@pytest.mark.parametrize(
    ("dist", "message"),
    [
        ({"distribution": "uniform", "min": 512, "max": 128}, "min"),
        ({"distribution": "choice", "values": [1, 2], "weights": [1.0]}, "same length"),
        ({"distribution": "lognormal", "median": 10, "sigma": 1, "min": 20, "max": 30}, "median"),
        ({"distribution": "zipf", "a": 2}, "distribution"),
    ],
)
def test_invalid_distributions(dist: dict[str, Any], message: str) -> None:
    cfg = with_()
    cfg["workload"]["classes"][0]["prompt"] = dist
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(cfg)


def test_duplicate_class_names() -> None:
    cfg = with_()
    cfg["workload"]["classes"].append(cfg["workload"]["classes"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        ExperimentConfig.model_validate(cfg)


def test_scheduler_requires_gateway() -> None:
    # Only FIFO is registered so far; register a stand-in to exercise the rule.
    from llmserve.scheduler import registry
    from llmserve.scheduler.fifo import FIFOScheduler

    registry.SCHEDULERS["other"] = FIFOScheduler
    try:
        with pytest.raises(ValidationError, match="gateway disabled"):
            ExperimentConfig.model_validate(with_(gateway={"scheduler": {"name": "other"}}))
        ExperimentConfig.model_validate(
            with_(gateway={"enabled": True, "scheduler": {"name": "other"}})
        )
    finally:
        del registry.SCHEDULERS["other"]


def write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_hash_ignores_key_order_and_explicit_defaults(tmp_path: Path) -> None:
    a = load_config(write(tmp_path / "a.yaml", BASE))
    reordered = dict(reversed(list(BASE.items())))
    b = load_config(write(tmp_path / "b.yaml", {**reordered, "repetitions": 5}))
    assert config_hash(a) == config_hash(b)
    c = load_config(write(tmp_path / "c.yaml", {**BASE, "seed": 2}))
    assert config_hash(a) != config_hash(c)


def test_workload_ref_equals_inline(tmp_path: Path) -> None:
    (tmp_path / "workloads").mkdir()
    write(tmp_path / "workloads" / "w.yaml", BASE["workload"])
    inline = load_config(write(tmp_path / "inline.yaml", BASE))
    ref = load_config(
        write(tmp_path / "ref.yaml", {**BASE, "workload": {"ref": "workloads/w.yaml"}})
    )
    assert config_hash(inline) == config_hash(ref)


def test_workload_ref_cannot_mix_inline(tmp_path: Path) -> None:
    bad = {**BASE, "workload": {"ref": "w.yaml", "classes": []}}
    with pytest.raises(ValueError, match="cannot be combined"):
        load_config(write(tmp_path / "bad.yaml", bad))


def test_cli_validate_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = write(tmp_path / "good.yaml", BASE)
    bad = write(tmp_path / "bad.yaml", with_(**{"load.concurrency": None}))
    with pytest.raises(SystemExit) as ok:
        main(["validate", str(good)])
    assert ok.value.code == 0
    with pytest.raises(SystemExit) as fail:
        main(["validate", str(good), str(bad)])
    assert fail.value.code == 1
    assert "FAIL" in capsys.readouterr().err

"""Shared fixtures: real run directories produced by the real runner against the in-process mock."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from test_experiment import run_in_process

from llmserve.analysis.runs import Run
from llmserve.config.schema import ExperimentConfig
from llmserve.runner import experiment, macos

RunFactory = Callable[..., Run]


def experiment_config(**overrides: Any) -> ExperimentConfig:
    """A small valid mock config; top-level keys in `overrides` replace the defaults."""
    raw: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "fixture",
        "seed": 42,
        "repetitions": 2,
        "server": {"kind": "mock", "endpoint": "http://test/v1", "model": "mock"},
        "load": {"mode": "closed", "concurrency": 2, "requests": 12},
        "workload": {
            "classes": [
                {
                    "name": "c",
                    "prompt": {"distribution": "fixed", "tokens": 16},
                    "output": {"distribution": "fixed", "tokens": 4},
                }
            ]
        },
        "measurement": {"warmup": {"requests": 0}},
    }
    raw.update(overrides)
    return ExperimentConfig.model_validate(raw)


@pytest.fixture
def run_factory(tmp_path: Path) -> RunFactory:
    """`run_factory(cfg, **run_in_process_kwargs) -> Run`: run `cfg` end to end, read it back."""

    def make(cfg: ExperimentConfig, **kwargs: Any) -> Run:
        kwargs.setdefault("allow_dirty", True)
        run_dir, _ = run_in_process(cfg, tmp_path, **kwargs)
        return Run(run_dir)

    return make


@pytest.fixture(autouse=True, scope="session")
def _not_a_mac() -> Iterator[None]:
    """Every test sees a non-Mac host, so a test run on the Mac never spawns caffeinate or reads
    its battery; `test_macos.py` opts back in with fakes. Probes skip their idle pause.
    Session-scoped so module-scoped fixtures that run whole experiments see it too."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(macos, "is_darwin", lambda: False)
        patch.setattr(macos, "_read_clocks", lambda: None)
        patch.setattr(experiment, "PROBE_SETTLE_S", 0.0)  # 30 s per probe would stall the suite
        yield

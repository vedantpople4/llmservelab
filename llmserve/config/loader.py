from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from llmserve.config.schema import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML config, resolve `workload.ref`, and validate it.

    `workload.ref` is a path relative to the config file. The referenced file holds `classes:`.
    """
    path = Path(path)
    raw = _read_yaml(path)
    workload = raw.get("workload")
    if isinstance(workload, dict) and "ref" in workload:
        if len(workload) != 1:
            raise ValueError(f"{path}: workload.ref cannot be combined with inline keys")
        raw["workload"] = _read_yaml(path.parent / workload["ref"])
    return ExperimentConfig.model_validate(raw)


def config_hash(config: ExperimentConfig) -> str:
    """SHA-256 of the resolved config. Stable across key order, comments, and `ref` vs inline."""
    blob = json.dumps(config.canonical(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data

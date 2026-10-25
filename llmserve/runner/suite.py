"""Suites: one YAML file, many experiment configs that differ along one axis (plan Phase 3 item 2).

A suite names a `base` experiment config (path relative to the suite file), suite-wide `set`
overrides, and either a `sweep` (one dotted field over a list of `values`, with optional
`set_per_value` for overrides that scale with the value, e.g. request count with concurrency)
or explicit `points` (each a `label` plus its own `set`). A sweep is sugar for points: loading
normalizes both into the same `(label, value, overrides)` list.

Design rules:

- Overrides are dotted paths into the base config's canonical dict (refs resolved, defaults
  filled, via `load_config`); integer segments index lists (`workload.classes.0.prompt.tokens`).
  A path whose value is a mapping replaces that whole subtree (`load: {mode: open, ...}`).
  They apply in order: suite `set`, the swept value, the point's own `set`, runtime overrides
  (`--set`). The point's experiment name `<suite>-<label>` is applied last.
- Runtime overrides are part of the config and therefore of its hash. A point that fails
  validation only because a value measured at run time is missing (`load.capacity_rps` for
  `rho`) raises `MissingRuntimeValue`, which names the `--set` to pass.
- Sweep labels come from a `str.format` template over `value`; a `.` in the result becomes `p`
  (`r{value:.2f}` with 0.25 → `r0p25`), because experiment names cannot contain dots.
- Resumable: a point whose latest valid run (`analysis.runs.latest_run`) has the point's config
  hash is skipped, unless that run drifted, slept or ran on battery (`Run.machine_reasons`).
  One point's failure never stops the suite. The manifest (`<out_root>/_suites/<suite>.json`)
  is rewritten atomically before the first point and after every point, so a dropped session
  loses at most the point in flight.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llmserve.analysis.runs import latest_run
from llmserve.config.loader import config_hash, load_config
from llmserve.config.schema import (
    Bursty,
    Choice,
    Constant,
    Empirical,
    ExperimentConfig,
    Fixed,
    LengthDistribution,
    Load,
    LogNormal,
    Poisson,
    Uniform,
    Workload,
)
from llmserve.runner.experiment import run_experiment

NAME_PATTERN = r"^[a-z0-9][a-z0-9_\-]*$"

Scalar = int | float | str

# Fields only a measurement can supply, with what to pass for each.
RUNTIME_FIELDS: dict[str, str] = {
    "load.capacity_rps": "<req/s> (saturation throughput from the matching capacity-* suite)",
}
_PLACEHOLDER = 1.0  # any positive number: used only to check that nothing else is wrong


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- The suite file ---


class SweepSpec(_Model):
    field: str
    values: Annotated[list[Scalar], Field(min_length=1)]
    label: str = "{value}"
    set_per_value: dict[Scalar, dict[str, Any]] = {}

    @model_validator(mode="after")
    def _values(self) -> Self:
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"sweep values must be unique: {self.values}")
        stray = [k for k in self.set_per_value if k not in self.values]
        if stray:
            raise ValueError(f"set_per_value keys {stray} are not sweep values")
        return self


class PointSpec(_Model):
    label: Annotated[str, Field(pattern=NAME_PATTERN)]
    set: dict[str, Any] = {}


class SuiteFile(_Model):
    suite: Annotated[str, Field(pattern=NAME_PATTERN)]
    description: str = ""
    base: str
    set: dict[str, Any] = {}
    sweep: SweepSpec | None = None
    points: Annotated[list[PointSpec], Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def _one_shape(self) -> Self:
        if (self.sweep is None) == (self.points is None):
            raise ValueError("set exactly one of `sweep` or `points`")
        return self


# --- The loaded suite and its plan ---


@dataclass(frozen=True)
class PointOverrides:
    label: str
    value: Scalar | None
    overrides: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class Suite:
    name: str
    description: str
    path: Path
    base: ExperimentConfig
    axis: str | None
    points: tuple[PointOverrides, ...]


@dataclass(frozen=True)
class SuitePoint:
    label: str
    value: Scalar | None
    experiment: str
    config: ExperimentConfig
    config_hash: str


class MissingRuntimeValue(ValueError):
    """A point is valid except for a value that only a measurement can supply."""

    def __init__(self, experiment: str, field: str) -> None:
        super().__init__(
            f"{experiment} needs a value measured at run time: "
            f"pass --set {field}={RUNTIME_FIELDS[field]}"
        )
        self.experiment = experiment
        self.field = field


def load_suite(path: Path) -> Suite:
    """Parse a suite file, load its base config, and expand the sweep into points."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    spec = SuiteFile.model_validate(raw)
    base = load_config(path.parent / spec.base)
    shared = tuple(spec.set.items())
    points: list[PointOverrides] = []
    if spec.sweep is not None:
        sweep = spec.sweep
        for value in sweep.values:
            label = sweep.label.format(value=value).replace(".", "p")
            per_value = tuple(sweep.set_per_value.get(value, {}).items())
            points.append(PointOverrides(label, value, (*shared, (sweep.field, value), *per_value)))
    else:
        assert spec.points is not None
        points = [PointOverrides(p.label, None, (*shared, *p.set.items())) for p in spec.points]
    labels = [p.label for p in points]
    bad = [label for label in labels if not _matches_name(label)]
    if bad:
        raise ValueError(f"{path}: labels {bad} must match {NAME_PATTERN}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: duplicate point labels: {labels}")
    return Suite(
        name=spec.suite,
        description=spec.description,
        path=path,
        base=base,
        axis=spec.sweep.field if spec.sweep is not None else None,
        points=tuple(points),
    )


def plan_suite(suite: Suite, *, overrides: Mapping[str, Any] | None = None) -> list[SuitePoint]:
    """Every point's validated config and hash, in suite order; `overrides` are runtime `--set`s."""
    runtime = tuple((overrides or {}).items())
    planned = []
    for point in suite.points:
        experiment = f"{suite.name}-{point.label}"
        raw = suite.base.canonical()
        for dotted, value in (*point.overrides, *runtime, ("experiment", experiment)):
            _set_path(raw, dotted, value)
        cfg = _validate_point(raw, experiment, where=f"{suite.path}: point {point.label}")
        planned.append(SuitePoint(point.label, point.value, experiment, cfg, config_hash(cfg)))
    return planned


def parse_set(items: Sequence[str]) -> dict[str, Any]:
    """`["load.capacity_rps=4.2", "repetitions=1"]` → overrides; values are YAML-parsed."""
    out: dict[str, Any] = {}
    for item in items:
        dotted, sep, value = item.partition("=")
        if not sep or not dotted:
            raise ValueError(f"--set expects dotted.path=value, got {item!r}")
        out[dotted] = yaml.safe_load(value)
    return out


def validate_suite(path: Path) -> str:
    """One line describing a valid suite; raises with the reason otherwise.

    A suite that needs runtime values (see `RUNTIME_FIELDS`) is valid: the line says which
    `--set` it needs.
    """
    suite = load_suite(path)
    stand_ins: dict[str, Any] = {}
    while True:
        try:
            points = plan_suite(suite, overrides=stand_ins)
            break
        except MissingRuntimeValue as e:
            if e.field in stand_ins:
                raise
            stand_ins[e.field] = _PLACEHOLDER
    axis = f"along {suite.axis}" if suite.axis else "explicit"
    labels = f"{points[0].label}..{points[-1].label}" if len(points) > 1 else points[0].label
    parts = [
        f"suite [{suite.name}] {len(points)} point(s) {axis} ({labels})",
        points[0].config.server.kind,
    ]
    estimates = [estimate_minutes(p) for p in points]
    if all(m is not None for m in estimates):
        parts.append(f"est ~{_duration_text(sum(m for m in estimates if m is not None))}")
    if stand_ins:
        parts.append("needs " + " ".join(f"--set {f}={RUNTIME_FIELDS[f]}" for f in stand_ins))
    return " · ".join(parts)


# --- Duration estimate ---


@dataclass(frozen=True)
class Hardware:
    """A rough latency model: exclusive prefill per prompt token, plus a decode step whose time
    grows linearly with batch size between batch 1 and `ref_batch`."""

    prefill_s_per_token: float
    tpot_batch1_s: float
    tpot_ref_s: float
    ref_batch: int

    def tpot_s(self, batch: int) -> float:
        slope = (self.tpot_ref_s - self.tpot_batch1_s) / (self.ref_batch - 1)
        return self.tpot_batch1_s + slope * (batch - 1)

    def seconds_per_request(self, prompt: float, output: float, batch: int) -> float:
        """Throughput-limited time per request at `batch` concurrent streams: prefill blocks the
        GPU, decode steps are shared by the batch."""
        return prompt * self.prefill_s_per_token + output * self.tpot_s(batch) / batch


# T4 + Qwen2.5-7B AWQ, from `scripts/envelope.py --preset t4`.
T4 = Hardware(prefill_s_per_token=0.06 / 128, tpot_batch1_s=0.022, tpot_ref_s=0.060, ref_batch=128)
# M5 MacBook Air + vllm-metal 0.31.0, closed-loop probes (ADR-0013): prefill ~700 tok/s at
# 2K-8K prompts; TPOT p50 47 ms at batch 1, 200 / 292 / 537 ms at 16 / 32 / 64.
M5_METAL = Hardware(
    prefill_s_per_token=1 / 700, tpot_batch1_s=0.047, tpot_ref_s=0.537, ref_batch=64
)
HARDWARE: dict[str, Hardware] = {"vllm": T4, "vllm-metal": M5_METAL}
RUN_OVERHEAD_S = 30.0  # process start, settle, metadata


def estimate_minutes(point: SuitePoint) -> float | None:
    """An estimate (within ~2x) of the point's wall time, for planning Colab sessions.

    None for backends without a latency model, empirical lengths, and replayed traces. Open
    loop takes the arrival window, or the time to drain the arrivals when they exceed what the
    hardware can serve at its reference batch (or the gateway's `max_in_flight`, if lower).
    """
    cfg = point.config
    hw = HARDWARE.get(cfg.server.kind)
    means = _mean_lengths(cfg.workload)
    if hw is None or means is None:
        return None
    prompt, output = means
    load = cfg.load
    batch = load.concurrency or hw.ref_batch
    if cfg.gateway.enabled:
        batch = min(batch, cfg.gateway.max_in_flight)
    per_request_s = hw.seconds_per_request(prompt, output, batch)
    if load.mode == "closed":
        rep_s = load.requests * per_request_s if load.requests else load.duration_s or 0.0
    else:
        arrivals = _arrivals(load)
        if arrivals is None:
            return None
        window_s, count = arrivals
        rep_s = max(window_s, count * per_request_s)
    warmup = cfg.measurement.warmup.requests
    warm_batch = max(1, min(load.concurrency or 4, warmup))
    warm_s = warmup * hw.seconds_per_request(prompt, output, warm_batch)
    return (RUN_OVERHEAD_S + warm_s + cfg.repetitions * rep_s) / 60


def _arrivals(load: Load) -> tuple[float, float] | None:
    """`(window_s, expected arrivals)` of an open-loop load; None for a replayed trace."""
    arrival = load.arrival
    if isinstance(arrival, Constant | Poisson):
        if arrival.rate is not None:
            rate = arrival.rate
        else:
            assert arrival.rho is not None and load.capacity_rps is not None
            rate = arrival.rho * load.capacity_rps
        window = load.duration_s if load.duration_s is not None else (load.requests or 0) / rate
        count = rate * window
    elif isinstance(arrival, Bursty):
        window = load.duration_s or sum(p.duration_s for p in arrival.phases)
        count = sum(p.rate * p.duration_s for p in arrival.phases)
    else:
        return None
    if load.requests is not None:
        count = min(count, load.requests)
    return window, count


def _mean_lengths(workload: Workload) -> tuple[float, float] | None:
    total = sum(c.weight for c in workload.classes)
    prompt = output = 0.0
    for c in workload.classes:
        p, o = _mean_tokens(c.prompt), _mean_tokens(c.output)
        if p is None or o is None:
            return None
        prompt += c.weight / total * p
        output += c.weight / total * o
    return prompt, output


def _mean_tokens(dist: LengthDistribution) -> float | None:
    match dist:
        case Fixed():
            return float(dist.tokens)
        case Uniform():
            return (dist.min + dist.max) / 2
        case Choice():
            weights = dist.weights or [1.0] * len(dist.values)
            return sum(v * w for v, w in zip(dist.values, weights, strict=True)) / sum(weights)
        case LogNormal():
            return min(float(dist.max), dist.median * math.exp(dist.sigma**2 / 2))
        case Empirical():
            return None


# --- Running and the manifest ---

Status = Literal["done", "skipped", "failed", "pending"]


class ManifestPoint(_Model):
    label: str
    value: Scalar | None
    experiment: str
    config_hash: str
    run_dir: str | None
    status: Status
    error: str | None


class Manifest(_Model):
    suite: str
    suite_file: str
    axis: str | None
    points: list[ManifestPoint]


class RunFn(Protocol):
    def __call__(
        self,
        cfg: ExperimentConfig,
        *,
        out_root: Path,
        endpoint: str | None,
        allow_dirty: bool,
        progress: Callable[[str], None],
    ) -> Awaitable[tuple[Path, dict[str, Any]]]: ...


def manifest_path(out_root: Path, suite: str) -> Path:
    return out_root / "_suites" / f"{suite}.json"


async def run_suite(
    suite: Suite,
    *,
    out_root: Path,
    overrides: Mapping[str, Any] | None = None,
    only: Sequence[str] | None = None,
    endpoint: str | None = None,
    allow_dirty: bool = False,
    progress: Callable[[str], None] = print,
    run: RunFn = run_experiment,
) -> Manifest:
    """Run the selected points one after another, skipping any already done; returns the
    manifest as last written.

    `only` selects points by label. Unselected points keep their previous manifest entry when
    its config is unchanged, else they show what is on disk (`skipped` or `pending`).
    """
    points = plan_suite(suite, overrides=overrides)
    labels = [p.label for p in points]
    if only is not None:
        unknown = sorted(set(only) - set(labels))
        if unknown:
            raise ValueError(f"unknown point(s) {unknown}; {suite.name} has {labels}")
    selected = [p for p in points if only is None or p.label in only]
    path = manifest_path(out_root, suite.name)
    previous = _previous_entries(path, progress)

    entries = [_resting_entry(p, previous.get(p.label), out_root) for p in points]

    def write() -> Manifest:
        manifest = Manifest(
            suite=suite.name, suite_file=str(suite.path), axis=suite.axis, points=list(entries)
        )
        _write_atomic(manifest, path)
        return manifest

    manifest = write()
    progress(
        f"suite {suite.name}: {len(selected)} of {len(points)} point(s) selected · manifest {path}"
    )
    for n, point in enumerate(selected, start=1):
        tag = f"[{n}/{len(selected)}] {point.experiment}"
        entries[labels.index(point.label)] = await _run_point(
            point,
            tag=tag,
            out_root=out_root,
            endpoint=endpoint,
            allow_dirty=allow_dirty,
            progress=progress,
            run=run,
        )
        manifest = write()
    counts = {s: sum(e.status == s for e in entries) for s in ("done", "skipped", "failed")}
    pending = sum(e.status == "pending" for e in entries)
    progress(
        f"suite {suite.name}: {counts['done']} done · {counts['skipped']} skipped · "
        f"{counts['failed']} failed · {pending} pending"
    )
    return manifest


async def _run_point(
    point: SuitePoint,
    *,
    tag: str,
    out_root: Path,
    endpoint: str | None,
    allow_dirty: bool,
    progress: Callable[[str], None],
    run: RunFn,
) -> ManifestPoint:
    done = _finished_run(point, out_root)
    if done is not None:
        progress(f"{tag}: skipped, {done} already has this config")
        return _entry(point, "skipped", run_dir=done)
    minutes = estimate_minutes(point)
    progress(f"{tag}: running" + (f" (estimate ~{minutes:.0f} min)" if minutes else ""))
    try:
        run_dir, cross = await run(
            point.config,
            out_root=out_root,
            endpoint=endpoint,
            allow_dirty=allow_dirty,
            progress=progress,
        )
    except Exception as e:  # one point's failure must not cost the rest of the session
        error = f"{type(e).__name__}: {e}"
        progress(f"{tag}: FAILED {error}")
        return _entry(point, "failed", error=error)
    relative = run_dir.relative_to(out_root).as_posix()
    if cross["invalid"]:
        error = f"run {relative} is invalid (>1% of requests failed)"
        progress(f"{tag}: FAILED {error}")
        return _entry(point, "failed", error=error)
    progress(f"{tag}: done → {relative}")
    return _entry(point, "done", run_dir=relative)


def _resting_entry(
    point: SuitePoint, previous: ManifestPoint | None, out_root: Path
) -> ManifestPoint:
    if previous is not None and previous.config_hash == point.config_hash:
        return previous
    done = _finished_run(point, out_root)
    return _entry(point, "pending") if done is None else _entry(point, "skipped", run_dir=done)


def _finished_run(point: SuitePoint, out_root: Path) -> str | None:
    """The run directory (relative to `out_root`) of the point's latest valid run, if that run
    has exactly the point's config and the Mac was fit to measure it."""
    run = latest_run(out_root, point.experiment)
    if run is None or run.metadata.get("config_hash") != point.config_hash:
        return None
    if run.machine_reasons:
        return None
    return run.path.relative_to(out_root).as_posix()


def _entry(
    point: SuitePoint, status: Status, *, run_dir: str | None = None, error: str | None = None
) -> ManifestPoint:
    return ManifestPoint(
        label=point.label,
        value=point.value,
        experiment=point.experiment,
        config_hash=point.config_hash,
        run_dir=run_dir,
        status=status,
        error=error,
    )


def _previous_entries(path: Path, progress: Callable[[str], None]) -> dict[str, ManifestPoint]:
    if not path.exists():
        return {}
    try:
        manifest = Manifest.model_validate_json(path.read_text())
    except ValidationError as e:
        progress(f"ignoring unreadable manifest {path} (rebuilding from run directories): {e}")
        return {}
    return {p.label: p for p in manifest.points}


def _write_atomic(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


# --- Overrides ---


def _validate_point(raw: dict[str, Any], experiment: str, *, where: str) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as e:
        for field in RUNTIME_FIELDS:
            if _get_path(raw, field) is None and _valid_with(raw, field, _PLACEHOLDER):
                raise MissingRuntimeValue(experiment, field) from None
        raise ValueError(f"{where}: {e}") from e


def _valid_with(raw: dict[str, Any], dotted: str, value: Any) -> bool:
    trial = copy.deepcopy(raw)
    _set_path(trial, dotted, value)
    try:
        ExperimentConfig.model_validate(trial)
    except ValidationError:
        return False
    return True


def _set_path(raw: dict[str, Any], dotted: str, value: Any) -> None:
    """Set `dotted` in `raw`, creating mappings where the path is missing or null."""
    *parents, last = dotted.split(".")
    node: Any = raw
    for seg in parents:
        if isinstance(node, list):
            node = node[_index(node, seg, dotted)]
        else:
            if node.get(seg) is None:
                node[seg] = {}
            node = node[seg]
        if not isinstance(node, dict | list):
            raise ValueError(f"--set {dotted}: {seg!r} is not a mapping or list")
    if isinstance(node, list):
        node[_index(node, last, dotted)] = copy.deepcopy(value)
    else:
        node[last] = copy.deepcopy(value)


def _get_path(raw: dict[str, Any], dotted: str) -> Any:
    node: Any = raw
    for seg in dotted.split("."):
        if isinstance(node, dict):
            node = node.get(seg)
        elif isinstance(node, list) and seg.isdigit() and int(seg) < len(node):
            node = node[int(seg)]
        else:
            return None
    return node


def _index(node: list[Any], seg: str, dotted: str) -> int:
    if not seg.isdigit() or int(seg) >= len(node):
        raise ValueError(f"{dotted}: {seg!r} is not an index into a list of {len(node)}")
    return int(seg)


def _matches_name(label: str) -> bool:
    return re.fullmatch(NAME_PATTERN, label) is not None


def _duration_text(minutes: float) -> str:
    return f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.1f} h"

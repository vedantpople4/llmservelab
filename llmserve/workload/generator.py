"""Materialize a frozen workload from a config, before any request is sent (ADR-004).

Seeds come from `SeedSequence(seed, spawn_key=(rep,))` with five independent child streams:
arrivals, class assignment, prompt lengths, output lengths and prompt content. ADR-0004's
"lengths" stream is realized as two children so changing one workload dimension never
reshuffles the others: editing `output` leaves arrivals, classes, prompt lengths and prompt
content byte-identical.

`prompt_text` is deliberately left as `None`: the parquet stores token IDs, and the runner
fills in text only for backends that cannot take token IDs (ADR-009).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.random import Generator, SeedSequence, default_rng

from llmserve.config.schema import ExperimentConfig
from llmserve.workload.arrivals import generate_arrivals
from llmserve.workload.distributions import sample
from llmserve.workload.prompts import PromptBuilder
from llmserve.workload.spec import PRIORITY_BY_NAME, RequestSpec, Workload

_COLUMNS = (
    "request_id",
    "workload_class",
    "prompt_tokens",
    "output_tokens",
    "prompt_ids",
    "arrival_offset_s",
    "priority",
    "slo_ttft_ms",
    "slo_tpot_ms",
    "output_tokens_est",
)


def _streams(seed: int, rep: int) -> tuple[Generator, Generator, Generator, Generator, Generator]:
    arrivals_s, classes_s, prompt_len_s, output_len_s, prompts_s = SeedSequence(
        seed, spawn_key=(rep,)
    ).spawn(5)
    return (
        default_rng(arrivals_s),
        default_rng(classes_s),
        default_rng(prompt_len_s),
        default_rng(output_len_s),
        default_rng(prompts_s),
    )


def materialize(
    cfg: ExperimentConfig,
    *,
    rep: int = 0,
    builder: PromptBuilder | None = None,
) -> Workload:
    """Build the `rep`-th repetition's workload for `cfg`, deterministically from `cfg.seed`."""
    builder = builder or PromptBuilder.default()
    rng_arrivals, rng_classes, rng_prompt_len, rng_output_len, rng_prompts = _streams(cfg.seed, rep)

    if cfg.load.mode == "closed":
        if cfg.load.requests is None:
            raise ValueError(
                "materialize needs load.requests for closed-loop workloads; "
                "closed-loop + duration_s streams requests at run time (Phase 2 runner)"
            )
        offsets = np.zeros(cfg.load.requests, dtype=np.float64)
    else:
        offsets = generate_arrivals(cfg.load, rng_arrivals)

    classes = cfg.workload.classes
    weights = np.array([c.weight for c in classes], dtype=np.float64)
    weights /= weights.sum()
    picks = rng_classes.choice(len(classes), size=len(offsets), p=weights)

    specs: Workload = []
    for i, pick in enumerate(picks):
        cls = classes[int(pick)]
        prompt_tokens = int(sample(cls.prompt, rng_prompt_len, 1)[0])
        output_tokens = int(sample(cls.output, rng_output_len, 1)[0])
        ids = builder.build(prompt_tokens, rng_prompts)
        specs.append(
            RequestSpec(
                request_id=f"r{i:06d}",
                workload_class=cls.name,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                prompt_ids=tuple(ids),
                arrival_offset_s=float(offsets[i]),
                priority=PRIORITY_BY_NAME[cls.priority],
                slo_ttft_ms=cls.slo.ttft_ms,
                slo_tpot_ms=cls.slo.tpot_ms,
            )
        )
    return specs


def save_workload(workload: Workload, path: str | Path) -> None:
    """Write `workload.parquet` (ADR-0004). Same workload in, byte-identical file out."""
    table = pa.table(
        {
            "request_id": pa.array([s.request_id for s in workload], pa.string()),
            "workload_class": pa.array([s.workload_class for s in workload], pa.string()),
            "prompt_tokens": pa.array([s.prompt_tokens for s in workload], pa.int64()),
            "output_tokens": pa.array([s.output_tokens for s in workload], pa.int64()),
            "prompt_ids": pa.array([list(s.prompt_ids) for s in workload], pa.list_(pa.int64())),
            "arrival_offset_s": pa.array([s.arrival_offset_s for s in workload], pa.float64()),
            "priority": pa.array([int(s.priority) for s in workload], pa.int64()),
            "slo_ttft_ms": pa.array([s.slo_ttft_ms for s in workload], pa.float64()),
            "slo_tpot_ms": pa.array([s.slo_tpot_ms for s in workload], pa.float64()),
            "output_tokens_est": pa.array([s.output_tokens_est for s in workload], pa.int64()),
        }
    )
    pq.write_table(table, path)


def load_workload(path: str | Path) -> Workload:
    """Read a workload file written by `save_workload` and verify its schema."""
    table = pq.read_table(path)
    got = set(table.column_names)
    if got != set(_COLUMNS):
        missing = sorted(set(_COLUMNS) - got)
        extra = sorted(got - set(_COLUMNS))
        raise ValueError(f"{path}: unexpected workload schema (missing={missing}, extra={extra})")
    cols = {name: table.column(name).to_pylist() for name in _COLUMNS}
    return [
        RequestSpec(
            request_id=cols["request_id"][i],
            workload_class=cols["workload_class"][i],
            prompt_tokens=cols["prompt_tokens"][i],
            output_tokens=cols["output_tokens"][i],
            prompt_ids=tuple(cols["prompt_ids"][i]),
            arrival_offset_s=cols["arrival_offset_s"][i],
            priority=cols["priority"][i],
            slo_ttft_ms=cols["slo_ttft_ms"][i],
            slo_tpot_ms=cols["slo_tpot_ms"][i],
            output_tokens_est=cols["output_tokens_est"][i],
        )
        for i in range(table.num_rows)
    ]


def summarize(workload: Workload, cfg: ExperimentConfig, rep: int) -> str:
    """One-line + one-breakdown description of a generated workload, for the CLI."""
    n = len(workload)
    if cfg.load.mode == "closed":
        shape = "closed loop"
    else:
        span = max(s.arrival_offset_s for s in workload)
        shape = f"open span {span:.1f}s"
    counts: dict[str, int] = {}
    for spec in workload:
        counts[spec.workload_class] = counts.get(spec.workload_class, 0) + 1
    mix = " · ".join(f"{name} {100 * count / n:.1f}%" for name, count in sorted(counts.items()))
    return (
        f"[{cfg.experiment}] {n} requests · {shape} · rep {rep} · seed {cfg.seed}\n"
        f"    classes: {mix}"
    )

"""Trace workloads: lengths and arrivals from one file, classes by match rule."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import experiment_config
from pydantic import ValidationError

from llmserve.config.schema import ExperimentConfig
from llmserve.workload.generator import materialize
from llmserve.workload.prompts import PromptBuilder
from llmserve.workload.trace import load_trace

ROWS = [(0.0, 3000, 50), (1.0, 100, 400), (2.0, 200, 20), (4.0, 2048, 300)]


class _Tok:
    vocab_size = 1000

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 997 + 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids)


def builder() -> PromptBuilder:
    return PromptBuilder(_Tok(), np.random.default_rng(7).integers(1, 1000, 5000).astype(np.int64))


def write(tmp_path: Path, rows: list[tuple[float, int, int]] = ROWS) -> str:
    path = tmp_path / "trace.csv"
    body = "\n".join(f"{o},{p},{n}" for o, p, n in rows)
    path.write_text(f"# source: test\noffset_s,prompt_tokens,output_tokens\n{body}\n")
    return str(path)


def cls(name: str, **match: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": name,
        "prompt": {"distribution": "fixed", "tokens": 16},
        "output": {"distribution": "fixed", "tokens": 4},
    }
    if match:
        out["match"] = match
    return out


def trace_cfg(trace: str, *, time_scale: float = 1.0, **load: Any) -> ExperimentConfig:
    return experiment_config(
        load={
            "mode": "open",
            "arrival": {"process": "replay", "trace": trace, "time_scale": time_scale},
            **load,
        },
        workload={
            "trace": trace,
            "classes": [
                cls("long_context", prompt_min=2048),
                cls("generation_heavy", output_min=256),
                cls("interactive"),
            ],
        },
    )


def test_requests_take_trace_lengths_offsets_and_first_matching_class(tmp_path: Path) -> None:
    specs = materialize(trace_cfg(write(tmp_path), time_scale=0.5, requests=4), builder=builder())
    assert [(s.prompt_tokens, s.output_tokens) for s in specs] == [(p, n) for _, p, n in ROWS]
    assert [s.arrival_offset_s for s in specs] == [0.0, 2.0, 4.0, 8.0]
    assert [s.workload_class for s in specs] == [
        "long_context",
        "generation_heavy",
        "interactive",
        "long_context",
    ]
    assert all(len(s.prompt_ids) == s.prompt_tokens for s in specs)


def test_the_same_file_and_seed_give_identical_requests(tmp_path: Path) -> None:
    cfg = trace_cfg(write(tmp_path), requests=4)
    assert materialize(cfg, builder=builder()) == materialize(cfg, builder=builder())


def test_load_cuts_the_trace_in_order(tmp_path: Path) -> None:
    specs = materialize(trace_cfg(write(tmp_path), duration_s=1.5), builder=builder())
    assert [s.prompt_tokens for s in specs] == [3000, 100]


def test_config_ties_the_trace_to_its_replay(tmp_path: Path) -> None:
    trace = write(tmp_path)
    other = str(tmp_path / "other.csv")
    with pytest.raises(ValidationError, match="replay the same file"):
        experiment_config(
            load={"mode": "open", "arrival": {"process": "replay", "trace": other}, "requests": 1},
            workload={"trace": trace, "classes": [cls("interactive")]},
        )
    with pytest.raises(ValidationError, match="catch-all"):
        experiment_config(
            load={"mode": "open", "arrival": {"process": "replay", "trace": trace}, "requests": 1},
            workload={"trace": trace, "classes": [cls("long_context", prompt_min=2048)]},
        )
    with pytest.raises(ValidationError, match="only with workload.trace"):
        experiment_config(workload={"classes": [cls("long_context", prompt_min=2048)]})


def test_trace_fields_stay_out_of_the_hash_unless_set() -> None:
    dump = experiment_config().canonical()
    assert "trace" not in dump["workload"]
    assert "match" not in dump["workload"]["classes"][0]


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("offset,prompt,output\n0,1,1\n", "header"),
        ("offset_s,prompt_tokens,output_tokens\n", "rows"),
        ("offset_s,prompt_tokens,output_tokens\n1,1,1\n0,1,1\n", "non-decreasing"),
        ("offset_s,prompt_tokens,output_tokens\n0,0,1\n", ">= 1"),
    ],
)
def test_malformed_traces_are_rejected(tmp_path: Path, text: str, error: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=error):
        load_trace(str(path))

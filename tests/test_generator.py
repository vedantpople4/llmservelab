"""Tests for the workload materializer and its parquet contract (ADR-004)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from llmserve import cli
from llmserve.config.schema import ExperimentConfig
from llmserve.workload import generator
from llmserve.workload.generator import load_workload, materialize, save_workload
from llmserve.workload.prompts import PromptBuilder


class _StubTokenizer:
    """Character-level stand-in so these tests never touch the network."""

    vocab_size = 1000

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 997 + 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids)


@pytest.fixture()
def builder() -> PromptBuilder:
    corpus = np.random.default_rng(7).integers(1, 1000, size=5000).astype(np.int64)
    return PromptBuilder(_StubTokenizer(), corpus)


def make_cfg(
    *,
    seed: int = 42,
    output: dict[str, Any] | None = None,
    load: dict[str, Any] | None = None,
) -> ExperimentConfig:
    explicit = load or {}
    load_cfg: dict[str, Any] = {"mode": "closed", "concurrency": 1, "requests": 40}
    load_cfg.update(explicit)
    if load_cfg["mode"] == "open":
        load_cfg.pop("concurrency", None)
        if "requests" not in explicit:
            load_cfg.pop("requests", None)
    return ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": "gen-test",
            "seed": seed,
            "server": {"kind": "mock", "endpoint": "http://localhost:8001/v1", "model": "mock"},
            "load": load_cfg,
            "workload": {
                "classes": [
                    {
                        "name": "interactive",
                        "weight": 3.0,
                        "priority": "high",
                        "slo": {"ttft_ms": 500, "tpot_ms": 40},
                        "prompt": {"distribution": "uniform", "min": 32, "max": 64},
                        "output": output or {"distribution": "fixed", "tokens": 16},
                    },
                    {
                        "name": "long",
                        "weight": 1.0,
                        "prompt": {"distribution": "fixed", "tokens": 128},
                        "output": {"distribution": "fixed", "tokens": 8},
                    },
                ]
            },
        }
    )


def open_cfg(**kw: Any) -> ExperimentConfig:
    arrival = kw.pop("arrival", {"process": "poisson", "rate": 5})
    output = kw.pop("output", None)
    return make_cfg(output=output, load={"mode": "open", "arrival": arrival, **kw})


def test_same_seed_same_workload(builder: PromptBuilder) -> None:
    assert materialize(make_cfg(), builder=builder) == materialize(make_cfg(), builder=builder)


def test_different_rep_differs(builder: PromptBuilder) -> None:
    a = materialize(make_cfg(), rep=0, builder=builder)
    b = materialize(make_cfg(), rep=1, builder=builder)
    assert a != b


def test_changing_output_changes_only_outputs(builder: PromptBuilder) -> None:
    a = materialize(open_cfg(duration_s=3.0), builder=builder)
    changed = {"distribution": "fixed", "tokens": 32}
    b = materialize(open_cfg(duration_s=3.0, output=changed), builder=builder)
    assert [s.arrival_offset_s for s in a] == [s.arrival_offset_s for s in b]
    assert [s.workload_class for s in a] == [s.workload_class for s in b]
    assert [s.prompt_tokens for s in a] == [s.prompt_tokens for s in b]
    assert [s.prompt_ids for s in a] == [s.prompt_ids for s in b]
    for spec_a, spec_b in zip(a, b, strict=True):
        if spec_a.workload_class == "interactive":
            assert (spec_a.output_tokens, spec_b.output_tokens) == (16, 32)
        else:  # `long` keeps its own output dist; only the override class changed
            assert spec_a.output_tokens == spec_b.output_tokens == 8


def test_spec_invariants(builder: PromptBuilder) -> None:
    wl = materialize(make_cfg(), builder=builder)
    assert len(wl) == 40
    assert len({s.request_id for s in wl}) == 40
    assert [s.request_id for s in wl] == [f"r{i:06d}" for i in range(40)]
    for spec in wl:
        assert len(spec.prompt_ids) == spec.prompt_tokens
        assert spec.workload_class in {"interactive", "long"}
        if spec.workload_class == "interactive":
            assert 32 <= spec.prompt_tokens <= 64
            assert spec.slo_ttft_ms == 500.0
            assert spec.slo_tpot_ms == 40.0
            assert int(spec.priority) == 0  # high
        else:
            assert spec.prompt_tokens == 128
            assert spec.slo_ttft_ms is None
            assert int(spec.priority) == 1  # medium (class default)
        assert spec.prompt_text is None  # filled later, per backend (ADR-009)


def test_class_weights_respected(builder: PromptBuilder) -> None:
    wl = materialize(make_cfg(), builder=builder)
    share = sum(1 for s in wl if s.workload_class == "interactive") / len(wl)
    assert 0.55 <= share <= 0.93  # 3:1 weights; expected 0.75


def test_closed_loop_arrivals_are_zero(builder: PromptBuilder) -> None:
    assert all(s.arrival_offset_s == 0.0 for s in materialize(make_cfg(), builder=builder))


def test_open_loop_arrivals_are_ordered_and_bounded(builder: PromptBuilder) -> None:
    offsets = [s.arrival_offset_s for s in materialize(open_cfg(duration_s=3.0), builder=builder)]
    assert offsets == sorted(offsets)
    assert 0 < offsets[-1] < 3.0
    assert len(offsets) > 5  # ~15 expected at 5 req/s over 3 s


def test_closed_loop_without_requests_is_rejected(builder: PromptBuilder) -> None:
    cfg = make_cfg(load={"requests": None, "duration_s": 60.0})
    with pytest.raises(ValueError, match="load.requests"):
        materialize(cfg, builder=builder)


def test_parquet_round_trip(tmp_path: Path, builder: PromptBuilder) -> None:
    wl = materialize(make_cfg(), builder=builder)
    path = tmp_path / "workload.parquet"
    save_workload(wl, path)
    assert load_workload(path) == wl


def test_parquet_bytes_are_deterministic(tmp_path: Path, builder: PromptBuilder) -> None:
    wl = materialize(make_cfg(), builder=builder)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    save_workload(wl, a)
    save_workload(wl, b)
    assert a.read_bytes() == b.read_bytes()

    other = tmp_path / "rep1.parquet"
    save_workload(materialize(make_cfg(), rep=1, builder=builder), other)
    assert a.read_bytes() != other.read_bytes()


def test_load_rejects_unexpected_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"nope": pa.array([1], pa.int64())}), path)
    with pytest.raises(ValueError, match="unexpected workload schema"):
        load_workload(path)


def test_cli_generate_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    builder: PromptBuilder,
) -> None:
    class _Stub:
        @classmethod
        def default(cls) -> PromptBuilder:
            return builder

    monkeypatch.setattr(generator, "PromptBuilder", _Stub)
    cfg_path = tmp_path / "cfg.yaml"
    out = tmp_path / "wl.parquet"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": "cli-test",
                "seed": 7,
                "server": {
                    "kind": "mock",
                    "endpoint": "http://localhost:8001/v1",
                    "model": "mock",
                },
                "load": {"mode": "closed", "concurrency": 1, "requests": 12},
                "workload": {
                    "classes": [
                        {
                            "name": "c",
                            "prompt": {"distribution": "fixed", "tokens": 64},
                            "output": {"distribution": "fixed", "tokens": 8},
                        }
                    ]
                },
            }
        )
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["generate-workload", str(cfg_path), "-o", str(out), "--rep", "0"])
    assert exit_info.value.code == 0
    assert out.exists()
    assert len(load_workload(out)) == 12
    output = capsys.readouterr().out
    assert "ok" in output and "[cli-test]" in output


def test_cli_generate_workload_bad_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": "bad",
                "seed": 1,
                "server": {
                    "kind": "mock",
                    "endpoint": "http://localhost:8001/v1",
                    "model": "mock",
                },
                "load": {"mode": "closed", "concurrency": 1, "requests": None},
                "workload": {
                    "classes": [
                        {
                            "name": "c",
                            "prompt": {"distribution": "fixed", "tokens": 64},
                            "output": {"distribution": "fixed", "tokens": 8},
                        }
                    ]
                },
            }
        )
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["generate-workload", str(cfg_path)])
    assert exit_info.value.code == 1
    assert "FAIL" in capsys.readouterr().err

"""Tests for `scripts/smoke.py`: workload sampling and the exit-code contract."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from llmserve.config.schema import ExperimentConfig
from llmserve.metrics.records import RequestRecord, RequestStatus
from llmserve.workload.prompts import PromptBuilder


def _load_smoke() -> Any:
    path = Path(__file__).resolve().parent.parent / "scripts" / "smoke.py"
    module_spec = importlib.util.spec_from_file_location("llmserve_smoke", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


smoke = _load_smoke()


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
    classes: list[dict[str, Any]] | None = None,
    server: dict[str, Any] | None = None,
) -> ExperimentConfig:
    server_cfg: dict[str, Any] = {
        "kind": "mock",
        "endpoint": "http://localhost:8001/v1",
        "model": "mock",
    }
    server_cfg.update(server or {})
    return ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": "smoke-test",
            "seed": seed,
            "server": server_cfg,
            "load": {"mode": "closed", "concurrency": 1, "requests": 10},
            "workload": {
                "classes": classes
                or [
                    {
                        "name": "p128_o16",
                        "prompt": {"distribution": "fixed", "tokens": 128},
                        "output": {"distribution": "fixed", "tokens": 16},
                    }
                ]
            },
        }
    )


def record(status: RequestStatus = RequestStatus.OK, request_id: str = "r00000") -> RequestRecord:
    ok = status is RequestStatus.OK
    return RequestRecord(
        request_id=request_id,
        workload_class="c",
        priority=1,
        prompt_tokens_req=128,
        output_tokens_req=16,
        status=status,
        t_arrival=0,
        t_first_token=1_000_000 if ok else None,
        t_last_token=17_000_000 if ok else None,
        output_tokens_usage=16 if ok else None,
        error=None if ok else "boom",
    )


def test_build_specs_is_deterministic(builder: PromptBuilder) -> None:
    cfg = make_cfg()
    assert smoke.build_specs(cfg, 10, builder) == smoke.build_specs(cfg, 10, builder)


def test_different_seed_gives_a_different_workload(builder: PromptBuilder) -> None:
    a = smoke.build_specs(make_cfg(seed=1), 10, builder)
    b = smoke.build_specs(make_cfg(seed=2), 10, builder)
    assert a != b


def test_prompt_ids_match_the_requested_length(builder: PromptBuilder) -> None:
    for spec in smoke.build_specs(make_cfg(), 20, builder):
        assert len(spec.prompt_ids) == spec.prompt_tokens == 128
        assert spec.output_tokens == 16
        assert spec.prompt_text


def test_request_ids_are_unique_and_ordered(builder: PromptBuilder) -> None:
    specs = smoke.build_specs(make_cfg(), 25, builder)
    ids = [s.request_id for s in specs]
    assert len(set(ids)) == 25
    assert ids == sorted(ids)


def test_class_weights_are_respected(builder: PromptBuilder) -> None:
    classes = [
        {
            "name": "small",
            "weight": 1.0,
            "prompt": {"distribution": "fixed", "tokens": 64},
            "output": {"distribution": "fixed", "tokens": 8},
        },
        {
            "name": "large",
            "weight": 3.0,
            "prompt": {"distribution": "fixed", "tokens": 256},
            "output": {"distribution": "fixed", "tokens": 32},
        },
    ]
    specs = smoke.build_specs(make_cfg(classes=classes), 400, builder)
    share_large = sum(1 for s in specs if s.workload_class == "large") / len(specs)
    assert 0.65 < share_large < 0.85  # expected 0.75
    assert all(len(s.prompt_ids) == s.prompt_tokens for s in specs)


def test_report_result_exits_zero_when_everything_is_ok(
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = [record(request_id=f"r{i:05d}") for i in range(5)]
    assert smoke.report_result(make_cfg(), records, 2.0, quiet=True) == 0
    out = capsys.readouterr().out
    assert "ok         5/5" in out
    assert "failures   0" in out


def test_report_result_exits_one_on_any_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = [record(request_id="r00000"), record(RequestStatus.TIMEOUT, "r00001")]
    assert smoke.report_result(make_cfg(), records, 2.0, quiet=True) == 1
    captured = capsys.readouterr()
    assert "FAIL r00001 [timeout]" in captured.out
    assert "exit 1" in captured.err


def test_report_result_groups_failures_by_status(capsys: pytest.CaptureFixture[str]) -> None:
    records = [
        record(RequestStatus.HTTP_ERROR, "r00000"),
        record(RequestStatus.HTTP_ERROR, "r00001"),
        record(RequestStatus.LENGTH_MISMATCH, "r00002"),
    ]
    assert smoke.report_result(make_cfg(), records, 1.0, quiet=True) == 1
    assert "{'http_error': 2, 'length_mismatch': 1}" in capsys.readouterr().err


def test_report_result_handles_zero_successful_requests(
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = [record(RequestStatus.CONN_ERROR, f"r{i:05d}") for i in range(3)]
    assert smoke.report_result(make_cfg(), records, 1.0, quiet=True) == 1
    out = capsys.readouterr().out
    assert "ok         0/3" in out
    assert "nan" in out  # no percentiles without successes


def test_missing_api_key_env_exits_two_before_any_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LLMSERVE_TEST_KEY", raising=False)
    cfg = make_cfg(server={"kind": "nim", "api_key_env": "LLMSERVE_TEST_KEY"})
    assert asyncio.run(smoke.smoke(cfg, 1, quiet=True)) == 2
    assert "LLMSERVE_TEST_KEY" in capsys.readouterr().err

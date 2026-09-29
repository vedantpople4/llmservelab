import dataclasses

import pytest

from llmserve.scheduler.base import Priority
from llmserve.workload.spec import RequestSpec


def make(**overrides: object) -> RequestSpec:
    base: dict[str, object] = {
        "request_id": "r1",
        "workload_class": "interactive",
        "prompt_tokens": 8,
        "output_tokens": 4,
        "prompt_ids": (1, 2, 3, 4, 5, 6, 7, 8),
    }
    base.update(overrides)
    return RequestSpec(**base)  # type: ignore[arg-type]


def test_defaults() -> None:
    spec = make()
    assert spec.priority == Priority.MEDIUM
    assert spec.arrival_offset_s == 0.0
    assert spec.prompt_text is None
    assert spec.output_tokens_est is None
    assert spec.slo_ttft_ms is None


def test_prompt_length_must_match_token_count() -> None:
    with pytest.raises(ValueError, match="len\\(prompt_ids\\)=3"):
        make(prompt_ids=(1, 2, 3))


def test_token_counts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="prompt_tokens must be >= 1"):
        make(prompt_tokens=0, prompt_ids=())
    with pytest.raises(ValueError, match="output_tokens must be >= 1"):
        make(output_tokens=0)


def test_request_id_appears_in_the_error() -> None:
    with pytest.raises(ValueError, match='request_id="req-42"|req-42'):
        make(request_id="req-42", prompt_tokens=2, prompt_ids=(1,))


def test_spec_is_frozen() -> None:
    spec = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.output_tokens = 99  # type: ignore[misc]


def test_specs_compare_by_value() -> None:
    assert make() == make()
    assert make() != make(request_id="r2")

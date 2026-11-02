"""Output-length estimators: what each returns before and after completions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmserve.gateway import estimators
from llmserve.metrics.records import RequestRecord, RequestStatus
from llmserve.workload.spec import RequestSpec


def spec(cls: str, output: int = 100) -> RequestSpec:
    return RequestSpec(
        request_id="r", workload_class=cls, prompt_tokens=1, output_tokens=output, prompt_ids=(1,)
    )


def done(s: RequestSpec, tokens: int | None) -> RequestRecord:
    return RequestRecord(
        request_id=s.request_id,
        workload_class=s.workload_class,
        priority=s.priority,
        prompt_tokens_req=s.prompt_tokens,
        output_tokens_req=s.output_tokens,
        status=RequestStatus.OK,
        t_arrival=0,
        output_tokens_usage=tokens,
    )


def test_class_prior_returns_the_class_median_and_nothing_for_unknown_classes() -> None:
    prior = estimators.ClassPrior({"a": 128, "b": 768})
    assert prior.estimate(spec("a", 999)) == 128
    assert prior.estimate(spec("b")) == 768
    assert prior.estimate(spec("c")) is None
    prior.observe(spec("a"), done(spec("a"), 5))
    assert prior.estimate(spec("a")) == 128


def test_online_learns_each_class_by_ewma() -> None:
    online = estimators.Online(alpha=0.5)
    assert online.estimate(spec("a")) is None
    online.observe(spec("a"), done(spec("a"), 100))
    assert online.estimate(spec("a")) == 100
    online.observe(spec("a"), done(spec("a"), 200))
    assert online.estimate(spec("a")) == 150
    online.observe(spec("a"), done(spec("a"), None))  # no usage: no update
    assert online.estimate(spec("a")) == 150
    assert online.estimate(spec("b")) is None


def test_online_starts_from_the_prior() -> None:
    online = estimators.Online({"a": 512}, alpha=0.25)
    assert online.estimate(spec("a")) == 512
    online.observe(spec("a"), done(spec("a"), 1024))
    assert online.estimate(spec("a")) == 640


def test_create_reads_the_prior_file(tmp_path: Path) -> None:
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"a": 128}))
    assert estimators.create("class_prior", path).estimate(spec("a")) == 128
    assert estimators.create("online", path).estimate(spec("a")) == 128
    assert estimators.create("online").estimate(spec("a")) is None
    assert estimators.create("oracle").estimate(spec("a", 7)) == 7
    assert estimators.create("none").estimate(spec("a", 7)) is None
    with pytest.raises(ValueError, match="needs gateway.output_prior"):
        estimators.create("class_prior")

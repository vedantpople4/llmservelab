"""Pre-registered verdict rules: every status reachable, missing inputs never raise."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from conftest import RunFactory, experiment_config
from test_analysis_suites import closed, write_manifest

from llmserve.analysis.hypotheses import (
    H1_TAIL,
    H1_THROUGHPUT,
    h1,
    h2,
    h3,
    judge_h1,
    judge_h2,
    judge_h3,
    prefill_points,
)
from llmserve.analysis.suites import SuiteRuns, load_suite_runs
from llmserve.mock import DelayModel, MockEngine

CS = (1, 2, 4, 8, 16)


def sweep(tput: Sequence[float], p99: Sequence[float], cs: Sequence[int] = CS) -> pd.DataFrame:
    rows = []
    for c, t, tail in zip(cs, tput, p99, strict=True):
        rows.append({"value": c, "metric": H1_THROUGHPUT, "mean": t})
        rows.append({"value": c, "metric": H1_TAIL, "mean": tail})
    return pd.DataFrame(rows)


# --- H1 ---


def test_h1_supported_when_the_knee_follows_the_plateau() -> None:
    # 350 → 370 is +5.7% (< 10%): plateau at C = 4. 0.25 > 2 × 0.1 first at C = 8.
    v = judge_h1(sweep([100, 190, 350, 370, 375], [0.1, 0.1, 0.12, 0.25, 0.6]))
    assert v.status == "supported"
    assert (v.evidence["plateau_c"], v.evidence["knee_c"]) == (4, 8)


def test_h1_knee_at_the_plateau_still_supports() -> None:
    v = judge_h1(sweep([100, 190, 350, 370, 375], [0.1, 0.1, 0.21, 0.3, 0.6]))
    assert v.status == "supported" and v.evidence["knee_c"] == v.evidence["plateau_c"] == 4


def test_h1_refuted_when_tail_latency_knees_before_throughput_plateaus() -> None:
    v = judge_h1(sweep([100, 190, 350, 370, 375], [0.1, 0.25, 0.3, 0.4, 0.6]))
    assert v.status == "refuted"
    assert (v.evidence["plateau_c"], v.evidence["knee_c"]) == (4, 2)


def test_h1_inconclusive_when_throughput_never_plateaus() -> None:
    v = judge_h1(sweep([100, 200, 400, 800, 1600], [0.1, 0.1, 0.12, 0.25, 0.6]))
    assert v.status == "inconclusive"
    assert v.evidence["plateau_c"] is None and v.evidence["knee_c"] == 8
    assert "throughput plateau" in v.summary


def test_h1_inconclusive_with_too_few_points() -> None:
    v = judge_h1(sweep([100, 101], [0.1, 0.5], cs=(1, 2)))
    assert v.status == "inconclusive"


def test_h1_not_run_without_a_usable_suite() -> None:
    assert h1(None).status == "not-run"
    empty = SuiteRuns("concurrency-p128", "f.yaml", "load.concurrency", ())
    assert h1(empty).status == "not-run"


def test_h1_on_a_real_sweep_reads_each_point(run_factory: RunFactory, tmp_path: Path) -> None:
    root = tmp_path / "results"
    points = []
    for c in (1, 2, 4):
        run = run_factory(experiment_config(experiment=f"h1-c{c:03d}", load=closed(c)))
        points.append((f"c{c:03d}", float(c), run, "done"))
    write_manifest(root, "concurrency-p128", "load.concurrency", points)
    v = h1(load_suite_runs(root, "concurrency-p128"))
    assert v.status in ("supported", "refuted", "inconclusive")
    assert v.evidence["concurrency"] == [1.0, 2.0, 4.0]
    assert v.evidence["suite"] == "concurrency-p128"


# --- H2 ---


def test_h2_supported_when_the_mixed_tail_is_worse() -> None:
    rng = np.random.default_rng(0)
    homo = rng.normal(0.2, 0.02, 500)
    mixed = np.concatenate([rng.normal(0.2, 0.02, 450), rng.normal(0.8, 0.1, 50)])
    v = judge_h2(mixed, homo)
    assert v.status == "supported"
    assert v.evidence["p95"]["ci_lo_s"] > 0
    assert v.evidence["p99_indicative"] is True  # 500 < 1,000 requests


def test_h2_refuted_when_both_tails_are_within_ten_percent() -> None:
    rng = np.random.default_rng(1)
    v = judge_h2(rng.normal(1.0, 0.05, 3000), rng.normal(1.0, 0.05, 3000))
    assert v.status == "refuted"
    for q in ("p95", "p99"):
        assert v.evidence[q]["ci_hi_s"] < 0.1 * v.evidence[q]["homogeneous_s"]


def test_h2_inconclusive_when_the_ci_is_too_wide_to_decide() -> None:
    rng = np.random.default_rng(3)
    v = judge_h2(rng.exponential(1.0, 40), rng.exponential(1.0, 40))
    assert v.status == "inconclusive"
    assert v.evidence["p95"]["ci_lo_s"] <= 0


def test_h2_inconclusive_with_too_few_requests() -> None:
    v = judge_h2(np.full(19, 5.0), np.full(100, 0.1))
    assert v.status == "inconclusive" and "Too few" in v.summary


def test_h2_not_run_names_the_missing_run(run_factory: RunFactory) -> None:
    run = run_factory(experiment_config())
    v = h2(run, None)
    assert v.status == "not-run" and "h2-interactive" in v.summary
    assert h2(None, None).status == "not-run"


# --- H3 ---


def test_h3_supported_on_a_quadratic() -> None:
    p = np.repeat([128, 512, 1024, 2048, 4096], 20).astype(float)
    rng = np.random.default_rng(0)
    ttft = 0.03 + 4e-4 * p + 2e-8 * p**2 + rng.normal(0, 0.005, len(p))
    v = judge_h3(p, ttft)
    assert v.status == "supported" and v.evidence["r2"] >= 0.95


def test_h3_refuted_when_prompt_length_does_not_explain_ttft() -> None:
    p = np.repeat([128, 512, 1024, 2048], 20).astype(float)
    v = judge_h3(p, np.random.default_rng(0).exponential(1.0, len(p)))
    assert v.status == "refuted" and v.evidence["r2"] < 0.95


def test_h3_inconclusive_with_two_lengths() -> None:
    assert judge_h3([128, 128, 512], [0.1, 0.1, 0.2]).status == "inconclusive"


def test_h3_not_run_without_a_suite() -> None:
    assert h3(None).status == "not-run"


def test_h3_fits_c1_points_and_drops_queued_ones(run_factory: RunFactory, tmp_path: Path) -> None:
    root = tmp_path / "results"
    engine = MockEngine(delay=DelayModel(ttft_base_s=0.005, ttft_per_prompt_token_s=1e-4))
    points = []
    for tokens, c in ((32, 1), (128, 1), (256, 1), (512, 4)):
        cfg = experiment_config(
            experiment=f"pl-p{tokens:05d}",
            repetitions=1,
            load=closed(c, requests=6),
            workload={
                "classes": [
                    {
                        "name": "c",
                        "prompt": {"distribution": "fixed", "tokens": tokens},
                        "output": {"distribution": "fixed", "tokens": 2},
                    }
                ]
            },
        )
        points.append((f"p{tokens:05d}", float(tokens), run_factory(cfg, engine=engine), "done"))
    write_manifest(root, "prompt-length", "workload.classes.0.prompt.tokens", points)
    suite = load_suite_runs(root, "prompt-length")
    assert suite is not None
    assert set(prefill_points(suite)["prompt_tokens"]) == {32.0, 128.0, 256.0}
    v = h3(suite)
    assert v.status == "supported", v.summary
    assert 5e-5 < v.evidence["b"] < 2e-4  # the mock's 1e-4 s per prompt token

"""Cost-model fits: exact recovery on noiseless data, and the guards on degenerate input."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from llmserve.scheduler.estimators import CostModel, fit_decode, fit_prefill


def test_prefill_fit_recovers_a_known_quadratic() -> None:
    p = np.array([128, 512, 1024, 2048, 4096, 8192], dtype=float)
    ttft = 0.03 + 2e-4 * p + 5e-9 * p**2
    fit = fit_prefill(p, ttft)
    assert math.isclose(fit.a, 0.03, rel_tol=1e-6)
    assert math.isclose(fit.b, 2e-4, rel_tol=1e-6)
    assert math.isclose(fit.c, 5e-9, rel_tol=1e-6)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.n == 6
    assert fit.predict(2048) == pytest.approx(0.03 + 2e-4 * 2048 + 5e-9 * 2048**2)


def test_noise_lowers_r2_below_one() -> None:
    rng = np.random.default_rng(0)
    p = np.repeat([128, 1024, 4096, 8192], 25).astype(float)
    ttft = 0.03 + 2e-4 * p + rng.normal(0, 0.05, size=p.size)
    fit = fit_prefill(p, ttft)
    assert 0.5 < fit.r2 < 1.0


def test_decode_fit_recovers_a_line_and_drops_nan() -> None:
    batch = [1, 2, 4, 8, 16, float("nan")]
    tpot = [0.04 + 0.006 * b for b in batch[:-1]] + [0.2]
    fit = fit_decode(batch, tpot)
    assert fit.d0 == pytest.approx(0.04)
    assert fit.d1 == pytest.approx(0.006)
    assert fit.n == 5


def test_degenerate_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="3 distinct"):
        fit_prefill([128, 128, 512], [0.1, 0.1, 0.2])
    with pytest.raises(ValueError, match="2 distinct"):
        fit_decode([8, 8, 8], [0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="differ in length"):
        fit_decode([1, 2], [0.1])


def test_cost_model_round_trips_and_predicts_service_time(tmp_path: Path) -> None:
    model = CostModel(
        prefill=fit_prefill([128, 1024, 4096], [0.05, 0.25, 1.0]),
        decode=fit_decode([1, 8], [0.04, 0.09]),
    )
    path = tmp_path / "cost.json"
    model.save(path)
    loaded = CostModel.load(path)
    assert loaded == model
    expected = model.prefill.predict(1024) + 127 * model.decode.predict(8)
    assert loaded.service_time(1024, 128, 8) == pytest.approx(expected)

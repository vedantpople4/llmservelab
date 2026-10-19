"""Mock calibration (plan Phase 3 item 6): can the fit recover step costs it was not told?

Identification runs the real runner against the cb mock served with KNOWN params, then fits.
The server runs as a subprocess over a real socket (started with `--params`, so these runs also
prove the flag loads): `httpx.ASGITransport`, which `run_factory` uses, buffers a streamed
response whole, so every token would arrive at once and TTFT/ITL would be meaningless.

The known params are scaled so the five runs take about 3 s: steps of 5-17 ms, prompts of
64-1024 tokens (1024 needs two 512-token prefill chunks), 16 output tokens.
"""

from __future__ import annotations

import asyncio
import importlib.util
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import experiment_config
from test_experiment import make_builder

from llmserve.analysis.runs import Run
from llmserve.mock.__main__ import build_engine, parse_args
from llmserve.mock.calibration import (
    COEFFICIENTS,
    Point,
    fit_cb_params,
    fit_step_costs,
    points_of,
)
from llmserve.mock.engine import CbParams, ContinuousBatchEngine
from llmserve.runner.experiment import run_experiment

REPO = Path(__file__).resolve().parent.parent
TRUE = CbParams(alpha_s=4e-3, beta_s=2e-5, gamma_s=1e-3, delta_s=2e-6)
# (concurrency, prompt tokens, requests); each run's requests give a steady window of 3-17.
SPECS = [(1, 64, 4), (1, 256, 4), (1, 1024, 4), (4, 256, 12), (8, 256, 24)]
OUTPUT_TOKENS = 16


def _load_script() -> Any:
    path = REPO / "scripts" / "calibrate_mock.py"
    module_spec = importlib.util.spec_from_file_location("llmserve_calibrate_mock", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


calibrate_mock = _load_script()


@pytest.fixture(scope="module")
def mock_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    params_file = tmp_path_factory.mktemp("params") / "true.json"
    params_file.write_text(TRUE.to_json())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    proc = subprocess.Popen(
        [sys.executable, "-m", "llmserve.mock", "--engine", "cb", "--params", str(params_file)]
        + ["--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=REPO,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while True:
            if proc.poll() is not None:
                pytest.fail(f"mock server exited early with code {proc.returncode}")
            try:
                if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.TransportError:
                if time.monotonic() > deadline:
                    pytest.fail("mock server never became healthy")
                time.sleep(0.1)
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def runs(mock_url: str, tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[int, int], Run]:
    out_root = tmp_path_factory.mktemp("results")

    async def one(concurrency: int, prompt: int, requests: int) -> Run:
        cfg = experiment_config(
            experiment=f"cal-c{concurrency}-p{prompt}",
            repetitions=1,
            server={"kind": "mock", "endpoint": f"{mock_url}/v1", "model": "mock"},
            load={"mode": "closed", "concurrency": concurrency, "requests": requests},
            workload={
                "classes": [
                    {
                        "name": "c",
                        "prompt": {"distribution": "fixed", "tokens": prompt},
                        "output": {"distribution": "fixed", "tokens": OUTPUT_TOKENS},
                    }
                ]
            },
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            run_dir, _ = await run_experiment(
                cfg, out_root=out_root, client=client, builder=make_builder(), allow_dirty=True
            )
        return Run(run_dir)

    return {(c, p): asyncio.run(one(c, p, n)) for c, p, n in SPECS}


def test_fit_recovers_known_params_through_the_real_harness(
    runs: dict[tuple[int, int], Run],
) -> None:
    """Tolerances: alpha 35%, the rest 25%; predictions 10% (TPOT) and 25% (TTFT).

    Measured on macOS, every step carries ~0.7 ms the engine does not model (asyncio timer
    overshoot, the SSE write and the client's parse), and it lands in alpha (+17% on a 4 ms
    step); timer slack proportional to the sleep inflates delta by ~10%. Beta and gamma come
    back within ~3%. Decode ITL is a median over 15+ steps, so TPOT predictions hold to ~1%;
    TTFT at concurrency > 1 includes closed-loop queueing over a 9-17 request window (~15%).
    """
    cal = fit_cb_params(list(runs.values()), kv_capacity_tokens=TRUE.kv_capacity_tokens)

    assert cal.warnings == ()
    assert cal.params.alpha_s == pytest.approx(TRUE.alpha_s, rel=0.35)
    for name in ("beta_s", "gamma_s", "delta_s"):
        assert getattr(cal.params, name) == pytest.approx(getattr(TRUE, name), rel=0.25), name
    assert 0 <= cal.request_overhead_s < 0.01
    assert len(cal.table) == len(SPECS)
    assert (cal.table["tpot_rel_err"].abs() < 0.10).all(), cal.table
    assert (cal.table["ttft_rel_err"].abs() < 0.25).all(), cal.table


def test_fit_is_exact_on_the_engines_own_timings(runs: dict[tuple[int, int], Run]) -> None:
    """With timing noise removed (the engine replayed on a virtual clock), what is left is the
    model's approximation of the step rule: for these workloads it is exact."""
    points = [p for run in runs.values() for p in points_of(run, replay=TRUE)]

    fitted, warnings = fit_step_costs(points, TRUE)

    assert warnings == ()
    for name in COEFFICIENTS:
        assert fitted[name] == pytest.approx(getattr(TRUE, name), rel=0.01), name
    assert fitted["request_overhead_s"] == pytest.approx(0.0, abs=1e-6)


def test_negative_fit_is_reported_and_zeroed() -> None:
    """TTFT that falls with prompt length forces beta negative; the rest refit without it."""

    def point(concurrency: int, prompt: int, ttft: float, itl: float, context: float) -> Point:
        return Point(
            run="synthetic",
            concurrency=concurrency,
            prompt_tokens=prompt,
            output_tokens=16,
            requests=4,
            ttft_s=ttft,
            tpot_s=itl,
            itl_s=itl,
            batch_context_tokens=context,
        )

    points = [
        point(1, 100, 4.2e-3, 5.216e-3, 108),
        point(1, 400, 1.8e-3, 5.816e-3, 408),
        point(4, 100, 9.0e-3, 8.864e-3, 432),
    ]

    fitted, warnings = fit_step_costs(points, CbParams())

    assert any(w.startswith("beta_s fitted negative") for w in warnings), warnings
    assert fitted["beta_s"] == 0.0
    assert all(value >= 0 for value in fitted.values()), fitted


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        ([(4, 256), (8, 256)], "no concurrency-1 run"),
        ([(1, 256), (4, 256)], "fewer than 2 prompt lengths"),
        ([(1, 64), (1, 256), (1, 1024)], "fewer than 2 concurrency levels"),
    ],
)
def test_cli_rejects_runs_that_cannot_identify_the_params(
    runs: dict[tuple[int, int], Run],
    keys: list[tuple[int, int]],
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "params.json"
    argv = [str(runs[k].path) for k in keys] + ["--kv-capacity-tokens", "16384", "-o", str(out)]

    assert calibrate_mock.main(argv) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_cli_rejects_a_directory_that_is_not_a_run(tmp_path: Path) -> None:
    out = tmp_path / "params.json"
    assert calibrate_mock.main([str(tmp_path), "--kv-capacity-tokens", "1", "-o", str(out)]) == 2


def test_cli_writes_params_the_mock_server_loads(
    runs: dict[tuple[int, int], Run], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "params.json"
    argv = [str(run.path) for run in runs.values()]
    argv += ["--kv-capacity-tokens", "30000", "--max-batched-tokens", "4096", "-o", str(out)]

    assert calibrate_mock.main(argv) == 0

    printed = capsys.readouterr().out
    assert "alpha_s" in printed and "tpot_rel_err" in printed
    engine = build_engine(parse_args(["--engine", "cb", "--params", str(out)]))
    assert isinstance(engine, ContinuousBatchEngine)
    assert engine.params.kv_capacity_tokens == 30000
    assert engine.params.max_num_batched_tokens == 4096
    assert engine.params.alpha_s > 0


def test_params_json_round_trips() -> None:
    params = replace(TRUE, kv_capacity_tokens=1234, prefill_chunk=256)
    assert CbParams.from_json(params.to_json()) == params
    assert CbParams.from_json('{"alpha_s": 0.01}') == replace(CbParams(), alpha_s=0.01)


@pytest.mark.parametrize(
    "text",
    [
        '{"alpha": 0.01}',
        '{"alpha_s": "fast"}',
        '{"alpha_s": -1}',
        '{"kv_capacity_tokens": 1.5}',
        '{"prefill_chunk": 0}',
        "[1, 2]",
    ],
)
def test_params_json_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        CbParams.from_json(text)


@pytest.mark.parametrize(
    "argv",
    [
        ["--engine", "cb", "--params", "p.json", "--instant"],
        ["--engine", "delay", "--params", "p.json"],
    ],
)
def test_mock_cli_rejects_params_with_instant_or_the_delay_engine(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2

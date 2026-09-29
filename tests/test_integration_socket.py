"""End-to-end over a real socket: the mock server in a subprocess, the client over TCP.

Everything else in the suite runs against `httpx.ASGITransport`, which never opens a socket. This
file is the one that catches transport-level breakage: chunked transfer, SSE framing on a real
connection, the health/version endpoints over HTTP, and the smoke script's exit code.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from llmserve.client.capabilities import capabilities
from llmserve.client.openai_stream import stream_completion
from llmserve.config.schema import ExperimentConfig
from llmserve.runner.clock import RunClock
from llmserve.runner.env_check import check_server
from llmserve.workload.prompts import PromptBuilder
from llmserve.workload.spec import RequestSpec

REPO = Path(__file__).resolve().parent.parent
CAPS = capabilities("mock")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def mock_url() -> Iterator[str]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "llmserve.mock", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=REPO,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    healthy = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            proc.wait()
            pytest.fail(f"mock server exited early with code {proc.returncode}")
        try:
            if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                healthy = True
                break
        except httpx.TransportError:
            time.sleep(0.2)
    if not healthy:
        proc.terminate()
        proc.wait(timeout=10)
        pytest.fail("mock server never became healthy")
    try:
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def make_cfg(base: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": "socket-it",
            "seed": 42,
            "server": {"kind": "mock", "endpoint": f"{base}/v1", "model": "mock"},
            "load": {"mode": "closed", "concurrency": 1, "requests": 5},
            "workload": {
                "classes": [
                    {
                        "name": "c",
                        "prompt": {"distribution": "fixed", "tokens": 16},
                        "output": {"distribution": "fixed", "tokens": 4},
                    }
                ]
            },
        }
    )


def test_env_check_over_a_real_socket(mock_url: str) -> None:
    async def run() -> None:
        async with httpx.AsyncClient() as client:
            report = await check_server(make_cfg(mock_url), client=client)
        assert "health ok" in report.checks
        assert report.version is not None

    asyncio.run(run())


def test_stream_completion_over_a_real_socket(mock_url: str) -> None:
    spec = RequestSpec(
        request_id="sock-1",
        workload_class="c",
        prompt_tokens=16,
        output_tokens=4,
        prompt_ids=tuple(range(100, 116)),
    )

    async def run() -> None:
        clock = RunClock()
        async with httpx.AsyncClient(timeout=30.0) as client:
            rec = await stream_completion(
                client,
                spec,
                clock,
                endpoint=f"{mock_url}/v1",
                model="mock",
                t_arrival=clock(),
                caps=CAPS,
            )
        assert rec.status.value == "ok", rec.error
        assert len(rec.chunks) == 4
        assert rec.output_tokens_usage == 4
        assert rec.http_status == 200

    asyncio.run(run())


def test_smoke_script_exit_code_over_a_real_socket(mock_url: str) -> None:
    try:
        PromptBuilder.default()
    except (RuntimeError, OSError) as e:
        pytest.skip(f"tokenizer unavailable: {e}")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke.py",
            "configs/dev/smoke_mock.yaml",
            "--requests",
            "5",
            "--quiet",
            "--endpoint",
            f"{mock_url}/v1",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "0 unexplained failures, 0 usage mismatches" in result.stdout
    assert "ok         5/5" in result.stdout

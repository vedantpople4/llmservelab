import asyncio
from typing import Any

import httpx
import pytest

from llmserve.config.schema import ExperimentConfig
from llmserve.mock import DelayModel, MockEngine, create_app
from llmserve.runner import env_check
from llmserve.runner.env_check import EnvCheckError, check_server


def make_cfg(**server: Any) -> ExperimentConfig:
    base: dict[str, Any] = {"kind": "mock", "endpoint": "http://test/v1", "model": "mock"}
    base.update(server)
    return ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "experiment": "env-check",
            "seed": 1,
            "server": base,
            "load": {"mode": "closed", "concurrency": 1, "requests": 1},
            "workload": {
                "classes": [
                    {
                        "name": "c",
                        "prompt": {"distribution": "fixed", "tokens": 8},
                        "output": {"distribution": "fixed", "tokens": 4},
                    }
                ]
            },
        }
    )


def mock_client() -> httpx.AsyncClient:
    app = create_app(MockEngine(delay=DelayModel.instant()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_passes_against_the_mock() -> None:
    async def run() -> None:
        async with mock_client() as client:
            report = await check_server(make_cfg(), client=client)
        assert report.kind == "mock"
        assert report.version is not None
        assert (report.prefix_cache_probe or "").startswith("skipped")
        assert report.gpu_idle is None
        assert "health ok" in report.checks

    asyncio.run(run())


def test_unreachable_server_is_reported() -> None:
    async def run() -> None:
        with pytest.raises(EnvCheckError, match="not reachable"):
            await check_server(make_cfg(endpoint="http://127.0.0.1:9/v1"))

    asyncio.run(run())


def test_model_mismatch_is_a_failure() -> None:
    async def run() -> None:
        async with mock_client() as client:
            with pytest.raises(EnvCheckError, match="model mismatch"):
                await check_server(make_cfg(model="other-model"), client=client)

    asyncio.run(run())


def test_version_mismatch_is_a_failure() -> None:
    async def run() -> None:
        async with mock_client() as client:
            with pytest.raises(EnvCheckError, match="version mismatch"):
                await check_server(make_cfg(expected_version="9.9.9"), client=client)

    asyncio.run(run())


def test_prefix_cache_enabled_in_config_is_a_failure() -> None:
    async def run() -> None:
        async with mock_client() as client:
            with pytest.raises(EnvCheckError, match="prefix_caching must be false"):
                await check_server(make_cfg(kind="vllm", prefix_caching=True), client=client)

    asyncio.run(run())


def test_vllm_config_probes_a_clean_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(*args: Any, **kwargs: Any) -> tuple[float, float]:
        return (0.030, 0.029)

    monkeypatch.setattr(env_check, "_identical_prompt_ttfts", fake)

    async def run() -> None:
        async with mock_client() as client:
            report = await check_server(make_cfg(kind="vllm"), client=client)
        assert report.prefix_cache_probe == "clean"

    asyncio.run(run())


def test_suspected_prefix_cache_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(*args: Any, **kwargs: Any) -> tuple[float, float]:
        return (0.100, 0.005)

    monkeypatch.setattr(env_check, "_identical_prompt_ttfts", fake)

    async def run() -> None:
        async with mock_client() as client:
            with pytest.raises(EnvCheckError, match="prefix cache suspected"):
                await check_server(make_cfg(kind="vllm"), client=client)

    asyncio.run(run())


def test_endpoint_override_checks_the_override() -> None:
    async def run() -> None:
        with pytest.raises(EnvCheckError, match="not reachable"):
            await check_server(make_cfg(), endpoint="http://127.0.0.1:9/v1")

    asyncio.run(run())

"""Pre-flight checks: is the server the one the config says it is? (Phase 1 design item 4).

Run before any measured request. Everything that would silently invalidate a run lives here:
an unreachable or wrong server, a version mismatch, prefix caching left on (which makes TTFT lie),
and a GPU that is already busy.

Checks that a backend cannot answer (ADR-009) are recorded as skipped rather than passed, so the
run metadata can say what was and was not verified.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import httpx

from llmserve.client.capabilities import Capabilities, capabilities
from llmserve.client.openai_stream import stream_completion
from llmserve.config.schema import ExperimentConfig
from llmserve.metrics import latency
from llmserve.runner.clock import RunClock
from llmserve.workload.spec import RequestSpec

GPU_IDLE_UTIL_PCT = 5.0
CACHE_PROBE_PROMPT = tuple(range(1000, 1128))  # 128 valid token ids, fixed so probes are comparable


class EnvCheckError(RuntimeError):
    """A pre-flight check failed; the run must not start."""


@dataclass(frozen=True)
class EnvReport:
    kind: str
    endpoint: str
    model: str
    version: str | None = None
    prefix_cache_probe: str | None = None
    gpu_idle: bool | None = None
    checks: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        return "; ".join(self.checks)


def _base_url(endpoint: str) -> str:
    """`http://host:8001/v1` → `http://host:8001` (health and version are served at the root)."""
    base = endpoint.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


async def check_server(
    cfg: ExperimentConfig,
    *,
    probe: bool = True,
    endpoint: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> EnvReport:
    """Verify the server matches the config. Raises `EnvCheckError` listing every failure.

    `endpoint` overrides `server.endpoint`, so a run against a different port is checked where it
    will actually send.
    """
    caps = capabilities(cfg.server.kind)
    target = endpoint or cfg.server.endpoint
    base = _base_url(target)
    failures: list[str] = []
    checks: list[str] = []
    version: str | None = None

    owns_client = client is None
    client = client if client is not None else httpx.AsyncClient(timeout=10.0)
    try:
        try:
            health = await client.get(f"{base}/health")
        except httpx.TransportError as e:
            raise EnvCheckError(f"server not reachable at {base}: {e}") from e
        if health.status_code != 200:
            failures.append(f"GET /health → HTTP {health.status_code}")
        else:
            checks.append("health ok")

        if caps.version_endpoint:
            version = await _version(client, base, failures, checks, cfg)
        else:
            checks.append("version check skipped (backend has no /version)")

        if caps.model_lookup:
            await _model(client, target, cfg, failures, checks)
        else:
            checks.append("model check skipped (backend has no /v1/models)")

        prefix_note = await _check_prefix_cache(cfg, caps, probe, client, failures, checks, target)
        gpu_idle = _check_gpu(caps, failures, checks)
    finally:
        if owns_client:
            await client.aclose()

    if failures:
        raise EnvCheckError("; ".join(failures))
    return EnvReport(
        kind=cfg.server.kind,
        endpoint=target,
        model=cfg.server.model,
        version=version,
        prefix_cache_probe=prefix_note,
        gpu_idle=gpu_idle,
        checks=tuple(checks),
    )


async def _version(
    client: httpx.AsyncClient,
    base: str,
    failures: list[str],
    checks: list[str],
    cfg: ExperimentConfig,
) -> str | None:
    try:
        resp = await client.get(f"{base}/version")
        version = str(resp.json().get("version")) if resp.status_code == 200 else None
    except (httpx.TransportError, ValueError) as e:
        failures.append(f"GET /version failed: {e}")
        return None
    if version is None:
        failures.append(f"GET /version → HTTP {resp.status_code}")
        return None
    expected = cfg.server.expected_version
    if expected is not None and version != expected:
        failures.append(f"version mismatch: expected {expected}, server reports {version}")
    else:
        checks.append(f"version {version}")
    return version


async def _model(
    client: httpx.AsyncClient,
    endpoint: str,
    cfg: ExperimentConfig,
    failures: list[str],
    checks: list[str],
) -> None:
    try:
        resp = await client.get(f"{endpoint.rstrip('/')}/models")
        ids = [str(m.get("id")) for m in resp.json().get("data", [])]
    except (httpx.TransportError, ValueError, AttributeError) as e:
        failures.append(f"GET /v1/models failed: {e}")
        return
    if cfg.server.model not in ids:
        failures.append(f"model mismatch: config wants {cfg.server.model!r}, server has {ids}")
    else:
        checks.append(f"model {cfg.server.model}")


async def _check_prefix_cache(
    cfg: ExperimentConfig,
    caps: Capabilities,
    probe: bool,
    client: httpx.AsyncClient,
    failures: list[str],
    checks: list[str],
    endpoint: str,
) -> str:
    if not caps.prefix_cache_risk:
        return "skipped (backend has no prefix cache)"
    if cfg.server.prefix_caching:
        failures.append("server.prefix_caching must be false for measurements (ADR-005)")
        return "not probed (prefix caching enabled in config)"
    if not probe:
        return "skipped (probe disabled)"

    ttfts = await _identical_prompt_ttfts(cfg, caps, client, endpoint)
    if ttfts is None:
        return "failed (probe request did not succeed)"
    first, second = ttfts
    if first is None or second is None:
        return "failed (no TTFT for the probe requests)"
    if second < 0.5 * first:
        failures.append(
            f"prefix cache suspected: identical-prompt TTFT {first * 1e3:.0f} ms → "
            f"{second * 1e3:.0f} ms"
        )
        return "suspected"
    checks.append(f"prefix cache probe clean ({first * 1e3:.0f} ms → {second * 1e3:.0f} ms)")
    return "clean"


async def _identical_prompt_ttfts(
    cfg: ExperimentConfig,
    caps: Capabilities,
    client: httpx.AsyncClient,
    endpoint: str,
) -> tuple[float | None, float | None] | None:
    """Two identical short requests; a much faster second TTFT means a prefix cache hit."""
    clock = RunClock()
    base_spec = RequestSpec(
        request_id="env-probe",
        workload_class="env_check",
        prompt_tokens=len(CACHE_PROBE_PROMPT),
        output_tokens=1,
        prompt_ids=CACHE_PROBE_PROMPT,
    )
    ttfts: list[float | None] = []
    for i in range(2):
        t0 = clock()
        record = await stream_completion(
            client,
            replace(base_spec, request_id=f"env-probe-{i}"),
            clock,
            endpoint=endpoint,
            model=cfg.server.model,
            t_arrival=t0,
            caps=caps,
            timeout_s=30.0,
        )
        if not record.ok:
            return None
        ttfts.append(latency.ttft(record))
    return ttfts[0], ttfts[1]


def _check_gpu(caps: Capabilities, failures: list[str], checks: list[str]) -> bool | None:
    if not caps.gpu_metrics:
        checks.append("GPU check skipped (backend has no GPU metrics)")
        return None
    util = _gpu_utilization()
    if util is None:
        checks.append("GPU check skipped (NVML unavailable)")
        return None
    if util > GPU_IDLE_UTIL_PCT:
        failures.append(f"GPU is busy: {util:.0f}% utilization (need < {GPU_IDLE_UTIL_PCT:.0f}%)")
        return False
    checks.append(f"GPU idle ({util:.0f}%)")
    return True


def _gpu_utilization() -> float | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:  # NVML errors mean "cannot tell", not "busy"
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


async def main() -> None:  # pragma: no cover - manual entry point
    """`uv run python -m llmserve.runner.env_check <config.yaml>`."""
    import sys

    from llmserve.config.loader import load_config

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m llmserve.runner.env_check <config.yaml>")
    report = await check_server(load_config(sys.argv[1]))
    print(f"ok  {report.kind} · {report.endpoint} · {report.summary}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())

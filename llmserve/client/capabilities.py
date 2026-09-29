"""Backend capability table (ADR-009, ADR-0010).

`server.kind` picks the backend; the harness adapts to what that backend can actually do instead
of assuming vLLM's API everywhere. Flags are read by the streaming client (prompt format, forced
output length, usage ground truth) and by `runner/env_check.py` (which probes exist, and whether a
prefix cache can make TTFT lie).

Update this table deliberately: adding a flag is a harness change, not a config change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ServerKind = Literal["mock", "mlx", "llamacpp", "ollama", "lmstudio", "nim", "vllm"]


@dataclass(frozen=True)
class Capabilities:
    token_id_prompts: bool
    """Accepts `prompt: [int, ...]` on `/v1/completions` (ADR-005)."""

    forced_output_length: bool
    """`ignore_eos` + `max_tokens`/`min_tokens` produce exactly N tokens."""

    usage_block: bool
    """`stream_options.include_usage` returns a `usage` object, so token counts are ground truth."""

    version_endpoint: bool
    """`GET /version` exists, so `expected_version` can be asserted."""

    model_lookup: bool
    """`GET /v1/models` exists, so the served model can be checked."""

    metrics_endpoint: bool
    """Prometheus `/metrics` exists for the server sampler (Phase 2)."""

    prefix_cache_risk: bool
    """Identical prompts may be served from a prefix cache, so TTFT must be probed."""

    prefix_cache_control: bool
    """The cache can be turned off server-side; otherwise a suspected cache is a note (ADR-0010)."""

    health_endpoint: bool
    """`GET /health` must answer 200; backends without it are checked for reachability only."""

    gpu_metrics: bool
    """NVML/DCGM sampling is meaningful (there is a real GPU behind this backend)."""


SERVER_KINDS: tuple[ServerKind, ...] = (
    "mock",
    "mlx",
    "llamacpp",
    "ollama",
    "lmstudio",
    "nim",
    "vllm",
)

CAPABILITIES: dict[str, Capabilities] = {
    "mock": Capabilities(
        token_id_prompts=True,
        forced_output_length=True,
        usage_block=True,
        version_endpoint=True,
        model_lookup=True,
        metrics_endpoint=False,  # added with the Phase 2 engine
        prefix_cache_risk=False,  # the mock has no cache
        prefix_cache_control=True,
        health_endpoint=True,
        gpu_metrics=False,
    ),
    "mlx": Capabilities(
        token_id_prompts=False,
        forced_output_length=False,
        usage_block=True,
        version_endpoint=False,
        model_lookup=True,
        metrics_endpoint=False,
        prefix_cache_risk=False,
        prefix_cache_control=True,
        health_endpoint=False,
        gpu_metrics=False,
    ),
    "llamacpp": Capabilities(
        token_id_prompts=False,
        forced_output_length=False,
        usage_block=True,
        version_endpoint=False,
        model_lookup=True,
        metrics_endpoint=False,
        prefix_cache_risk=False,  # llama.cpp only reuses prompt chunks when opted in
        prefix_cache_control=True,
        health_endpoint=False,
        gpu_metrics=False,
    ),
    "ollama": Capabilities(
        token_id_prompts=False,
        forced_output_length=False,
        usage_block=True,
        version_endpoint=False,
        model_lookup=True,
        metrics_endpoint=False,
        prefix_cache_risk=False,  # no cross-request prefix cache in the OpenAI API
        prefix_cache_control=True,
        health_endpoint=False,  # serves /api/version, not /health
        gpu_metrics=False,
    ),
    "lmstudio": Capabilities(
        token_id_prompts=False,
        forced_output_length=False,
        usage_block=True,
        version_endpoint=False,
        model_lookup=True,
        metrics_endpoint=False,
        prefix_cache_risk=False,
        prefix_cache_control=True,
        health_endpoint=False,
        gpu_metrics=False,
    ),
    "nim": Capabilities(
        token_id_prompts=False,  # text prompts; not verified for token IDs
        forced_output_length=False,  # no ignore_eos contract on the OpenAI surface
        usage_block=True,
        version_endpoint=False,
        model_lookup=True,
        metrics_endpoint=True,  # container exposes Prometheus /metrics
        prefix_cache_risk=True,  # TensorRT-LLM context caching may be on
        prefix_cache_control=False,  # cannot be disabled on a hosted endpoint (ADR-0010)
        health_endpoint=False,
        gpu_metrics=True,  # NVML sampling works behind a container; skipped when absent
    ),
    "vllm": Capabilities(
        token_id_prompts=True,
        forced_output_length=True,
        usage_block=True,
        version_endpoint=True,
        model_lookup=True,
        metrics_endpoint=True,
        prefix_cache_risk=True,  # must be disabled and asserted at run start
        prefix_cache_control=True,
        health_endpoint=True,
        gpu_metrics=True,
    ),
}


def capabilities(kind: str) -> Capabilities:
    if kind not in CAPABILITIES:
        raise ValueError(f"unknown server kind {kind!r} (known: {sorted(CAPABILITIES)})")
    return CAPABILITIES[kind]

"""Backend capability table (ADR-009).

`server.kind` picks the backend; the harness adapts to what that backend can actually do instead
of assuming vLLM's API everywhere. Flags are read by the streaming client (prompt format, forced
output length, usage ground truth) and by `runner/env_check.py` (which probes exist, and whether a
prefix cache can make TTFT lie).

Update this table deliberately: adding a flag is a harness change, not a config change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ServerKind = Literal["mock", "mlx", "llamacpp", "vllm"]


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

    gpu_metrics: bool
    """NVML/DCGM sampling is meaningful (there is a real GPU behind this backend)."""


SERVER_KINDS: tuple[ServerKind, ...] = ("mock", "mlx", "llamacpp", "vllm")

CAPABILITIES: dict[str, Capabilities] = {
    "mock": Capabilities(
        token_id_prompts=True,
        forced_output_length=True,
        usage_block=True,
        version_endpoint=True,
        model_lookup=True,
        metrics_endpoint=False,  # added with the Phase 2 engine
        prefix_cache_risk=False,  # the mock has no cache
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
        gpu_metrics=False,
    ),
    "llamacpp": Capabilities(
        token_id_prompts=False,
        forced_output_length=False,
        usage_block=True,
        version_endpoint=False,
        model_lookup=True,
        metrics_endpoint=False,
        prefix_cache_risk=False,
        gpu_metrics=False,
    ),
    "vllm": Capabilities(
        token_id_prompts=True,
        forced_output_length=True,
        usage_block=True,
        version_endpoint=True,
        model_lookup=True,
        metrics_endpoint=True,
        prefix_cache_risk=True,  # must be disabled and asserted at run start
        gpu_metrics=True,
    ),
}


def capabilities(kind: str) -> Capabilities:
    if kind not in CAPABILITIES:
        raise ValueError(f"unknown server kind {kind!r} (known: {sorted(CAPABILITIES)})")
    return CAPABILITIES[kind]

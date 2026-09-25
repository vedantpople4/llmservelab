# ADR-0002: The gateway core is a library that runs in the benchmark process

Status: accepted (September 2026)

## Context

A standalone proxy adds a network hop and a second clock domain. Aligning proxy and client
timestamps is error-prone.

## Decision

`llmserve/gateway/core.py` is an asyncio admission controller with no HTTP dependency.
Experiments run it inside the benchmark process. A thin FastAPI wrapper (`gateway/app.py`)
exposes the same core as an OpenAI-compatible proxy for demos and external clients.

## Consequences

- All request timestamps come from one monotonic clock.
- One Phase 5 check confirms that proxy mode and in-process mode give the same latency within
  noise.
- The benchmark process is both load generator and scheduler, so client lag must be monitored
  (Phase 2).

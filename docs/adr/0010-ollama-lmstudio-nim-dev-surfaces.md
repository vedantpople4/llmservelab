# ADR-0010: Ollama, LM Studio and NIM are development surfaces

Status: accepted (September 2026)

## Context

ADR-009 opened the harness to local Mac backends (`mlx`, `llamacpp`). Three more
OpenAI-compatible servers are worth exercising the client against: Ollama and LM Studio run
locally with almost no setup, and NVIDIA NIM appears both as a container and as the hosted
API we may lean on during development. None of them can carry reportable measurements, and
each breaks a different assumption the env check used to make unconditionally.

## Decision

`server.kind` gains `ollama`, `lmstudio` and `nim`. They are development surfaces; the
analysis layer keeps treating `kind: vllm` as the only reportable backend (ADR-009). Three
capabilities change shape:

- **`health_endpoint`** — only the mock and vLLM answer `GET /health` with 200. Backends
  without it are checked for *reachability* (any HTTP response proves the connection works)
  and then verified through `/v1/models`.
- **`prefix_cache_control`** — a suspected prefix cache is a hard failure only when the
  backend can turn caching off. NIM cannot (the hosted endpoint decides), so a suspicion is
  recorded as a note in the run report instead: never silently ignored, never blocking a dev
  session. vLLM keeps the hard-failure behavior.
- **`api_key_env`** — `server.api_key_env` names the environment variable holding the bearer
  token (NIM needs one). The harness fails loudly when the variable is unset or empty instead
  of sending unauthenticated requests and debugging 401s; tokens never appear in configs,
  metadata or logs.

## Consequences

- The streaming client, env check and smoke check now run against three more real APIs, with
  capability flags — not per-backend code branches — deciding which checks apply.
- Adding a backend stays a harness change (the capability table), not a config change.
- Dev backends may produce notes where vLLM produces failures; that asymmetry is deliberate
  and is asserted by the capability tests.

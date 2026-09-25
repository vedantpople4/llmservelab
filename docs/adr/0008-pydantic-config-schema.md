# ADR-0008: Experiment configs are validated with pydantic

Status: accepted (September 2026)

## Context

Configs drive every run, and every figure must map to a committed config. A typo that silently
falls back to a default would invalidate results.

## Decision

`llmserve/config/schema.py` defines the v1 schema with pydantic models that forbid unknown
keys. `llmserve validate` and the runner share it. The config hash is SHA-256 of the fully
resolved config (defaults filled, `workload.ref` inlined) and is part of each run ID.

## Consequences

- CI validates every committed config.
- Adding a field is a schema change; bump `schema_version` for breaking changes.

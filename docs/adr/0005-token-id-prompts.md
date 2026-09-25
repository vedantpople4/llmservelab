# ADR-0005: Send prompts as token IDs with forced output length

Status: accepted (September 2026)

## Context

Text prompts go through a chat template and a tokenizer, so their token counts drift from what
was requested. Identical prompts also hit the prefix cache and make TTFT look better than it is.

## Decision

Prompts are sent to `/v1/completions` as token-ID arrays drawn from a real text corpus at a
random offset, each starting with a unique nonce block. Prefix caching is also disabled on the
server and asserted at run start. Output length is forced with `max_tokens = min_tokens` and
`ignore_eos`. The server's `usage` block is ground truth; a mismatch marks the request
`length_mismatch`.

## Consequences

- Exact lengths on vLLM. Backends without these features (ADR-009) fall back to text prompts
  and recorded lengths.
- Must be verified against the pinned vLLM version on day 1 of Phase 1.

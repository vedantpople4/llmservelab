# ADR-0001: Schedule at a gateway that caps requests in flight

Status: accepted (September 2026)

## Context

vLLM already schedules internally (continuous batching, chunked prefill, preemption). Our
scheduler can control batch composition only by patching vLLM, which ties the project to one vLLM
version.

## Decision

The research scheduler controls **admission order**: requests wait in our queue, and the
gateway admits at most `max_in_flight` of them into vLLM at a time. vLLM's built-in
`--scheduling-policy priority` is used as a cross-check. Patching vLLM's scheduler is considered
only if Phase 5 data shows admission order can't fix the observed problem.

## Consequences

- Without the cap every request passes straight into vLLM's queue and all policies measure the
  same, so the cap is mandatory whenever the gateway is enabled.
- The cap itself costs throughput if set too low; Phase 5 sweeps it and picks the smallest value
  that keeps at least 95% of uncapped throughput.
- The paper's System Design section must state that batch composition is out of scope.

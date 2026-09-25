# ADR-0003: Closed-loop and open-loop load are separate modes

Status: accepted (September 2026)

## Context

The PRD's benchmark matrix (E01–E07) is defined by concurrency, while its arrival distributions
(constant, Poisson, bursty, replay) are rates. They are different load models: under closed-loop
load a queue can never exceed the concurrency, so schedulers rarely matter.

## Decision

`load.mode: closed` takes `concurrency`; `load.mode: open` takes an `arrival` process. The
schema rejects mixing them. Open-loop rates can be given as utilization `rho = rate /
capacity_rps`, where capacity is the saturation throughput measured in Phase 3.

## Consequences

- E01–E07 and saturation curves (H1) use closed loop; scheduler evaluation (H4–H6) uses open
  loop at rho ≥ 0.8.
- Expressing load as rho makes results comparable across hardware.
- Capacity must be measured before any rho-based experiment can run.

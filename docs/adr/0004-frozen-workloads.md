# ADR-0004: Workloads are generated before a run and saved

Status: accepted (September 2026)

## Context

Scheduler comparisons must use identical requests. Generating requests on the fly couples the
workload to timing and to the scheduler.

## Decision

The generator writes `workload.parquet` (arrival offset, class, lengths, priority, SLO,
estimate) before any request is sent. Every scheduler in a comparison replays the same file.
Seeds come from `numpy.random.SeedSequence` with independent child streams for arrivals, class
assignment, lengths and prompt content; repetition r uses spawn key `(r,)`.

## Consequences

- Comparisons are paired, which lowers the number of repetitions needed.
- Changing one workload dimension does not reshuffle the others.
- Trace replay uses the same code path.

# ADR-0006: One monotonic clock for all measurements

Status: accepted (September 2026)

## Context

Wall-clock time can jump (NTP), and mixing clocks across samplers produces misaligned
timelines.

## Decision

All timestamps are `time.perf_counter_ns()` relative to run start. At run start, one pair
`(perf_counter_ns, time_ns)` is recorded in metadata to align samples and to report wall-clock
times.

## Consequences

- Samplers (NVML, `/metrics` scraping) run on the same host and use the same clock.
- Records store integer nanoseconds; metric functions convert to seconds.

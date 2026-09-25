# ADR-0007: The run, not the request, is the unit of replication

Status: accepted (September 2026)

## Context

Requests within a run share queue state, so they are not independent samples. Treating them as
independent makes confidence intervals far too narrow.

## Decision

Percentiles are computed per run. Confidence intervals come from variation across repetitions,
using paired bootstrap over repetitions for scheduler comparisons. At least 5 repetitions per
configuration and 10 for the headline comparison; P99 claims need at least 1,000 measured requests
per run.

## Consequences

- With n = 5 a two-sided Wilcoxon test cannot reach p < 0.05, so effect sizes with CIs are the
  primary reporting format.
- Run orders are interleaved and randomized to avoid drift bias.

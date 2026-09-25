# Architecture decision records

| # | Decision |
|---|---|
| [0001](0001-gateway-admission-control.md) | Schedule at a gateway that caps requests in flight |
| [0002](0002-in-process-gateway.md) | The gateway core is a library that runs in the benchmark process |
| [0003](0003-closed-and-open-loop-load.md) | Closed-loop and open-loop load are separate modes |
| [0004](0004-frozen-workloads.md) | Workloads are generated before a run and saved |
| [0005](0005-token-id-prompts.md) | Send prompts as token IDs with forced output length |
| [0006](0006-single-monotonic-clock.md) | One monotonic clock for all measurements |
| [0007](0007-run-is-the-unit-of-replication.md) | The run, not the request, is the unit of replication |
| [0008](0008-pydantic-config-schema.md) | Experiment configs are validated with pydantic |
| [0009](0009-local-development-backends.md) | Local backends on a Mac for development |

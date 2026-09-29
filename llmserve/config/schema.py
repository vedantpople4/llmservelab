"""Experiment config schema, version 1 (ADR-008).

The schema separates closed-loop load (fixed concurrency) from open-loop load (an arrival
process), per ADR-003. Unknown keys are rejected, so a typo cannot silently fall back to a
default.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from llmserve.scheduler import registry


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Length distributions (PRD §9) ---


class Fixed(_Model):
    distribution: Literal["fixed"]
    tokens: PositiveInt


class Uniform(_Model):
    distribution: Literal["uniform"]
    min: PositiveInt
    max: PositiveInt

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self


class Choice(_Model):
    """A discrete mix of lengths: the PRD's `mixed` distribution."""

    distribution: Literal["choice"]
    values: Annotated[list[PositiveInt], Field(min_length=1)]
    weights: list[PositiveFloat] | None = None

    @model_validator(mode="after")
    def _weights_match(self) -> Self:
        if self.weights is not None and len(self.weights) != len(self.values):
            raise ValueError("weights must have the same length as values")
        return self


class LogNormal(_Model):
    distribution: Literal["lognormal"]
    median: PositiveFloat
    sigma: PositiveFloat
    min: PositiveInt = 1
    max: PositiveInt

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.min <= self.median <= self.max:
            raise ValueError("need min <= median <= max")
        return self


class Empirical(_Model):
    distribution: Literal["empirical"]
    path: str


LengthDistribution = Annotated[
    Fixed | Uniform | Choice | LogNormal | Empirical, Field(discriminator="distribution")
]


# --- Workload classes (PRD §10) ---


class SLO(_Model):
    ttft_ms: PositiveFloat | None = None
    tpot_ms: PositiveFloat | None = None


class WorkloadClass(_Model):
    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    weight: PositiveFloat = 1.0
    priority: Literal["high", "medium", "low"] = "medium"
    slo: SLO = SLO()
    prompt: LengthDistribution
    output: LengthDistribution


class Workload(_Model):
    classes: Annotated[list[WorkloadClass], Field(min_length=1)]

    @field_validator("classes")
    @classmethod
    def _unique_names(cls, v: list[WorkloadClass]) -> list[WorkloadClass]:
        names = [c.name for c in v]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate workload class names: {names}")
        return v


# --- Arrival processes (PRD §9) ---


class _RateOrRho(_Model):
    """A rate in req/s, or a target utilization rho = rate / capacity (ADR-003)."""

    rate: PositiveFloat | None = None
    rho: PositiveFloat | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.rate is None) == (self.rho is None):
            raise ValueError("set exactly one of `rate` or `rho`")
        return self


class Constant(_RateOrRho):
    process: Literal["constant"]


class Poisson(_RateOrRho):
    process: Literal["poisson"]


class BurstPhase(_Model):
    duration_s: PositiveFloat
    rate: PositiveFloat


class Bursty(_Model):
    """Piecewise-constant Poisson. The PRD's default is 5 → 50 → 5 req/s over 10 s each."""

    process: Literal["bursty"]
    phases: Annotated[list[BurstPhase], Field(min_length=1)]


class Replay(_Model):
    process: Literal["replay"]
    trace: str
    time_scale: PositiveFloat = 1.0


Arrival = Annotated[Constant | Poisson | Bursty | Replay, Field(discriminator="process")]


class Load(_Model):
    mode: Literal["closed", "open"]
    concurrency: PositiveInt | None = None
    arrival: Arrival | None = None
    requests: PositiveInt | None = None
    duration_s: PositiveFloat | None = None
    # Measured saturation throughput (req/s) for this workload on this hardware (Phase 3).
    # Required when the arrival rate is given as `rho`.
    capacity_rps: PositiveFloat | None = None

    @model_validator(mode="after")
    def _mode_fields(self) -> Self:
        if self.mode == "closed":
            if self.concurrency is None:
                raise ValueError("closed-loop load needs `concurrency`")
            if self.arrival is not None:
                raise ValueError("closed-loop load cannot have `arrival`; use mode: open")
        else:
            if self.arrival is None:
                raise ValueError("open-loop load needs `arrival`")
            if self.concurrency is not None:
                raise ValueError("open-loop load cannot have `concurrency`; use mode: closed")
            if getattr(self.arrival, "rho", None) is not None and self.capacity_rps is None:
                raise ValueError("`rho` needs `capacity_rps` (measured in Phase 3)")
        if self.requests is None and self.duration_s is None:
            raise ValueError("set `requests`, `duration_s`, or both")
        return self


# --- Server, gateway, measurement ---


class Server(_Model):
    # mock: in-package simulator. mlx / llamacpp / ollama / lmstudio / nim: development surfaces
    # (ADR-009, ADR-0010). vllm: the only backend whose results may appear in the paper.
    # `kind` must stay in sync with llmserve.client.capabilities.SERVER_KINDS.
    kind: Literal["mock", "mlx", "llamacpp", "ollama", "lmstudio", "nim", "vllm"]
    endpoint: Annotated[str, Field(pattern=r"^https?://")]
    model: str
    model_revision: str | None = None
    expected_version: str | None = None
    prefix_caching: bool = False
    scheduling_policy: Literal["fcfs", "priority"] = "fcfs"
    api_key_env: Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")] | None = None
    """Environment variable holding the bearer token (NIM's `NVIDIA_API_KEY`); None = no auth."""

    @model_validator(mode="after")
    def _policy_needs_vllm(self) -> Self:
        if self.scheduling_policy == "priority" and self.kind != "vllm":
            raise ValueError("scheduling_policy: priority is a vLLM feature")
        return self


class SchedulerSpec(_Model):
    name: str = "fifo"
    params: dict[str, float | int | str | bool] = {}

    @field_validator("name")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in registry.SCHEDULERS:
            raise ValueError(f"unknown scheduler {v!r} (known: {sorted(registry.SCHEDULERS)})")
        return v


class Gateway(_Model):
    enabled: bool = False
    max_in_flight: PositiveInt = 16
    scheduler: SchedulerSpec = SchedulerSpec()
    output_estimator: Literal["oracle", "class_prior", "online", "none"] = "oracle"


class Warmup(_Model):
    requests: NonNegativeInt = 20
    settle_timeout_s: PositiveFloat = 60


class Samplers(_Model):
    gpu_hz: PositiveFloat = 10
    server_hz: PositiveFloat = 2
    dcgm: Literal["auto", "on", "off"] = "auto"


class Measurement(_Model):
    warmup: Warmup = Warmup()
    # Open loop: requests arriving in these windows are sent but excluded from statistics.
    warmup_s: NonNegativeFloat = 0
    cooldown_s: NonNegativeFloat = 0
    request_timeout_s: PositiveFloat = 600
    samplers: Samplers = Samplers()


class ExperimentConfig(_Model):
    schema_version: Literal[1]
    experiment: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")]
    description: str = ""
    seed: NonNegativeInt
    repetitions: PositiveInt = 5
    server: Server
    gateway: Gateway = Gateway()
    load: Load
    workload: Workload
    measurement: Measurement = Measurement()

    @model_validator(mode="after")
    def _scheduler_needs_gateway(self) -> Self:
        if not self.gateway.enabled and self.gateway.scheduler.name != "fifo":
            raise ValueError(
                "a scheduler has no effect with the gateway disabled; set gateway.enabled: true"
            )
        return self

    def canonical(self) -> dict[str, Any]:
        """JSON-compatible dump with every default filled in; the input to `config_hash`."""
        return self.model_dump(mode="json")

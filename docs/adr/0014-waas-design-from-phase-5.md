# ADR-0014: WAAS v1 is an age-weighted cost order with a bypass bound

Status: proposed (November 2026)

## Context

The plan (Phase 6) lists a starting design space for WAAS and requires the final design to come
from Phase 5 evidence. That evidence is `matrix-mixed` (mixed workload, in-flight cap 16, ρ 0.5,
0.8 and 0.95, FIFO, SPF, SJF and gateway Priority). All numbers come from `llmserve report`,
section "Scheduling baselines (H4)".

1. **Cost ordering helps at ρ 0.8.** Interactive P95 TTFT is 33.3 s under FIFO, 14.8 s under SPF,
   15.7 s under SJF and 13.2 s under Priority.
2. **Greedy ordering starves at ρ 0.95.** Generation-heavy P95 TTFT is 14.3 s under FIFO, 79.1 s
   under SJF and 106.0 s under Priority, with gateway waits up to 178 s and 229 s. SPF does not
   starve that class only because its prompts are short.
3. **Most of the remaining interactive tail is inside vLLM.** At ρ ≥ 0.8 the interactive P95 of
   dispatch to first token is 10 to 16 s under every policy. Interactive requests dispatched
   while no long-context prompt is prefilling have a server P95 of 2.2 to 2.7 s. Requests
   dispatched behind more than 8k long-context prefill tokens have a P95 of 23 to 29 s. 89% of
   the interactive requests whose server TTFT exceeds 5 s were dispatched behind a long-context
   prefill. vLLM schedules a running prefill's next chunk before any waiting prompt, so no
   admission order can put a short prompt ahead of a long prefill that is already running.
4. **There is no KV pressure at cap 16.** No run preempted a request, and KV usage peaked at 64%.
5. **SLOs are placeholders** (`configs/workloads/mixed.yaml`), so slack is not yet meaningful.

## Decision

WAAS v1 orders the gateway queue by an age-weighted cost and bounds how often any request can be
overtaken. It drops the plan's urgency term and KV-aware gate until evidence calls for them, and
treats prefill interference as a separate experiment.

- **Order.** Admit the request with the smallest `arrival_time + cost_weight · service_time`,
  where `service_time` is the cost model's prefill plus decode estimate at a fixed decode batch
  (as SJF uses). `cost_weight = 0` is FIFO, and a large weight approaches SJF. The key does not
  change while a request waits, which keeps decisions within the 100 µs budget at depth 1,000.
- **Bypass bound.** Each waiting request counts the dispatches of later arrivals. When the
  oldest waiting request reaches `max_bypass`, it is admitted next. The oldest waiting request
  always has the largest count, since every later arrival dispatched ahead of a younger waiting
  request was also dispatched ahead of the oldest. So checking only the oldest enforces the
  bound for every request.
- **Output estimate.** The gateway's estimator supplies `output_tokens_est`. `class_prior` uses a
  per-class median from earlier runs. `online` keeps a per-class EWMA of observed output tokens,
  updated on every completion. Both join the existing `oracle` and `none` for the ablation.
- **Prefill interference.** This is a server behavior, so it is tested on the server first.
  vLLM's `long_prefill_token_threshold` caps a long prefill's chunk per step and leaves budget
  for waiting prompts, and vllm-metal honors it. A FIFO sweep of the threshold at ρ 0.95 comes
  before any gateway prefill gate. If the threshold removes the tail, WAAS is evaluated on a
  server with it set. If it does not, a gateway gate that holds a long prompt while another long
  prefill is in flight is built next, which needs a first-token hook from the client.
- **Dropped for v1.** The urgency term waits for SLOs set from measured isolated latency. The
  KV-aware gate waits for a configuration that preempts, such as a larger cap or a smaller KV
  pool (H6).

## Consequences

- WAAS has two parameters, `cost_weight` and `max_bypass`, tuned on tuning seeds and frozen
  before the held-out evaluation (plan Phase 6 item 7).
- The ablation is smaller than the plan's. It removes the cost term (`cost_weight = 0`), the
  bypass bound (`max_bypass` unbounded) and each estimator.
- The server-threshold result may move the headline comparison from "gateway scheduling against
  FIFO" to "gateway scheduling on a server configured for mixed prompts". Both are reported.

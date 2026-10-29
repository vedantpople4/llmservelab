# Where the time goes: Phase 4 findings

This is the written answer the Phase 4 exit check asks for (IMPLEMENTATION_PLAN.md, Phase 4;
PRD §31 and §42). It covers each workload class on the measurement platform: the M5 MacBook Air
with vllm-metal 0.31.0, Qwen2.5-7B-Instruct-AWQ, chunked prefill with a 2048-token step budget
(ADR-0013).

Every number below comes from `llmserve report` (sections named in brackets) or from
`llmserve trace`. Both read the stored runs, so any reader with `results/` can regenerate them.

## Short answer

Decode dominates end-to-end time for every class, at 79% to 98% of the mean E2E. Time to first
token is the part schedulers can move, and its main cause differs by load shape.

- **Open loop, light KV use (128/128 at 0.1 to 0.4 req/s).** Requests never wait in vLLM's
  queue. Their server wait is prefill sharing engine steps with running decodes.
- **Closed loop with fixed lengths (E02, E03, the concurrency sweep).** Requests finish together,
  so their replacements arrive together. vLLM prefills each cohort as one batch, and every
  member's TTFT is the prefill of the whole cohort. This is the "TTFT at C ≥ 8 exceeds prefill
  cost" anomaly from ADR-0013. It is not queueing.
- **Mixed workload (E07, profile-mixed).** The KV cache fills, so requests queue in vLLM
  (1.36 s mean). Then a long-context prompt's prefill stalls every running request at once, for
  up to 5.6 s per step.

## Per class

Means in seconds per measured request. E01 to E07 are closed loop
[Where the time goes (E01–E07)]. The mixed profile run is open-loop Poisson at 0.16 req/s,
ρ 0.8 [Engine and power profile].

| Class | Run | Server wait | Prefill | Decode | E2E | Decode share |
|---|---|---|---|---|---|---|
| 128/128 | E01, C = 1 | 0.00 | 0.29 | 6.4 | 6.7 | 96% |
| 128/128 | E02, C = 8 | 1.90 | 0.33 | 20.5 | 22.7 | 90% |
| 128/128 | E03, C = 32 | 8.60 | 0.33 | 44.6 | 53.5 | 83% |
| 1024/128 | E04, C = 8 | 4.44 | 1.93 | 33.0 | 39.3 | 84% |
| 4096/128 | E05, C = 8 | 11.68 | 7.68 | 70.9 | 90.3 | 79% |
| 1024/512 | E06, C = 8 | 4.20 | 1.93 | 96.3 | 102.5 | 94% |
| interactive | E07, C = 32 | 7.33 | 0.67 | 81.3 | 89.3 | 91% |
| long_context | E07, C = 32 | 8.48 | 7.13 | 82.8 | 98.4 | 84% |
| generation_heavy | E07, C = 32 | 6.86 | 0.56 | 456.3 | 463.7 | 98% |
| interactive | profile-mixed | 4.10 | 0.67 | 45.9 | 50.6 | 91% |
| long_context | profile-mixed | 4.28 | 6.77 | 55.9 | 66.9 | 84% |
| generation_heavy | profile-mixed | 2.36 | 0.56 | 184.8 | 187.7 | 98% |

The phases are defined in `llmserve/analysis/breakdown.py`. Prefill is the unloaded prefill the
H3 fit predicts for the prompt length, and server wait is the rest of dispatch to first token.
There is no gateway yet, so gateway wait is zero throughout.

### 128/128, short prompts and outputs

At C = 1, TTFT is its own prefill (0.29 s against a 0.33 s fit) and decode runs at about 50 ms
per token. As concurrency rises, decode slows because every step carries more sequences. TPOT P50
goes from 51 ms at C = 1 to 161 ms at C = 8 and 351 ms at C = 32 [Baseline experiments].
Server wait grows from 0 to 8.6 s.

Under open-loop Poisson arrivals up to the 0.4 req/s capacity, vLLM's own queue time is 0 at
every rate, and its scheduling-to-first-token time is 0.43 to 0.93 s [Engine and power profile].
New requests do not queue. They join a step that is also decoding.

### Long prompts (1024 and 4096 tokens)

Prefill becomes a real share of TTFT, at 1.9 s for 1024 tokens and 7.7 s for 4096 tokens. Server
wait still exceeds it (4.4 s and 11.7 s), because each prompt above the 2048-token budget needs
several steps, and the other requests' decodes and prefills share those steps. Decode is still
most of E2E (79% to 84%), and it is slower than for short prompts at the same C. TPOT P50 is
261 ms at 1024 tokens and 581 ms at 4096 tokens, against 161 ms for 128 tokens, because attention
reads a larger KV cache every step.

### Long outputs (E06, generation_heavy)

Decode is the whole story, at 94% to 98% of E2E. A generation_heavy request in E07 spends 456 s
decoding 512 tokens. Each token waits for a step that carries every other running sequence.

### Mixed workload

KV usage reaches 100% in the mixed profile run, and vLLM then holds new requests in its waiting
queue. Its own timers split dispatch to first token (6.11 s mean across classes) as follows
[Engine and power profile]:

- 1.36 s in vLLM's waiting queue.
- 3.60 s from scheduling to first token. The unloaded fit predicts 2.18 s of that, so the
  remainder is prefill slowed by shared steps.
- About 1.15 s outside vLLM's request timers. That covers HTTP, tokenization and streaming,
  plus the difference between vLLM's whole-rep means and the measured window.

Three requests traced from the run show the mechanism. Each is the 90th-percentile TTFT of its
class.

```
llmserve trace results/profile-mixed-r0p16/20261029T052033Z-7212c392-66e64ee \
  --request r000126 --cost-model results/_report-session7/cost_model.json
class long_context · prompt 2048 tokens
  +0.000  arrival      vLLM running 7, waiting 1, KV 25.1%
 +23.182  first token  TTFT 23182 ms = server wait 19385 + unloaded prefill 3796 ms
                       vLLM running 13, waiting 1, KV 44.6%
+131.856  last token   TPOT 855.7 ms · ITL P50 386.7 ms, max 5577.0 ms at token 26
```

The interactive request r000129 (474 tokens) and the generation_heavy request r000130 have the
same 5.6 s maximum gap at token 25 or 26. One step stalled every running request together, the
signature of a large prefill entering the batch. The interactive request waited 15.2 s on top of
its 0.9 s prefill. That is the head-of-line blocking H2 measured: its P95 TTFT is 461% higher in
the mixed workload than in an interactive-only one [Head-of-line blocking (H2)].

TTFT does not rise steadily with KV usage at dispatch [figures/kv_vs_ttft.png]. Median TTFT is
1.3 s below 25% KV, 7.7 s at 25% to 50% and 13.2 s at 50% to 75%, then falls to 5.3 s above 75%
(15 requests). Which prompts share the step matters more than how full the cache is. These bins
come from `profile.kv_at_dispatch` on the mixed run, not from a report table.

## The C ≥ 8 TTFT anomaly

ADR-0013 recorded TTFT far above prefill cost at C ≥ 8 for 128/128, such as 3.5 s at C = 16 for
a 0.23 s prefill. The concurrency sweep shows vLLM's waiting queue at 0 for every C ≤ 16, so
the cause is not queueing.

Grouping requests by dispatch time explains it [Concurrency sweep, Why TTFT exceeds one prompt's
prefill]. Requests dispatched within 0.25 s of each other form a cohort.

| C | Median cohort size | Fit: one prompt (s) | Fit: whole cohort (s) | Dispatch to first token, P50 (s) |
|---|---|---|---|---|
| 1 | 1 | 0.33 | 0.33 | 0.36 |
| 2 | 2 | 0.33 | 0.56 | 0.80 |
| 4 | 4 | 0.33 | 1.01 | 1.37 |
| 8 | 8 | 0.33 | 1.93 | 2.40 |
| 16 | 16 | 0.33 | 3.80 | 4.14 |
| 32 | 15 | 0.33 | 3.56 | 8.07 |
| 64 | 15 | 0.33 | 3.56 | 9.15 |
| 128 | 15 | 0.33 | 3.56 | 10.90 |

In a closed loop with fixed 128/128 lengths, all C requests start together, decode in lockstep
and finish in the same step. The client replaces them at once, and vLLM prefills the C new prompts
as one batch. Each member's TTFT is therefore the prefill of C × 128 tokens. The fit for that
batch tracks the observed TTFT to within 0.5 s up to C = 16. The remainder is about one decode
step, since the first tokens within a cohort spread over 70 to 230 ms.

Above C = 16 the step budget caps a cohort at 15 prompts (1920 tokens). A 16th prompt would need
2048 tokens. My reading is that the decodes running in the same step also use part of the budget,
but I have not checked that against vLLM's scheduler code. The remaining requests wait for later
steps, which is where vLLM's waiting queue first becomes nonzero (C = 32).

A traced C = 16 request shows the same thing. It had 3.9 s of server wait on a 0.33 s prefill,
with vLLM at 16 running and 0 waiting throughout.

```
llmserve trace results/concurrency-p128-c016/20261025T034915Z-4d5c9a83-db2445a \
  --request r000139 --rep 2 --cost-model results/_report-session7/cost_model.json
  +0.000  dispatch     vLLM running 16, waiting 0, KV 7.8%
  +4.251  first token  TTFT 4250.7 ms = server wait 3918.5 + unloaded prefill 332.1 ms
```

This is an artifact of the load shape, not of the engine. Phase 5 should not read closed-loop
fixed-length TTFT as queueing delay.

## Engine and Mac

[Engine and power profile]

- **GPU active share** rises from 59% at 0.1 req/s to 91% at 0.3 req/s for 128/128. It reads
  76% at 0.4 req/s, and a rerun reproduced that, so heat does not explain it. Each point is one
  rep with its own workload sample. Treat the drop as variance until more reps exist. Active share
  only means the GPU was busy. It reads high during memory-bound decode too.
- **Energy per output token** falls from 0.54 J at 0.1 req/s to 0.17 J at 0.4 req/s, because the
  Mac's baseline draw of about 8 W is spread over more tokens. Power barely changes with load
  (7.8 to 8.7 W).
- **Thermal pressure** sits above Nominal for the whole of every run from 0.2 req/s up. The
  thermal probe's drift still stays within ±10% on all five runs. The GPU clock varies between
  roughly 600 and 1400 MHz from sample to sample. Its 5-minute means stay between 920 and
  1090 MHz over the 30-minute mixed run with no downward trend. The lowest one falls between
  600 s and 900 s, during the KV peak [figures/profile_timeline.png].

## Caveats

- **No Metal System Trace.** The plan asked for an Instruments trace. This Mac has only the
  Command Line Tools, so `xctrace` is unavailable. vLLM's `/metrics` timing histograms, its
  iteration-token histogram and `llmserve trace` timelines stand in for it. They show what each
  request waited for, not GPU kernel timing.
- **The cohort effect touches H1.** H1 was refuted with a P99 TTFT knee at C = 2, before the
  throughput plateau at C = 4. Closed-loop cohorts make TTFT grow with C from C = 2 even with no
  queueing, so that knee measures batched prefill, not saturation. H1 should be retested under
  open-loop arrivals or varied lengths.
- **H2's mixed run used a 0.15 req/s capacity**, not the 0.2 req/s the capacity suite measured.
  Its ρ 0.85 is therefore 0.13 req/s, about ρ 0.64 of measured capacity.
- **E05 ran 50 requests per rep**, so its P99 interval is wide (31 to 52 s).
- **The profile runs are one rep each**, and the vLLM totals and energy cover whole reps,
  including warm-up and cool-down.

## What this means for scheduling (Phase 5 onward)

- The lever on TTFT under mixed load is which prompts enter a step together. A 2048-token prompt
  blocks every running request for a full step. Admission order alone cannot fix that, since
  prefill already shares steps. A gateway that keeps long prefills from landing together, or caps
  in-flight long-context requests, attacks the 5.6 s stalls directly.
- KV pressure creates real queueing in the mixed workload, so the in-flight cap and the KV
  headroom it leaves are the variables to sweep.
- Benchmarks for scheduler comparisons should use open-loop arrivals. Closed-loop fixed-length
  load synchronizes requests and hides the effect a scheduler would have.

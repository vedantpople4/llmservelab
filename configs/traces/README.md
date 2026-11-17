# Request traces

Each file is a trace CSV (`llmserve/workload/trace.py`): `offset_s,prompt_tokens,output_tokens`,
one request per row in arrival order. A config replays one by setting `workload.trace` and a
`replay` arrival to the same path.

| File | Requests | Raw rate | Inter-arrival CV |
|---|---|---|---|
| `azure-conv-2023-steady.csv` | 150 | 4.36 req/s | lowest of the trace's 150-request blocks |
| `azure-conv-2023-bursty.csv` | 150 | 2.96 req/s | 1.45, the highest |

`scripts/azure_trace.py` regenerates both, plus `configs/output_priors/azure-conv.json` (per-class
median output length of the requests outside the two windows).

## Attribution

Both files are cut from the Azure LLM inference trace 2023 (conversation),
`AzureLLMInferenceTrace_conv.csv`, published in the
[Azure Public Dataset](https://github.com/Azure/AzurePublicDataset) under the
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
license. It was collected for Patel et al., "Splitwise: Efficient Generative LLM Inference
Using Phase Splitting", ISCA 2024.

Changes from the original: one request longer than 10,240 tokens was dropped, timestamps became
offsets from each window's first request, and only two windows of 150 consecutive requests are
kept.

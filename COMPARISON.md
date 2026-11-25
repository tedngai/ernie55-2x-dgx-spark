# ERNIE-5.5-Flash vs Hy4-flash on 2× DGX Spark

Both run on the same two GB10 Sparks (128 GB each, 200 Gb QSFP). They cannot
run at the same time; switching costs a 5–15 minute restart. ERNIE numbers
are from the [RoCE + wider CUDA graph tests](DECODE-TUNING.md) on 2026-11-25;
Hy4 numbers are from its [evaluation report](../hy4-flash-2x-dgx-spark/EVAL-REPORT.md)
and records (2026-11-22/23). These are different prompt suites, not a matched
quality or performance benchmark.

## What they are

| | **ERNIE-5.5-Flash** | **Hy4-flash** |
|---|---|---|
| Vendor | Baidu | Tencent |
| Total params | ~280B target + ~6.4B DSpark draft | 220B (incl. 64B Engram tables) |
| Active params/token | ~9B | 12B |
| Layers | 43 | 42 |
| Hidden | 4096 | 3072 |
| Experts | 512, top-10 | 512, top-16 |
| Expert intermediate | 2048 (latent 2048) | 1536 |
| Attention | 37 HCA + 6 DSA sparse MLA, window 128 | sparse MLA, window 512 |
| Modalities | text | **image + text** |
| Context served | **400,000** | **500,000** |
| Quantization | MXFP4 experts + FP8 blockwise rest | NVFP4 experts + FP8 Engram + BF16 rest |
| Checkpoint | 155 GiB | 164.7 GB (153.4 GiB) |

## Serving layout

| | ERNIE | Hy4 |
|---|---|---|
| Parallelism | TP2 (no EP), NCCL RoCE | TP2 + **EP2** |
| Speculative | DSpark, 5 tokens, probabilistic | DFly, 5 tokens |
| Draft acceptance | ~53% code, ~36% prose (C1) | ~89% code, ~12–16% chat |
| GPU weights/node | **80.9 GiB** | 51.7 GiB + Engram tables in host RAM |
| KV pool | **~1.05M tok** (2.62× @400K) | 19.32 GiB = **1,522,579 tok** |
| Concurrency at max window | 2.51–2.62× @400K | 3.05× @500K |

Hy4 has a smaller per-node GPU weight footprint and keeps its Engram tables
in host RAM. Its EP layout assigns whole experts to ranks, whereas ERNIE's
SiTU-capable b12x kernel requires TP shards of all routed experts.

## Measured speed

| Workload | ERNIE | Hy4 |
|---|---|---|
| Decode, code (C1) | **49.0 tok/s** | **36–55 tok/s** (code / JSON suite) |
| Decode, prose (C1) | **38.7 tok/s** | 17–29 tok/s (content-dependent) |
| Decode, long context (C1) | Not benchmarked on the new profile | 45–55 tok/s @200–250K |
| Prefill, short | 4.62 s (~1.73K tok/s) for 7,986 tokens | 1.16–1.57K tok/s |
| Long prefill | 399,403 tokens: 336.45 s (~1.19K tok/s) | 490K: 489 s (1,002 tok/s) |

The older ERNIE 19–21 tok/s and ~253 s long-needle result were from the TCP
profile. The old needle's exact prompt token count was not captured, so it
is not a matched 200K comparison.

Both models pass arithmetic, needle retrieval, and multi-request checks. Hy4
also passes image, JSON-schema, tool-call, and thinking-budget checks; ERNIE is
text-only. Neither has a standardized quality score (GPQA/IFEval) on this box.

## Thinking mode and latency

Hy4's published benchmark runs **thinking off** (`reasoning_effort: no_think`).
ERNIE thinks by default. For illustration, these historical ERNIE timings
are from before RoCE; the Hy4 values in this table are estimates:

| Prompt | ERNIE thinking on | ERNIE `enable_thinking:false` | Hy4 (thinking off, from its 45 tok/s) |
|---|---|---|---|
| 15% of 240 | 1.47 s (35 tok) | **0.70 s** (10 tok) | ~0.4–0.7 s |
| Python reverse one-liner | 5.74 s (117 tok) | **2.07 s** (14 tok) | ~0.6 s |
| Short chat paragraph | ~7–9 s | ~3–5 s | 3–6 s |

## Verdict

**ERNIE now reaches the Hy4-class decode range** on these short tests.
Hy4 still has the larger served window and KV pool, and supports images.
Neither a matched latency test nor a matched quality test has established a
winner.

For ERNIE, pass this for latency-sensitive calls:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

Keep thinking on for math, multi-step reasoning, or code synthesis where the
extra tokens buy accuracy. A true head-to-head quality run (same prompts,
GPQA/IFEval-style) has not been done for either model.

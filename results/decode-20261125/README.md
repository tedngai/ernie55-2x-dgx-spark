# Benchmark receipts — 2026-11-25

One JSON object per completed wave from [`bench-decode.py`](../../bench-decode.py).
Every row includes per-stream `ttft_s`, `tokens`, `prompt_tokens` and
`decode_tok_s`, plus server-side speculative-decoding counters:

- `median_stream_decode_tok_s`: median per-request decode rate.
- `aggregate_decode_tok_s`: all completion tokens in the wave divided by the
  common decode interval.
- `accepted_per_draft`: accepted draft tokens per verification step.
- `acceptance_pct`: accepted draft tokens / proposed draft tokens.
- `acceptance_by_position_pct`: positional acceptance; the denominator is the
  number of verification steps, so entry 0 is the first draft token.

Files:

| File | Transport | DSpark | CUDA graph sizes | KV tokens |
|---|---|---:|---|---:|
| `tcp-graph8.jsonl` | TCP over QSFP | 5 | 1,2,4,8 | 1,044,376 |
| `roce-graph8.jsonl` | RoCE | 5 | 1,2,4,8 | 1,015,568 |
| `roce-graph18.jsonl` | RoCE | 5 | 1,2,4,6,8,12,18 | 1,002,900 |
| `roce-dspark3.jsonl` | RoCE | 3 | 1,2,3,4,6,8,9,12 | 944,401 |

`nccl-probe.log` contains the two-rank CUDA-graph all-reduce probe used to
compare transports while the model was stopped. Payloads are 8 KiB, 48 KiB,
144 KiB and 3 MiB; values are microseconds per all-reduce.

`final-verification.log` is the boot/restart log of the saved final profile,
including the 399,403-token retrieval result. `verify-final.log` is
`./verify.sh` against that boot, and `final-c3.jsonl` is one three-stream
sanity wave.

Run conditions: two DGX Sparks, TP2, b12x W4A8, 400K context, four NCCL
channels, 192 output tokens per stream, `temperature=0`,
`enable_thinking=false`, three waves per workload/concurrency after one
warmup wave. Hosts were otherwise idle.

# ERNIE decode tuning on two DGX Sparks

## Why transport mattered (2026-11-25)

The original 400K profile ran TP2, b12x W4A8, DSpark 5 probabilistic, and a
CUDA graph limit of 8 tokens. Neither container had `/dev/infiniband`. Setting
`NCCL_IB_HCA` and using the QSFP IP selects the link, but does not give NCCL
access to RoCE. Hy4's launcher passes through the device; ERNIE's did not.

With the serving containers stopped, the same NCCL 2.29.7 used by vLLM
measured the following CUDA-graph all-reduce times (µs, two ranks). Four
channels beat one with RoCE. GPUDirect RDMA was disabled; the win is from
RoCE transport itself.

| Payload | TCP, 4 channels | RoCE, 4 channels | RoCE, 1 channel |
|---|---:|---:|---:|
| 8 KiB | 705.8 | 24.4 | 34.5 |
| 48 KiB | 614.5 | 32.0 | 64.0 |
| 144 KiB | 668.3 | 54.5 | 113.7 |
| 3 MiB | 1,882.2 | 335.3 | 469.0 |

## Serving A/B: TCP vs RoCE

`python3 bench-decode.py --concurrency 1 2 3 --repeats 3`, short prompts,
thinking off, temperature 0, 192 output tokens per request. Both builds use
the same checkpoint, 400K limit, four channels, DSpark 5 and graph cap 8.
Rates are median aggregate tok/s, excluding time to first content.

| Workload | Streams | TCP | RoCE | Change |
|---|---:|---:|---:|---:|
| Code | 1 | 22.39 | **53.83** | +140% |
| Prose | 1 | 16.16 | **36.67** | +127% |
| Code | 2 | 32.55 | **43.93** | +35% |
| Prose | 2 | 24.04 | **31.17** | +30% |
| Code | 3 | 53.09 | **65.99** | +24% |
| Prose | 3 | 36.33 | **46.06** | +27% |

Both ranks logged `Using network IB` and traffic via `NET/IB`. At C1,
median draft acceptance was virtually unchanged (code 2.92 → 3.06 accepted
per verify, prose 1.73 → 1.72); the estimate for a verify/draft step fell
from ~169–176 ms to ~74–75 ms. The 400K configuration retained 1,015,568 KV
tokens (2.54× at 400K), and a 7,986-token retrieval smoke test passed.

## Serving A/B: wider CUDA graphs

The original graph cap of 8 tokens only fit the six-token target verification
batch for one DSpark 5 request. `GRAPH_SIZES=1,2,4,6,8,12,18` adds full
replay for two and three requests, without disabling breakable graphs or
enabling `torch.compile`. Startup confirmed three full target graphs and
three full draft graphs.

| Workload | Streams | RoCE, cap 8 | RoCE, cap 18 | Change |
|---|---:|---:|---:|---:|
| Code | 1 | 53.83 | 48.95 | −9% |
| Prose | 1 | 36.67 | 38.66 | +5% |
| Code | 2 | 43.93 | **77.33** | +76% |
| Prose | 2 | 31.17 | **56.39** | +81% |
| Code | 3 | 65.99 | **99.02** | +50% |
| Prose | 3 | 46.06 | **71.15** | +54% |

Single-stream verify time remained ~73–75 ms; differences in draft acceptance
explain most of the C1 spread across the three waves. The expanded profile
kept 1,002,900 KV tokens at startup (2.51× at 400K).

## Serving A/B: three vs five draft tokens

Both runs used RoCE and full target/draft graph coverage for all three
concurrency levels. The shorter draft reduced estimated C1 verification
time to ~65 ms, but fewer accepted tokens offset that for single-stream
decode.

| Workload | Streams | DSpark 5 | DSpark 3 | Change with 3 |
|---|---:|---:|---:|---:|
| Code | 1 | **48.95** | 45.11 | −8% |
| Prose | 1 | 38.66 | 37.38 | −3% |
| Code | 2 | 77.33 | 74.86 | −3% |
| Prose | 2 | 56.39 | **61.70** | +9% |
| Code | 3 | 99.02 | 98.24 | −1% |
| Prose | 3 | 71.15 | **83.11** | +17% |

Keep **five tokens as the general default**. `NUM_SPEC=3 ./start.sh`
selects the tested alternative for workloads dominated by concurrent
prose, with 944,401 KV tokens in that run (2.36× at 400K). Both depths
passed the 7,986-token retrieval smoke test.

Raw receipts: [`results/decode-20261125/`](results/decode-20261125/).

## Reproducing the tests

Keep the checkpoint, 400K limit, and 4096 prefill batch fixed. Test the
transports with `RDMA=0 GRAPH_SIZES= ./start.sh` and
`RDMA=1 GRAPH_SIZES= ./start.sh` (cap 8). For graph coverage, repeat on RoCE with
`GRAPH_SIZES=1,2,4,6,8,12,18`; for DSpark 3, use `NUM_SPEC=3` and
`GRAPH_SIZES=1,2,3,4,6,8,9,12`. Run
`python3 bench-decode.py --concurrency 1 2 3 --repeats 3` in each profile.
Check `/health`, startup's 400K KV capacity, and
`python3 check-needle.py --tokens 8000`. `python3 check-needle.py` checks
near-window retrieval.

`RDMA=1` passes through `/dev/infiniband`, raises the pinned memory limit,
and forces NCCL's built-in `IB` transport. Check `Using network IB` and
`via NET/IB` on both ranks; `Bootstrap : Using enp1s0f1np1` alone does
not prove RDMA.

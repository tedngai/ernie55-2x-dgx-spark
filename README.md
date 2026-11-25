# ERNIE-5.5-Flash-Preview FP4 on 2× DGX Spark

**Status: serving on two GB10 / sm_121 Sparks.** TP=2 with NCCL RoCE over the
200 Gb QSFP link, **400,000-token context**, FP8 `fp8_ds_mla` KV cache, MXFP4
routed experts through b12x W4A8, and the checkpoint's bundled DSpark draft as
5-token speculative decoding.

- **API:** `http://<head-host>:8000/v1` (replace `<head-host>` with the head
  node's LAN, Wi-Fi or Tailscale address)
- **Model ID:** `baidu/ERNIE-5.5-Flash-Preview`
- **Container:** `ernie55` (same name on both nodes)

Measured single-stream decode is **49.0 tok/s on code** and **38.7 tok/s on
prose**, with **99.0 / 71.2 tok/s aggregate at three streams**. See
[Measured performance](#measured-performance) for conditions and receipts.

Hy4 remains stopped but intact on both nodes as
`hy4-flash-nvfp4-vllm-tp2-500k`; see [Rollback to Hy4](#rollback-to-hy4).
The earlier ERNIE-vs-Hy4 write-up is in [COMPARISON.md](COMPARISON.md), and
the tuning experiments are in [DECODE-TUNING.md](DECODE-TUNING.md).

## Quick start

```bash
curl http://<head-host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "baidu/ERNIE-5.5-Flash-Preview",
    "messages": [{"role": "user", "content": "Explain why the sky is blue."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "max_tokens": 300,
    "temperature": 0
  }'
```

Python with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://<head-host>:8000/v1", api_key="not-needed")
answer = client.chat.completions.create(
    model="baidu/ERNIE-5.5-Flash-Preview",
    messages=[{"role": "user", "content": "Write a Python LRU cache."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    max_tokens=512,
)
print(answer.choices[0].message.content)
```

**Thinking is on by default.** ERNIE emits reasoning tokens unless the request
sets `chat_template_kwargs.enable_thinking=false`. On short prompts the
difference is large: a one-line math question took 1.47 s with thinking and
0.70 s without; a code one-liner took 5.74 s and 2.07 s. Use `false` for
latency-sensitive calls and leave it on for hard reasoning.

`reasoning_effort="none"` is not accepted by the template; use the
`chat_template_kwargs` form above. The server's `--reasoning-parser ernie5` and
`--tool-call-parser ernie5` expose reasoning content and OpenAI-style tool
calls.

## What is served

| | Value |
|---|---|
| Model | `baidu/ERNIE-5.5-Flash-Preview` (FP4 checkpoint, 155 GiB) |
| Parameters | ~280B total, ~9B active per token, plus ~6.4B DSpark draft |
| Layers | 43 |
| Hidden size | 4096 |
| Experts | 512 routed, top-10, shared expert 1, latent 2048 |
| Activation | **SiTU** (Baidu), not SiLU |
| Attention | 37 HCA layers + 6 DSA sparse-MLA layers, window 128 |
| Quantization | MXFP4 routed experts; FP8 blockwise projections |
| Served context | **400,000 tokens** |
| KV cache | FP8 `fp8_ds_mla`, 256-token blocks |
| Admission slots | 3 |
| GPU weights + non-torch | 80.9 GiB per node |
| Image | `ernie55:v0.32-b12x-sink`, 32.9 GB, on both nodes |

At the final boot the KV pool held **1,048,411 tokens** (**2.62×** the 400K
window), so two full-window requests fit with headroom. The pool varies
slightly by boot with unified-memory availability; the benchmark boot held
1,002,900 tokens (2.51×).

## Hardware and topology

- 2× NVIDIA DGX Spark (GB10, compute capability 12.1 / `sm_121`), 128 GB
  unified memory each.
- 200 Gb QSFP link configured for RoCE between the two nodes.
- Head and worker addresses are deployment-specific. Set `HEAD_IP`, `WORKER`,
  and `WORKER_IP` in an untracked `.env.local`; see `.env.example`.
- vLLM 0.32.0, NVIDIA PyTorch 26.07 base image, NCCL 2.29.7.
- TP=2, one process per Spark, launched with vLLM's multi-node MP mode.

## Why upstream does not run this model on GB10

Upstream's ERNIE plugin hard-wires two paths that reject `sm_121` or produce
wrong results. Two source patches, layered into the image, fix them.

### 1. Sparse MLA with per-head sinks (`attention_patched.py`)

Upstream routes ERNIE's 6 DSA layers to `FlashMLASparseBackend`, which rejects
`sm_121`. FlashInfer's SM120 sparse-MLA kernel accepts the per-head `sinks`
tensor and applies the FlashMLA V4 convention exactly
(`out *= sigmoid(lse - sink)`). The patch:

- adds `Ernie5_5FlashInferSM120Impl` / `Ernie5_5FlashInferSM120Backend`, which
  pass `sinks=` into the SM120 sparse kernel;
- routes DSA layers to that backend when capability major is 12;
- routes the 37 HCA layers to `Ernie5_5HCAFlashInferSM120Attention`
  (`DeepseekV4FlashInferSM120Attention`), which already passes sinks.

### 2. SiTU MoE W4A8 kwargs (`situ_patched.py`)

b12x 1.5.1's SiTU MoE wrapper omits the W4A8 layout kwargs
(`w4a8_n64_repacked`, `w4a8_n64_tail`) that its own dispatcher passes. The
patched wrapper forwards all parent kwargs, exactly like the SiLU wrapper. It
replaces `b12x/moe/_shared/kernels/situ.py`.

### Why not expert parallelism

ERNIE's routed experts use **SiTU**; the only MXFP4 MoE kernel on `sm_121`
that implements SiTU is b12x 1.5.1. b12x requires `use_ep=False` and reports
`supports_expert_map=False`, so enabling EP would route tokens to the wrong
experts. FlashInfer CUTLASS/TRTLLM and the MoE-EP CuTeDSL/DeepGEMM kernels
declare `_supported_activations = {SILU}`; CUTLASS rejects this checkpoint with
`does not support MoEActivation.SITU`. The current configuration therefore
keeps a TP shard of all 512 experts per node.

## Speculative decoding (DSpark)

The checkpoint ships a draft under `model.mtp.*` (9,358 tensors, including a
trained `confidence_head.projection.weight`). It is not a separate model: the
plugin patches `SpeculativeConfig` so `method=dspark` with no `model` builds
the draft from the target checkpoint (`dspark_bundled_in_target`). DFlash
(`dfly`) is Hy4's method; ERNIE uses DSpark.

`./start.sh` defaults to:

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":5,
                       "draft_sample_method":"probabilistic"}'
```

Measured on the final profile with `temperature=0`:

| Workload | Accepted draft tokens / verify | Acceptance | Position 0 | Position 4 |
|---|---:|---:|---:|---:|
| Code | 2.66 of 5 | 53.2% | 82.7% | 32.1% |
| Prose | 1.82 of 5 | 36.5% | 76.5% | 4.4% |

Five tokens is the general default. `NUM_SPEC=3 ./start.sh` selects the tested
alternative: it improved three-stream prose aggregate throughput by 17% but
slowed single-stream code by 8%, and it reduces the KV pool. See
[DECODE-TUNING.md](DECODE-TUNING.md) for the full A/B tables.

## Measured performance

All decode numbers below are **median aggregate tok/s over three waves**,
using `bench-decode.py`, 192 output tokens per request, `temperature=0`,
`enable_thinking=false`, after one warmup wave. Decode rates exclude prefill.
Hosts were otherwise idle. Raw receipts are in
[`results/decode-20261125/`](results/decode-20261125/).

### Final profile on this pair

| Workload | 1 stream | 2 streams | 3 streams |
|---|---:|---:|---:|
| Code | **49.0 tok/s** | **77.3 tok/s** | **99.0 tok/s** |
| Prose | **38.7 tok/s** | **56.4 tok/s** | **71.2 tok/s** |

- Time to first token, short prompt: ~0.3–0.5 s.
- 7,986-token retrieval prompt: **4.62 s** TTFT (~1.7K tok/s prefill).
- 399,403-token retrieval prompt: **336.5 s** TTFT (~1.19K tok/s prefill), correct.
- KV pool: **1,048,411 tokens** on the final boot (**2.62×** at 400K); 1,002,900 in the benchmark boot.

### What produced the gains

| Stage | Code C1 | Prose C1 | Code C3 | Prose C3 |
|---|---:|---:|---:|---:|
| TCP, graph cap 8, DSpark 5 | 22.4 | 16.2 | 53.1 | 36.3 |
| RoCE, graph cap 8, DSpark 5 | 53.8 | 36.7 | 66.0 | 46.1 |
| RoCE, graphs to 18, DSpark 5 | **49.0** | **38.7** | **99.0** | **71.2** |
| RoCE, graphs to 18, DSpark 3 | 45.1 | 37.4 | 98.2 | 83.1 |

1. **RoCE instead of TCP.** The original launcher set `NCCL_IB_HCA` but never
   passed `/dev/infiniband` into the containers, so NCCL silently used TCP over
   the QSFP link. A two-rank CUDA-graph all-reduce probe measured 8 KiB at
   **706 µs over TCP vs 24 µs over RoCE**; 3 MiB at **1.88 ms vs 0.34 ms**.
   More than doubling single-stream decode follows directly.
2. **Wider CUDA graphs.** One DSpark 5 verification needs six tokens per
   stream. The old cap of 8 covered only one stream. Capturing target and
   draft graphs at 1,2,4,6,8,12,18 tokens made two and three streams use full
   graph replay, adding 50–81% aggregate throughput.
3. **DSpark depth.** Five draft tokens is the best general default. Three is a
   niche choice for concurrent prose.

### Prefill

Prefill is chunked at 4,096 tokens (`--max-num-batched-tokens 4096`). The
7,986-token prompt took 4.62 s (**~1.73K tok/s**); the 399,403-token prompt
took 336.45 s (**~1.19K tok/s**), including attention over the growing
context.

## Verification

Run against the live server from this directory:

```bash
./verify.sh                 # health, model, 8K retrieval, one decode wave
python3 check-needle.py --tokens 8000
python3 check-needle.py --tokens 399500   # near the served 400K limit
python3 bench-decode.py --concurrency 1 2 3 --repeats 3
```

`check-needle.py` sizes the prompt with `/tokenize`, places
`OPAL-4917` at the midpoint, and checks the exact answer. `bench-decode.py`
computes decode from content-token timestamps and reads speculative counters
from `/metrics`; it does not count prefill as decode.

Passed on the final profile:

| Check | Result |
|---|---|
| `/health` | 200 |
| `/v1/models` | `baidu/ERNIE-5.5-Flash-Preview`, `max_model_len=400000` |
| 7,986-token retrieval | `OPAL-4917`, 4.62 s TTFT |
| 399,403-token retrieval | `OPAL-4917`, 336.45 s TTFT (~1.19K tok/s) |
| 3 concurrent requests | correct, 192 tokens each |
| LAN / Wi-Fi / Tailscale | `/health` 200 from the worker and all head interfaces |
| RoCE on both ranks | `Using network IB`, traffic via `NET/IB` |

## LAN access and security

`serve.sh` binds `--host 0.0.0.0` and the containers use `--network host`, so
the API is reachable on every head interface: LAN, Wi-Fi and Tailscale. The
worker does not serve the API; it runs `--headless`.

To bind only one interface (for example LAN only), set `HOST` to that
interface's address in `.env.local` or on the command line:

```bash
HOST=<head-lan-ip> ./start.sh
```

There is **no authentication by default**. To require a bearer token on the
OpenAI-compatible endpoints:

```bash
API_KEY='choose-a-long-random-token' ./start.sh
```

Clients then send `Authorization: Bearer <token>`, and the helper scripts take
`--api-key`. Read vLLM's warning first: `--api-key` only protects paths under
`/v1`, `/v2` and `/inference`; other endpoints, including `/invocations`,
remain unauthenticated. Do not expose this server directly to an untrusted
network. Put it behind a firewall or an authenticating reverse proxy, and
prefer binding `HOST` to the head's LAN address over binding every interface.

The head's host firewall is managed separately (`ufw` is active on this
machine). If LAN clients cannot connect while `curl 127.0.0.1:8000/health`
works, allow the port:

```bash
sudo ufw allow 8000/tcp
```

## Launch

```bash
cd ~/data/ernie55-2x-dgx-spark
cp .env.example .env.local   # fill in HEAD_IP, WORKER and WORKER_IP
./start.sh                   # worker first (headless), then head; waits for /health
```

`start.sh` copies `serve.sh` to the worker and starts one container per node.
The worker **must** run with `--headless` in multi-node MP mode; the follower
cannot run its own engine core. Startup takes about 11 minutes: ~9 minutes to
load 128+10 safetensors shards and ~2 minutes for compilation/graph capture.
JIT caches are mounted so restarts do not recompile from scratch.

Key settings in `serve.sh`:

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | 2 | One rank per Spark. Do not add `--enable-expert-parallel`; b12x requires `use_ep=False`. |
| `--moe-backend` | `b12x` | Only `sm_121` MXFP4 kernel with SiTU. |
| `--kv-cache-dtype` | `fp8` | `fp8_ds_mla`; ~1.0M-token pool. |
| `--max-model-len` | 400000 | Served window. |
| `--max-num-seqs` | 3 | Three admission slots. |
| `--max-num-batched-tokens` | 4096 | 2048 was ~2× slower prefill. |
| `--cudagraph-capture-sizes` | 1,2,4,6,8,12,18 | Full target+draft replay at C1/C2/C3. |
| `--gpu-memory-utilization` | 0.85 | Leaves headroom on the unified-memory box. |
| `--speculative-config` | DSpark 5, probabilistic | ~2× decode versus no speculation. |
| `--api-key` | optional | Only when `API_KEY` is set. |

Environment (`start.sh`):

| Env | Default | Notes |
|---|---|---|
| `WORKER` | from `.env.local` | Worker QSFP address (or SSH host); required. |
| `WORKER_IP` | from `.env.local` | Worker IP used for `VLLM_HOST_IP`; required. |
| `HEAD_IP` | from `.env.local` | Head rendezvous address; required. |
| `IMG` | `ernie55:v0.32-b12x-sink` | Must exist on both nodes. |
| `HOST` | `0.0.0.0` | API bind address. |
| `PORT` | `8000` | API port. |
| `API_KEY` | empty | Adds `--api-key` when set. |
| `RDMA` | `1` | Passes `/dev/infiniband`, raises memlock, forces NCCL `IB`. `0` restores TCP. |
| `NCCL_CHANNELS` | `4` | Faster than one channel in the probe. |
| `SPEC` | `1` | `0` disables DSpark. |
| `NUM_SPEC` | `5` | `3` selects smaller graphs automatically. |
| `GRAPH_SIZES` | auto | `1,2,4,6,8,12,18` for DSpark 5, `1,2,3,4,6,8,9,12` for 3. An explicit empty value restores the old cap of 8. |
| `MAX_LEN` / `MAX_NUMS` / `MAX_BATCHED` | 400000 / 3 / 4096 | Capacity and prefill chunk. |
| `GPU_UTIL` | `0.85` | Raise only with measured headroom. |
| `MAX_NUMS` | `3` | Graph defaults assume three slots. |

Two settings are easy to get wrong:

- `VLLM_USE_DEEP_GEMM=1` is **required**. It triggers the 3D BMM weight/scale
  post-processing that `fp8_einsum` in the HCA output projection expects. With
  it off, `wo_a` stays 2D and startup dies with a DeepGEMM layout assertion.
- The worker must be started with `--headless --node-rank 1`; otherwise vLLM
  fails with `collective_rpc should not be called on follower node`.

## Rejected experiments

| Try | Result |
|---|---|
| `NUM_SPEC=7` | 21.2 tok/s pre-RoCE, acceptance 38%; no gain, more KV |
| `draft_sample_method=greedy` | 18.6 tok/s pre-RoCE, acceptance 46%; worse |
| `enable_adaptive_verification=true` | rejected: `Ernie5_5IndexerBackend` cannot trim verification requests on device |
| b12x BF16 activations (`VLLM_B12X_MOE_FP4_FORCE_A16=1`) | 17.8 tok/s; slower than W4A8 |
| FlashInfer `sparse_indexer_topk_backend=flashinfer` | 19.2 tok/s; no gain over auto |
| `linear_backend=b12x` | fails: FP8 kernel needs output size multiple of 128, some ERNIE projections are not |
| FlashInfer CUTLASS/TRTLLM MXFP4 MoE | rejects SiTU (`does not support MoEActivation.SITU`) |
| Expert parallelism | every EP-capable MXFP4 kernel supports SiLU only; b12x forbids EP |
| `torch.compile` | disabled by the plugin's breakable-CUDA-graph architecture registration |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Engine core initialization failed`, follower errors | Worker not started with `--headless --node-rank 1`. |
| Startup dies in `fp8_einsum` with a DeepGEMM layout assertion | `VLLM_USE_DEEP_GEMM=1` is missing. |
| `does not support MoEActivation.SITU` | A non-b12x MoE backend was selected. Use `MOE_BACKEND=b12x`. |
| NCCL falls back to TCP | `/dev/infiniband` not mapped, or `NCCL_NET=IB` not set. Check both ranks log `Using network IB` and `via NET/IB`. |
| Repeated `NET/IB : GID table changed` warnings | Benign while both ranks keep progressing and log `Using network IB`; seen with IPv6 link-local churn on the second RoCE port. Restrict `NCCL_IB_HCA` to `rocep1s0f1` only if NCCL stops progressing. |
| `collective_rpc should not be called on follower node` | The worker is running a full engine; add `--headless`. |
| API unreachable from LAN but works on loopback | Firewall (`sudo ufw allow 8000/tcp`) or a non-`0.0.0.0` `HOST`. |
| Restart recompiles for minutes | JIT caches were not mounted; keep `~/.cache/ernie55-runtime` and `~/.cache/ernie55-tilelang`. |
| Long startup after image change | First boot recompiles FlashInfer/DeepGEMM/TileLang/b12x kernels; later boots reuse the caches. |

Logs:

```bash
docker logs -f ernie55
source .env.local && ssh "$WORKER" 'docker logs -f ernie55'
```

## Rollback to Hy4

ERNIE is stopped first, then the saved Hy4 containers are started. Both Hy4
containers must come up; see
[hy4-flash-2x-dgx-spark/RESTART.md](../hy4-flash-2x-dgx-spark/RESTART.md).

```bash
docker rm -f ernie55
source .env.local
ssh "$WORKER" 'docker rm -f ernie55; docker start hy4-flash-nvfp4-vllm-tp2-500k'
docker start hy4-flash-nvfp4-vllm-tp2-500k
```

Stopped ERNIE containers from the tuning runs are kept for reference:
`ernie55-tcp-baseline`, `ernie55-rdma-graph8`, `ernie55-rdma-graph18`,
`ernie55-rdma-dspark3`.

## Repo layout

| File | Purpose |
|---|---|
| `Dockerfile` | vLLM 0.32 + b12x + `vLLM-ERNIE`, plus the two patched sources. |
| `attention_patched.py` | SM120 sparse-MLA sink backend and HCA routing. |
| `situ_patched.py` | b12x SiTU W4A8 kwargs fix. |
| `attention_installed.py`, `situ_installed.py` | The upstream files the patches replace, kept for diffing. |
| `serve.sh` | In-container vLLM command; one rank per node. |
| `start.sh` | Two-node launcher, RDMA/graph defaults, health wait. |
| `stop.sh` | Stops both containers. |
| `smoke.sh` | Minimal health/model/chat probe. |
| `verify.sh` | Health, 8K retrieval and one decode wave. |
| `bench-decode.py` | Streaming decode benchmark with spec-decode counters. |
| `check-needle.py` | Token-sized long-context retrieval check. |
| `DECODE-TUNING.md` | RoCE, CUDA graph and DSpark depth experiments. |
| `COMPARISON.md` | ERNIE vs Hy4-flash on the same pair. |
| `results/decode-20261125/` | Raw benchmark and NCCL probe receipts. |

## Rebuild from scratch

The checkpoint and image must exist on **both** nodes.

```bash
# Checkpoint (155 GiB) into the shared HF cache layout.
huggingface-cli download baidu/ERNIE-5.5-Flash-Preview

# Image, on each node.
docker build -t ernie55:v0.32-b12x-sink .

# Launch.
./start.sh
```

The Dockerfile:

1. starts from `nvcr.io/nvidia/pytorch:26.07-py3`;
2. creates `/opt/ernie` and installs `vllm[b12x]==0.32.0` plus
   `git+https://github.com/baidu/vLLM-ERNIE.git`;
3. copies the two patched files over the installed sources.

The image is deliberately pinned. Rebuilding against a newer vLLM or b12x may
change the kernel oracle, the sparse-MLA backend API, or the SiTU dispatcher;
re-run `verify.sh` and the decode A/B before trusting a newer stack.

## Provenance

- vLLM 0.32.0; `vLLM-ERNIE` pinned at `ba322a5634f91d3e6f96ae711df72ddaf3c38b6e`.
- b12x 1.5.1.
- FlashInfer 0.7.0.post1 (compiled at startup; caches mounted).
- PyTorch 2.13.0a0 in NVIDIA 26.07.
- Measurements and A/B runs: 2026-11-25.

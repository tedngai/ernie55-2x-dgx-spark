#!/usr/bin/env bash
# Runs inside the container. One node = one TP rank.
set -euo pipefail
NODE_RANK="${NODE_RANK:?}"
HEAD_IP="${HEAD_IP:?}"

spec_args=()
if [[ "${SPEC:-0}" == "1" ]]; then
  adaptive=""
  if [[ "${ADAPTIVE:-0}" == "1" ]]; then adaptive=',"enable_adaptive_verification":true'; fi
  spec_args=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${NUM_SPEC:-5},\"draft_sample_method\":\"${DRAFT_METHOD:-probabilistic}\"${adaptive}}")
fi

rank_args=()
if [[ "$NODE_RANK" != "0" ]]; then
  rank_args=(--headless)
fi

kernel_args=()
kcfg=""
if [[ -n "${LIN_BACKEND:-}" ]]; then kcfg="\"linear_backend\":\"${LIN_BACKEND}\""; fi
if [[ -n "${TOPK_BACKEND:-}" ]]; then
  [[ -n "$kcfg" ]] && kcfg="$kcfg,"
  kcfg="$kcfg\"sparse_indexer_topk_backend\":\"${TOPK_BACKEND}\""
fi
if [[ -n "$kcfg" ]]; then kernel_args=(--kernel-config "{$kcfg}"); fi

graph_args=(--max-cudagraph-capture-size 8)
if [[ -n "${GRAPH_SIZES:-}" ]]; then
  IFS=, read -r -a graph_sizes <<< "$GRAPH_SIZES"
  graph_args=(--cudagraph-capture-sizes "${graph_sizes[@]}")
fi

auth_args=()
if [[ -n "${API_KEY:-}" ]]; then
  auth_args=(--api-key "$API_KEY")
fi

exec vllm serve "${MODEL:-baidu/ERNIE-5.5-Flash-Preview}" \
  --trust-remote-code \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --tensor-parallel-size 2 \
  --nnodes 2 --node-rank "$NODE_RANK" "${rank_args[@]}" \
  --master-addr "$HEAD_IP" --master-port "${MASTER_PORT:-29501}" \
  --moe-backend "${MOE_BACKEND:-b12x}" \
  "${kernel_args[@]}" \
  --kv-cache-dtype fp8 \
  --max-model-len "${MAX_LEN:-400000}" \
  --max-num-seqs "${MAX_NUMS:-3}" \
  --max-num-batched-tokens "${MAX_BATCHED:-2048}" \
  "${graph_args[@]}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --reasoning-parser ernie5 \
  --tool-call-parser ernie5 \
  --enable-auto-tool-choice \
  "${auth_args[@]}" \
  "${spec_args[@]}"

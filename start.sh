#!/usr/bin/env bash
# ERNIE-5.5-Flash-Preview FP4 on two DGX Sparks (GB10 / sm_121), TP=2.
# Worker-first headless launch; the head serves the API on :8000.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Deployment-specific addresses live in an untracked .env.local (see .env.example).
if [[ -f "$ROOT/.env.local" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
fi
: "${WORKER:?set WORKER in .env.local or the environment}"
: "${HEAD_IP:?set HEAD_IP in .env.local or the environment}"
WORKER_IP="${WORKER_IP:-$WORKER}"
IMG="${IMG:-ernie55:v0.32-b12x-sink}"
NAME="${NAME:-ernie55}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-}"
MASTER_PORT="${MASTER_PORT:-29501}"
MAX_LEN="${MAX_LEN:-400000}"
MAX_NUMS="${MAX_NUMS:-3}"
MAX_BATCHED="${MAX_BATCHED:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"
SPEC="${SPEC:-1}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
RUNTIME_CACHE="${RUNTIME_CACHE:-$HOME/.cache/ernie55-runtime}"
TILELANG_CACHE="${TILELANG_CACHE:-$HOME/.cache/ernie55-tilelang}"
NUM_SPEC="${NUM_SPEC:-5}"
ADAPTIVE="${ADAPTIVE:-0}"
DRAFT_METHOD="${DRAFT_METHOD:-probabilistic}"
LIN_BACKEND="${LIN_BACKEND:-}"
TOPK_BACKEND="${TOPK_BACKEND:-}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
RDMA="${RDMA:-1}"
NCCL_CHANNELS="${NCCL_CHANNELS:-4}"
default_graph_sizes=""
if [[ "$SPEC" == "1" && "$MAX_NUMS" == "3" ]]; then
  case "$NUM_SPEC" in
    5) default_graph_sizes=1,2,4,6,8,12,18 ;;
    3) default_graph_sizes=1,2,3,4,6,8,9,12 ;;
  esac
fi
# Set GRAPH_SIZES to an empty string to reproduce the original cap of 8.
GRAPH_SIZES="${GRAPH_SIZES-$default_graph_sizes}"

rdma_args=()
if [[ "$RDMA" == "1" ]]; then
  # Force IB (RoCE) so a missing device or broken route can't silently use TCP.
  rdma_args=(--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1
    -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0
    -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET)
fi

scp -q "$ROOT/serve.sh" "$WORKER:$ROOT/serve.sh"

common=(
  --gpus all --network host --ipc host --shm-size 32g
  -v "$HF_CACHE:/root/.cache/huggingface"
  -v "$RUNTIME_CACHE:/root/.cache"
  -v "$TILELANG_CACHE:/root/.tilelang"
  -v "$ROOT/serve.sh:/serve.sh:ro"
  -e "HEAD_IP=$HEAD_IP" -e "MASTER_PORT=$MASTER_PORT"
  -e "HOST=$HOST" -e "PORT=$PORT" -e "API_KEY=$API_KEY"
  -e "MAX_LEN=$MAX_LEN" -e "MAX_NUMS=$MAX_NUMS" -e "MAX_BATCHED=$MAX_BATCHED"
  -e "GPU_UTIL=$GPU_UTIL" -e "SPEC=$SPEC" -e "NUM_SPEC=$NUM_SPEC" -e "ADAPTIVE=$ADAPTIVE" -e "DRAFT_METHOD=$DRAFT_METHOD" -e "LIN_BACKEND=$LIN_BACKEND" -e "TOPK_BACKEND=$TOPK_BACKEND" -e "MOE_BACKEND=$MOE_BACKEND"
  -e "GRAPH_SIZES=$GRAPH_SIZES"
  -e VLLM_USE_DEEP_GEMM=1
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1
  -e "NCCL_MIN_NCHANNELS=$NCCL_CHANNELS" -e "NCCL_MAX_NCHANNELS=$NCCL_CHANNELS"
  "${rdma_args[@]}"
)

echo "worker: $WORKER ($WORKER_IP)"
printf -v worker_launch '%q ' docker run -d --name "$NAME" "${common[@]}" \
  -e NODE_RANK=1 -e "VLLM_HOST_IP=$WORKER_IP" "$IMG" bash /serve.sh
ssh "$WORKER" "docker rm -f $(printf '%q' "$NAME") >/dev/null 2>&1 || true; \
  $worker_launch >/dev/null"

echo "head:   $(hostname) ($HEAD_IP)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" "${common[@]}" -e NODE_RANK=0 -e "VLLM_HOST_IP=$HEAD_IP" "$IMG" bash /serve.sh >/dev/null

probe_host="$HOST"
if [[ "$probe_host" == "0.0.0.0" || -z "$probe_host" ]]; then probe_host=127.0.0.1; fi
echo "waiting for http://$probe_host:$PORT/health ..."
for _ in $(seq 1 180); do
  if curl -sf --max-time 2 "http://$probe_host:$PORT/health" >/dev/null 2>&1; then
    echo "ready"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "head container exited; last logs:" >&2
    docker logs --tail 80 "$NAME" >&2 || true
    exit 1
  fi
  sleep 10
done

echo "timed out; last head logs:" >&2
docker logs --tail 120 "$NAME" >&2 || true
exit 1

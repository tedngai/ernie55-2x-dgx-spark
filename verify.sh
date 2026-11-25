#!/usr/bin/env bash
# Non-destructive checks against a running server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-http://127.0.0.1:8000}"
MODEL="${MODEL:-baidu/ERNIE-5.5-Flash-Preview}"
KEY_ARGS=()
if [[ -n "${API_KEY:-}" ]]; then KEY_ARGS=(--api-key "$API_KEY"); fi

echo "== health =="
curl -s -o /dev/null -w '%{http_code}\n' "$BASE/health"
echo "== model =="
curl -sf ${API_KEY:+-H "Authorization: Bearer $API_KEY"} "$BASE/v1/models" |
  python3 -c 'import json,sys; d=json.load(sys.stdin); m=next(m for m in d["data"] if m["id"]=="'"$MODEL"'"); print(m["id"], m["max_model_len"])'
echo "== 8K retrieval =="
python3 "$ROOT/check-needle.py" --url "$BASE" --model "$MODEL" --tokens 8000 "${KEY_ARGS[@]}"
echo "== decode (one wave per workload) =="
python3 "$ROOT/bench-decode.py" --url "$BASE" --model "$MODEL" --concurrency 1 --repeats 1 "${KEY_ARGS[@]}"

#!/usr/bin/env bash
# Basic correctness + speed probe against the local server.
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8000}"

echo "== /health =="
curl -sf "$BASE/health" && echo

echo "== model list =="
curl -s "$BASE/v1/models" | python3 -m json.tool | head -20

echo "== chat =="
time curl -s "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "baidu/ERNIE-5.5-Flash-Preview",
    "messages": [{"role": "user", "content": "What is 15% of 240? Answer in one short sentence."}],
    "max_tokens": 512,
    "temperature": 0
  }' | python3 -c '
import json,sys
d=json.load(sys.stdin)
m=d["choices"][0]["message"]
print("reasoning:", (m.get("reasoning_content") or "")[:400])
print("content:  ", m.get("content"))
print("usage:    ", d.get("usage"))
'

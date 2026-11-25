#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/.env.local" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
fi
: "${WORKER:?set WORKER in .env.local or the environment}"
NAME="${NAME:-ernie55}"
ssh "$WORKER" "docker stop $NAME >/dev/null 2>&1 || true" || true
docker stop "$NAME" >/dev/null 2>&1 || true
echo "stopped $NAME on head and $WORKER"

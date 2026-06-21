#!/usr/bin/env bash
# Launch brain using its isolated venv (does not use repo-root .venv).
set -euo pipefail
BRAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$BRAIN_DIR/.." && pwd)"
PY="$BRAIN_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "brain/.venv not found. Run: ./brain/setup_venv.sh" >&2
  exit 1
fi

cd "$ROOT"
exec "$PY" -m brain.app "$@"

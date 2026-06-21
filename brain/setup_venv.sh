#!/usr/bin/env bash
# Create an isolated uv environment for brain (no anthill / ahlib install).
set -euo pipefail
BRAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BRAIN_DIR"

echo "=== brain/.venv (isolated from main anthill env) ==="

if [[ ! -d .venv ]]; then
  uv venv .venv
fi

export UV_PROJECT_ENVIRONMENT=.venv
uv sync

PY="$BRAIN_DIR/.venv/bin/python"
echo
echo "Verifying imports..."
"$PY" -c "import webview; import llama_cpp; print('  ok: pywebview + llama-cpp-python')"
echo
echo "Launch:"
echo "  ./brain/run.sh"
echo "  # or from repo root:"
echo "  brain/.venv/bin/python -m brain.app"

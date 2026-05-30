#!/usr/bin/env bash
# Linux: install triton + sageattention into .venvs/media for faster $image2video.
# Run from repo root after setup_external_venvs.sh. Requires CUDA toolkit for source builds.
set -euo pipefail

Root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$Root"

if [[ -x "$Root/.venvs/media/bin/python" ]]; then
  Py="$Root/.venvs/media/bin/python"
elif [[ -x "$Root/.venvs/media/Scripts/python.exe" ]]; then
  Py="$Root/.venvs/media/Scripts/python.exe"
else
  echo "No .venvs/media. Run: bash tools/setup_external_venvs.sh" >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT=".venvs/media"

echo "Installing triton..."
uv pip install triton --python "$Py"

echo "Installing sageattention..."
if ! uv pip install sageattention --python "$Py"; then
  echo "  pip wheel failed; trying git source (needs CUDA nvcc)..." >&2
  uv pip install "git+https://github.com/thu-ml/SageAttention.git" --python "$Py"
fi

echo ""
echo "Verify:"
"$Py" "$Root/tools/verify_sage.py"

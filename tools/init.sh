#!/usr/bin/env bash
# Fresh Anthill checkout: install Python deps, external venvs, and model weights.
# Run from repo root:
#   bash tools/init.sh
#   bash tools/init.sh --profile minimal --skip-sage
set -euo pipefail

ScriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Root="$(cd "$ScriptDir/.." && pwd)"
cd "$Root"

Profile="standard"
SkipVenvs=0
SkipModels=0
SkipSage=0
UpstreamFallback=0
DryRun=0

usage() {
  cat <<'EOF'
Usage: bash tools/init.sh [options]

Options:
  --profile minimal|standard|full   Model download profile (default: standard)
  --skip-venvs                      Skip .venvs/* setup
  --skip-models                     Skip model download
  --skip-sage                       Skip optional SageAttention install
  --upstream-fallback               Use upstream repos when anthill bundle is incomplete
  --dry-run                         Print model download plan only
  -h, --help                        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      Profile="${2:?--profile requires a value}"
      shift 2
      ;;
    --skip-venvs) SkipVenvs=1; shift ;;
    --skip-models) SkipModels=1; shift ;;
    --skip-sage) SkipSage=1; shift ;;
    --upstream-fallback) UpstreamFallback=1; shift ;;
    --dry-run) DryRun=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$Profile" in
  minimal|standard|full) ;;
  *)
    echo "Invalid --profile: $Profile (expected minimal, standard, or full)" >&2
    exit 1
    ;;
esac

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1. Install it and re-run init." >&2
    exit 1
  fi
}

echo "=== Anthill init ==="
echo "Repo: $Root"
echo ""

require_cmd uv
require_cmd hf

echo "=== 1/4 Base runtime (uv sync) ==="
uv sync

if [[ "$SkipVenvs" -eq 0 ]]; then
  echo ""
  echo "=== 2/4 External venvs (.venvs/*) ==="
  bash "$ScriptDir/setup_external_venvs.sh"
else
  echo ""
  echo "=== 2/4 External venvs: skipped (--skip-venvs) ==="
fi

env_template="$Root/.env.template"
env_file="$Root/.env"
if [[ ! -f "$env_file" && -f "$env_template" ]]; then
  cp "$env_template" "$env_file"
  echo ""
  echo "Created .env from .env.template"
fi

if [[ "$SkipSage" -eq 0 ]]; then
  echo ""
  echo "=== 3/4 Optional: SageAttention (.venvs/media) ==="
  if bash "$ScriptDir/setup_sage_linux.sh"; then
    :
  else
    echo "  SageAttention setup failed (optional)."
    echo "  Re-run later: UV_PROJECT_ENVIRONMENT=.venvs/media bash tools/setup_sage_linux.sh"
  fi
else
  echo ""
  echo "=== 3/4 SageAttention: skipped (--skip-sage) ==="
fi

if [[ "$SkipModels" -eq 0 ]]; then
  echo ""
  echo "=== 4/4 Model weights (models/) ==="
  dl_args=(tools/download_models.py --profile "$Profile")
  if [[ "$DryRun" -eq 1 ]]; then
    dl_args+=(--dry-run)
  fi
  if [[ "$UpstreamFallback" -eq 1 ]]; then
    dl_args+=(--upstream-fallback)
  fi
  uv run python "${dl_args[@]}"
else
  echo ""
  echo "=== 4/4 Models: skipped (--skip-models) ==="
fi

echo ""
echo "=== Init complete ==="
cat <<EOF

Next steps:
  1. Ensure .env has AH_EXTERNAL_VENV_* lines (printed by setup_external_venvs.sh)
  2. If models are still missing: wait for anthill upload, or bash tools/init.sh --upstream-fallback
  3. Run:  uv run python run_ah.py examples/example_simple_image_generation.ah

Model status:  uv run python tools/download_models.py --status
EOF

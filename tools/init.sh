#!/usr/bin/env bash
# Fresh Anthill checkout: install Python deps and external venvs.
# Models/test_data: bash tools/download_all_models.sh (not part of init).
# Run from repo root:
#   bash tools/init.sh
#   bash tools/init.sh --skip-sage
set -euo pipefail

ScriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Root="$(cd "$ScriptDir/.." && pwd)"
cd "$Root"

SkipVenvs=0
SkipSage=0
UploadTestData=0
DryRun=0

usage() {
  cat <<'EOF'
Usage: bash tools/init.sh [options]

Options:
  --skip-venvs                      Skip .venvs/* setup
  --skip-sage                       Skip optional SageAttention install
  --upload-test-data                Maintainer: upload test_data/ to skreling-eng/anthill
  --dry-run                         Dry-run test_data upload only
  -h, --help                        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-venvs) SkipVenvs=1; shift ;;
    --skip-sage) SkipSage=1; shift ;;
    --upload-test-data) UploadTestData=1; shift ;;
    --dry-run) DryRun=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

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

echo "=== 1/3 Base runtime (uv sync) ==="
uv sync

if [[ "$SkipVenvs" -eq 0 ]]; then
  echo ""
  echo "=== 2/3 External venvs (.venvs/*) ==="
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
  echo "=== 3/3 Optional: SageAttention (.venvs/media) ==="
  if bash "$ScriptDir/setup_sage_linux.sh"; then
    :
  else
    echo "  SageAttention setup failed (optional)."
    echo "  Re-run later: UV_PROJECT_ENVIRONMENT=.venvs/media bash tools/setup_sage_linux.sh"
  fi
else
  echo ""
  echo "=== 3/3 SageAttention: skipped (--skip-sage) ==="
fi

if [[ "$UploadTestData" -eq 1 ]]; then
  echo ""
  echo "=== Upload test_data/ -> skreling-eng/anthill ==="
  up_args=(tools/upload_to_hf.py --bundle test-data)
  if [[ "$DryRun" -eq 1 ]]; then
    up_args+=(--dry-run)
  fi
  uv run python "${up_args[@]}"
fi

echo ""
echo "=== Init complete ==="
cat <<EOF

Next steps:
  1. Ensure .env has AH_EXTERNAL_VENV_* lines (printed by setup_external_venvs.sh)
  2. Download models + test_data:  bash tools/download_all_models.sh
  3. Maintainer publish test_data:  bash tools/init.sh --upload-test-data
  4. Run:  uv run python run_ah.py examples/example_simple_image_generation.ah

Bundle status:  uv run python tools/download_models.py --status
EOF

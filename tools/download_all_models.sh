#!/usr/bin/env bash
# Download Anthill models/ + test_data/ from skreling-eng/anthill.
#   bash tools/download_all_models.sh
#   bash tools/download_all_models.sh --profile minimal --upstream-fallback
set -euo pipefail

ScriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Root="$(cd "$ScriptDir/.." && pwd)"
cd "$Root"

Profile="standard"
SkipTestData=0
UpstreamFallback=0
DryRun=0

usage() {
  cat <<'EOF'
Usage: bash tools/download_all_models.sh [options]

Options:
  --profile minimal|standard|full   Download profile (default: standard)
  --skip-test-data                  Skip test_data/** download
  --upstream-fallback               Fetch missing files from upstream HF repos
  --dry-run                         Print download plan only
  -h, --help                        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      Profile="${2:?--profile requires a value}"
      shift 2
      ;;
    --skip-test-data) SkipTestData=1; shift ;;
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
    echo "Required command not found: $1. Install it and re-run download_all_models." >&2
    exit 1
  fi
}

echo "=== Anthill model download ==="
echo "Repo:    $Root"
echo "Profile: $Profile"
echo "Source:  https://huggingface.co/skreling-eng/anthill"
echo ""

require_cmd uv
require_cmd hf

dl_args=(tools/download_models.py --profile "$Profile")
if [[ "$DryRun" -eq 1 ]]; then
  dl_args+=(--dry-run)
fi
if [[ "$UpstreamFallback" -eq 1 ]]; then
  dl_args+=(--upstream-fallback)
fi
if [[ "$SkipTestData" -eq 1 ]]; then
  dl_args+=(--skip-test-data)
fi

uv run python "${dl_args[@]}"

echo ""
echo "=== Download complete ==="
echo "Status:  uv run python tools/download_models.py --status"

# Download Wan aux files into models/wan/ for offline $image2video.
# Run from repo root. The AIO checkpoint (wan2.2-rapid-mega-aio-v12.safetensors) is separate.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

New-Item -ItemType Directory -Force -Path "models\wan\i2v-base", "models\wan\Wan2.2-T2V-A14B-Diffusers" | Out-Null

Write-Host "=== Wan2.1 I2V aux (exclude transformer shards) ==="
hf download "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers" `
  --local-dir "models/wan/i2v-base" `
  --exclude "transformer/*"

Write-Host "=== Wan2.2 T2V transformer config (JSON only) ==="
python -c @"
from huggingface_hub import snapshot_download
from pathlib import Path
root = Path('models/wan/Wan2.2-T2V-A14B-Diffusers')
root.mkdir(parents=True, exist_ok=True)
snapshot_download(
    'Wan-AI/Wan2.2-T2V-A14B-Diffusers',
    local_dir=str(root),
    allow_patterns=['transformer/*.json'],
)
print('  ok:', root)
"@

Write-Host ""
Write-Host "Add to .env:"
Write-Host "  WAN_I2V_BASE_DIR=wan/i2v-base"
Write-Host "  WAN_T2V_CONFIG_REPO=wan/Wan2.2-T2V-A14B-Diffusers"

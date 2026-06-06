# Populate models/face/ for $face and $face_enhancer.
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File tools\copy_face_models.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "=== PyTorch face-alignment weights ==="
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "fetch_face_alignment_models.ps1")

$Dest = Join-Path $Root "models\face"
$Src = Join-Path $Root "__\facelib2"
$enhancer = Join-Path $Dest "FaceEnhancer.npy"

if (-not (Test-Path $enhancer)) {
    if (-not (Test-Path $Src)) {
        Write-Host ""
        Write-Host "FaceEnhancer.npy missing and __\facelib2 not found."
        Write-Host "Get it from the anthill HF bundle:"
        Write-Host "  uv run python tools/download_models.py"
        exit 0
    }
    Copy-Item (Join-Path $Src "FaceEnhancer.npy") $enhancer -Force
    Write-Host "Copied FaceEnhancer.npy"
} else {
    Write-Host "Already present: FaceEnhancer.npy"
}

Write-Host ""
Write-Host "Face models ready in: $Dest"

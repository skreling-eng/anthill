# Fetch PyTorch face-alignment weights into models/face/ (for HF upload + offline $face).
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File tools\fetch_face_alignment_models.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root "models\face"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$files = @(
    @{
        Name = "s3fd-619a316812.pth"
        Url  = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
    },
    @{
        Name = "2DFAN4-11f355bf06.pth.tar"
        Url  = "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar"
    },
    @{
        Name = "3DFAN4-7835d9f11d.pth.tar"
        Url  = "https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar"
    },
    @{
        Name = "depth-2a464da4ea.pth.tar"
        Url  = "https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar"
    }
)

foreach ($item in $files) {
    $out = Join-Path $Dest $item.Name
    if (Test-Path $out) {
        Write-Host "Already present: $($item.Name)"
        continue
    }
    Write-Host "Downloading $($item.Name) ..."
    Invoke-WebRequest -Uri $item.Url -OutFile $out
    Write-Host "  -> $out"
}

Write-Host ""
Write-Host "Face-alignment models ready in: $Dest"
Write-Host "Upload to HF: uv run python tools/upload_to_hf.py --bundle models"
Write-Host "Also copy FaceEnhancer.npy: powershell -File tools\copy_face_models.ps1"

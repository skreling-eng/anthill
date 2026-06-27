# Create isolated uv environments for $ externals (avoids torch / ace-step conflicts).
# Run from repo root. Main interpreter can stay lean: uv sync (base only).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Sync-Venv($RelPath, $Extras) {
    $venv = Join-Path $Root $RelPath
    Write-Host ""
    Write-Host "=== $RelPath (extras: $Extras) ==="
    if (-not (Test-Path $venv)) {
        uv venv $venv
    }
    $env:UV_PROJECT_ENVIRONMENT = $RelPath
    $extraArgs = @()
    foreach ($e in ($Extras -split ",")) {
        if ($e.Trim()) { $extraArgs += @("--extra", $e.Trim()) }
    }
    uv sync @extraArgs
    $py = Join-Path $venv "Scripts\python.exe"
    uv pip install -e $Root --python $py
}

Write-Host "Anthill external venvs under .venvs/"
Sync-Venv ".venvs/media" "media,clip,music_separation,video_thumbnailer,split_video,split_video_fast,video_index"
$mediaPy = Join-Path $Root ".venvs\media\Scripts\python.exe"
Write-Host "`$openpose / `$canny / `$depth / `$controlnet (.venvs/media)"
& $mediaPy -c "import controlnet_aux; import matplotlib; import comfy_kitchen; from controlnet_aux import CannyDetector, MidasDetector; print('  ok: controlnet-aux + matplotlib + comfy-kitchen')"

# WanVideoWrapper + VHS under comfy_lib/ (heavier sklearn stack; do not merge with music)
Sync-Venv ".venvs/comfy-wan" "media,comfy-wan,clip"
$wanPy = Join-Path $Root ".venvs\comfy-wan\Scripts\python.exe"
Write-Host "`$avatar / `$image2video SkyReels colormatch (.venvs/comfy-wan)"
& $wanPy -c "import kornia; import av; print('  ok: kornia + av')"
Write-Host "SageAttention for `$avatar / WanVideoWrapper (.venvs/comfy-wan)"
powershell -NoProfile -File (Join-Path $Root "tools\setup_sage_windows.ps1") -VenvPath ".venvs/comfy-wan"

Write-Host ""
Write-Host "=== .venvs/change_voice (Python 3.10 + rvc-python; do NOT uv sync here) ==="
$cvVenv = Join-Path $Root ".venvs\change_voice"
if (-not (Test-Path $cvVenv)) {
    uv venv $cvVenv --python 3.10.16
}
$cvPy = Join-Path $cvVenv "Scripts\python.exe"
uv pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 --python $cvPy
uv pip install rvc-python --python $cvPy

Sync-Venv ".venvs/text2speech" "text2speech"
$t2sPy = Join-Path $Root ".venvs\text2speech\Scripts\python.exe"
Write-Host "spaCy en_core_web_sm for `$text2speech (misaki G2P)"
$spacyWheel = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
uv pip install $spacyWheel --python $t2sPy
& $t2sPy -c "import spacy; spacy.load('en_core_web_sm'); print('  ok: en_core_web_sm')"

Write-Host ""
Write-Host "=== .venvs/voice_enhance (Python 3.11 + resemble-enhance; do NOT uv sync here) ==="
$veVenv = Join-Path $Root ".venvs\voice_enhance"
if (-not (Test-Path $veVenv)) {
    uv venv $veVenv --python 3.11.11
}
$vePy = Join-Path $veVenv "Scripts\python.exe"
uv pip install torch==2.1.1 torchaudio==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118 --python $vePy
# resemble-enhance pins torch 2.1.1; install without deepspeed (stubbed at inference — see resemble_bootstrap.py)
uv pip install "numpy<2" resemble-enhance --no-deps --python $vePy
uv pip install soundfile huggingface-hub tqdm omegaconf scipy matplotlib pandas pyyaml --python $vePy
# tabulate>=0.9 required by pandas.to_markdown (resemble pins 0.8.10)
uv pip install librosa==0.10.1 resampy==0.4.2 celluloid==0.2.0 ptflops==0.7.1.2 rich==13.7.0 "tabulate>=0.9.0" --python $vePy
uv pip install -e $Root --python $vePy
& $vePy -c "from externals.voice_enhance.resemble_bootstrap import bootstrap_resemble_inference; bootstrap_resemble_inference(); from resemble_enhance.enhancer.inference import enhance; print('  ok: resemble-enhance')"

Sync-Venv ".venvs/music" "music"

Write-Host ""
Write-Host "`$face and `$face_enhancer use PyTorch in .venvs/media (GPU when CUDA torch is installed)."

Write-Host ""
Write-Host "=== .venvs/custom_actions (& custom actions: `$code + Pillow) ==="
Sync-Venv ".venvs/custom_actions" "custom_actions"
$caPy = Join-Path $Root ".venvs\custom_actions\Scripts\python.exe"
Write-Host "Default `$code GGUF for &action codegen"
& $caPy -c "import os; os.environ.setdefault('AH_MODEL_UPSTREAM_FALLBACK','1'); from externals.code.model_paths import ensure_model, resolve_profile_key; p=ensure_model(key=resolve_profile_key('default')); print('  ok:', p)"

Write-Host ""
Write-Host "Add to .env (or set in shell):"
Write-Host "  AH_EXTERNAL_VENV_image2text=.venvs/media
  AH_EXTERNAL_VENV_image2embedding=.venvs/media
  AH_EXTERNAL_VENV_text2embedding=.venvs/media
  AH_EXTERNAL_VENV_video2embedding=.venvs/media
  AH_EXTERNAL_VENV_split_video=.venvs/media
  AH_EXTERNAL_VENV_split_video_fast=.venvs/media
  AH_EXTERNAL_VENV_create_video_index=.venvs/media
  AH_EXTERNAL_VENV_search_local_video=.venvs/media"
Write-Host "  AH_EXTERNAL_VENV_image=.venvs/media"
Write-Host "  AH_EXTERNAL_VENV_image2image=.venvs/media"
Write-Host "  AH_EXTERNAL_VENV_image2video=.venvs/comfy-wan"
Write-Host "  AH_EXTERNAL_VENV_image2video=.venvs/media   # optional lighter venv"
Write-Host "  AH_EXTERNAL_VENV_music=.venvs/music"
Write-Host "  AH_EXTERNAL_VENV_music_separation=.venvs/media"
Write-Host "  AH_EXTERNAL_VENV_change_voice=.venvs/change_voice"
Write-Host "  AH_EXTERNAL_VENV_text2speech=.venvs/text2speech"
Write-Host "  AH_EXTERNAL_VENV_voice_enhance=.venvs/voice_enhance"
Write-Host "  AH_EXTERNAL_VENV_face=.venvs/media"
Write-Host "  AH_EXTERNAL_VENV_face_enhancer=.venvs/media"
Write-Host "  AH_EXTERNAL_VENV_openpose=.venvs/media"
Write-Host "  AH_EXTERNAL_VENV_canny=.venvs/media"
Write-Host "  AH_EXTERNAL_VENV_depth=.venvs/media"
Write-Host "  AH_EXTERNAL_VENV_controlnet=.venvs/media"
Write-Host "  AH_CUSTOM_ACTIONS_VENV=.venvs/custom_actions   # &action codegen + handlers"
Write-Host ""
Write-Host "Media venv: optional sage wheels (from repo root):"
Write-Host '  powershell -File tools\setup_sage_windows.ps1 -VenvPath ".venvs/media"'
Write-Host '  powershell -File tools\setup_sage_windows.ps1 -VenvPath ".venvs/comfy-wan"   # $avatar'
Write-Host ""
Write-Host "Run orchestrator with base deps only:"
Write-Host "  uv run python run_ah.py example.ah"

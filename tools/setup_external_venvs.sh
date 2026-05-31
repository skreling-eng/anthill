#!/usr/bin/env bash
# Create isolated uv environments for $ externals (avoids torch / ace-step conflicts).
# Run from repo root:  bash tools/setup_external_venvs.sh
set -euo pipefail

Root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$Root"

venv_python() {
  local rel="$1"
  if [[ -x "$Root/$rel/bin/python" ]]; then
    echo "$Root/$rel/bin/python"
  elif [[ -x "$Root/$rel/Scripts/python.exe" ]]; then
    echo "$Root/$rel/Scripts/python.exe"
  else
    echo "$Root/$rel/bin/python"
  fi
}

sync_venv() {
  local rel="$1"
  local extras="$2"
  echo ""
  echo "=== $rel (extras: $extras) ==="
  if [[ ! -d "$Root/$rel" ]]; then
    uv venv "$rel"
  fi
  export UV_PROJECT_ENVIRONMENT="$rel"
  local -a extra_args=()
  local IFS=,
  for e in $extras; do
    e="${e// /}"
    if [[ -n "$e" ]]; then
      extra_args+=(--extra "$e")
    fi
  done
  uv sync "${extra_args[@]}"
  local py
  py="$(venv_python "$rel")"
  uv pip install -e "$Root" --python "$py"
}

echo "Anthill external venvs under .venvs/"
sync_venv ".venvs/media" "media,clip,music_separation"
sync_venv ".venvs/comfy-wan" "media,comfy-wan,clip"

echo ""
echo "=== .venvs/change_voice (Python 3.10 + rvc-python; do NOT uv sync here) ==="
cv_venv=".venvs/change_voice"
if [[ ! -d "$Root/$cv_venv" ]]; then
  uv venv "$cv_venv" --python 3.10.16
fi
cv_py="$(venv_python "$cv_venv")"
uv pip install torch==2.1.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --python "$cv_py"
uv pip install rvc-python --python "$cv_py"

sync_venv ".venvs/text2speech" "text2speech"
t2s_py="$(venv_python ".venvs/text2speech")"
echo "spaCy en_core_web_sm for \$text2speech (misaki G2P)"
spacy_wheel="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
uv pip install "$spacy_wheel" --python "$t2s_py"
"$t2s_py" -c "import spacy; spacy.load('en_core_web_sm'); print('  ok: en_core_web_sm')"

echo ""
echo "=== .venvs/voice_enhance (Python 3.11 + resemble-enhance; do NOT uv sync here) ==="
ve_venv=".venvs/voice_enhance"
if [[ ! -d "$Root/$ve_venv" ]]; then
  uv venv "$ve_venv" --python 3.11.11
fi
ve_py="$(venv_python "$ve_venv")"
uv pip install torch==2.1.1 torchaudio==2.1.1 torchvision==0.16.1 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --python "$ve_py"
uv pip install "numpy<2" resemble-enhance --no-deps --python "$ve_py"
uv pip install soundfile huggingface-hub tqdm omegaconf scipy matplotlib pandas pyyaml --python "$ve_py"
uv pip install librosa==0.10.1 resampy==0.4.2 celluloid==0.2.0 ptflops==0.7.1.2 rich==13.7.0 "tabulate>=0.9.0" --python "$ve_py"
uv pip install -e "$Root" --python "$ve_py"
"$ve_py" -c "from externals.voice_enhance.resemble_bootstrap import bootstrap_resemble_inference; bootstrap_resemble_inference(); from resemble_enhance.enhancer.inference import enhance; print('  ok: resemble-enhance')"

sync_venv ".venvs/music" "music"

echo ""
echo "Add to .env (or export in shell):"
echo "  AH_EXTERNAL_VENV_image2text=.venvs/media"
echo "  AH_EXTERNAL_VENV_image=.venvs/media"
echo "  AH_EXTERNAL_VENV_image2image=.venvs/media"
echo "  AH_EXTERNAL_VENV_image2video=.venvs/comfy-wan"
echo "  AH_EXTERNAL_VENV_image2video=.venvs/media   # optional lighter venv"
echo "  AH_EXTERNAL_VENV_music=.venvs/music"
echo "  AH_EXTERNAL_VENV_music_separation=.venvs/media"
echo "  AH_EXTERNAL_VENV_change_voice=.venvs/change_voice"
echo "  AH_EXTERNAL_VENV_text2speech=.venvs/text2speech"
echo "  AH_EXTERNAL_VENV_voice_enhance=.venvs/voice_enhance"
echo ""
echo "Media venv: optional SageAttention (from repo root):"
echo "  UV_PROJECT_ENVIRONMENT=.venvs/media bash tools/setup_sage_linux.sh"
echo ""
echo "Run orchestrator with base deps only:"
echo "  uv run python run_ah.py example.ah"

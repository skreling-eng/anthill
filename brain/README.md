# Brain

Self-contained agentic application for analyzing the Anthill (`.ah`) codebase and
producing **unified diffs** for change requests. Diffs are proposals only — v1 does
not write changes to disk.

## Features

- **Local code model** — Qwen2.5-Coder GGUF via `llama-cpp-python` (same weights as `$code`)
- **Agent pipeline** — plan → read files → optional web search → generate diffs
- **Codebase tools** — file tree, content reading, grep, keyword search
- **Web search** — DuckDuckGo (stdlib HTTP, no anthill imports)
- **Desktop UI** — PyWebView layout:
  - **Left (large):** file tree + viewer
  - **Top right:** model output / diff report
  - **Bottom right:** change request + Analyze / Stop

## Isolated environment

Brain uses its **own** virtualenv at `brain/.venv` — separate from the repo-root
`.venv` / `.venvs/*`. Only `pywebview` and `llama-cpp-python` are installed there;
**anthill is not installed** into this env, so ahlib/externals cannot leak in at
import time.

### Setup (once)

```powershell
powershell -File brain\setup_venv.ps1
```

```bash
./brain/setup_venv.sh
```

### Run

From the `brain/` directory (recommended):

```powershell
cd brain
uv run brain
# or
uv run python run_brain.py
```

From the repo root:

```powershell
powershell -File brain\run.ps1
```

```bash
./brain/run.sh
```

### Without a local model

```powershell
$env:BRAIN_EMULATE = "1"
powershell -File brain\run.ps1
```

Emulation returns sample plans and diffs for UI testing.

## GPU (CUDA)

The default `uv sync` installs a **CPU-only** `llama-cpp-python` wheel (`gpu_offload=False`).
That is why you see **100% CPU** and no GPU usage during Planning / Generating.

**One command (Windows, RTX / CUDA GPU):**

```powershell
powershell -File brain\setup_gpu.ps1
```

This installs a prebuilt **cu124** wheel from the [llama-cpp-python wheel index](https://abetlen.github.io/llama-cpp-python/whl/cu124) (~500 MB).

Verify:

```powershell
brain\.venv\Scripts\python.exe -c "import llama_cpp.llama_cpp as lc; print(lc.llama_supports_gpu_offload())"
```

Should print `True`. On load, brain logs `gpu_offload=True` and uses `BRAIN_N_GPU_LAYERS=-1` (all layers on GPU).

**Manual install** (if the script fails):

```powershell
cd brain
$env:UV_PROJECT_ENVIRONMENT = ".venv"
uv pip install llama-cpp-python --force-reinstall --no-cache-dir `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**Build from source** (needs CUDA Toolkit + VS Build Tools):

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
uv pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Note:** Listing queries like *"give me the list of external calls"* now use a **fast catalog scan** (no LLM). Change/edit requests still use the model.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `BRAIN_CODEBASE_ROOT` | repo root | Root to analyze |
| `BRAIN_MODEL_GGUF` | `models/code/Qwen2.5-Coder-14B-Instruct/model.gguf` | Code model path |
| `BRAIN_N_CTX` | auto | Fixed context window; omit for auto power-of-two sizing |
| `BRAIN_AUTO_MAX_N_CTX` | `16384` | Auto context cap (16 GB GPU sweet spot) |
| `BRAIN_MAX_N_CTX` | `131072` | Hard upper cap when extended ctx enabled |
| `BRAIN_EXTENDED_CTX` | off | Allow YaRN context above 32k (needs VRAM) |
| `BRAIN_MAX_TOKENS` | `4096` | Max generation tokens per call |
| `BRAIN_MAX_CONTEXT_FILES` | `8` | Max source files loaded into one prompt |
| `BRAIN_EMULATE` | off | Skip LLM; use stub output |
| `BRAIN_EMULATE_SEARCH` | off | Stub web search |
| `BRAIN_N_GPU_LAYERS` | `-1` | GPU layers for llama.cpp |

Prompts are **trimmed automatically** to fit the resolved context (tree → search → grep → file contents). `.cache/` and similar paths are excluded from indexing.

## Sessions

Each analysis run writes artifacts under `brain/sessions/<timestamp>/`:

- `plan.json` — agent plan (files, grep, search queries)
- `context_index.json` — files read
- `trim_notes.json` — context trimming log (if any)
- `search.json` — web results (if any)
- `output.txt` — raw model output
- `report.md` — formatted report
- `diffs.patch` — extracted unified diffs (not applied)

Request history (queries and outputs) is stored in `brain/saves/conversation.json`. The UI shows a scrollable chat thread; follow-up messages use the last **8 turns** (default) as native ChatML history in the model. Use **New chat** to start fresh. Configure with `BRAIN_MAX_CONVERSATION_TURNS` and `BRAIN_CONVERSATION_HISTORY_CHARS`.

## Architecture

```
brain/
  app.py              # PyWebView entry
  config.py           # paths and model settings
  agent/              # orchestrator + prompts
  llm/                # local code model wrapper
  tools/              # codebase, search, diff parsing
  ui/                 # JS API + interface
  html/page.html      # UI layout
```

All Python code under `brain/` is independent of `ahlib/`, `externals/`, and `app.py`.
It reads the parent repository as analysis target only.

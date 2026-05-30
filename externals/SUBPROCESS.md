# Subprocess externals (default)

Each `$` external runs in its own subprocess. For **conflicting stacks** (torch 2.7.1 for
`media` vs ace-step’s torch 2.10 for `music`), use **separate virtualenvs** — never
`uv run --extra media --extra music` in one command (uv will fail to resolve).

## Recommended: isolated venvs (best with uv)

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_external_venvs.ps1
```

Creates:

| Path | `uv sync` extra | Used for |
|------|-----------------|----------|
| `.venvs/media` | `media` | `$image`, `$image2video`, … |
| `.venvs/music` | `music` | `$music` (`ace-step`, `model=st`) |

Point Anthill at them (in `.env` or shell):

```ini
AH_EXTERNAL_VENV_image=.venvs/media
AH_EXTERNAL_VENV_image2video=.venvs/media
AH_EXTERNAL_VENV_music=.venvs/music
```

Or explicit interpreters (same effect, skips `uv run`):

```ini
AH_EXTERNAL_PYTHON_image2video=G:\_cur\_anthill\.venvs\media\Scripts\python.exe
AH_EXTERNAL_PYTHON_music=G:\_cur\_anthill\.venvs\music\Scripts\python.exe
```

**Orchestrator** (lightweight — only `llama-cpp-python`, no torch):

```powershell
uv sync
uv run python run_ah.py example.ah
```

Subprocesses pick the right venv automatically when `.venvs/*` exists and env vars are set.

### Manual uv pattern (`UV_PROJECT_ENVIRONMENT`)

Same idea without the script:

```powershell
uv venv .venvs\media
$env:UV_PROJECT_ENVIRONMENT = ".venvs/media"
uv sync --extra media
uv pip install -e .

uv venv .venvs\music
$env:UV_PROJECT_ENVIRONMENT = ".venvs/music"
uv sync --extra music
uv pip install -e .
```

`UV_PROJECT_ENVIRONMENT` tells uv which folder is the project venv (relative to repo root).

## Fallback: single `.venv` + `uv run --extra`

If isolated venvs are missing, subprocess uses:

```text
uv run --extra media python -m externals.runner …
```

That installs into the **default** `.venv`. Mixing `media` + `music` in one sync can conflict;
prefer `.venvs/*` for production runs.

## IPC files (each op folder)

| File | Role |
|------|------|
| `input.json` | Input bundle |
| `invoke.json` | `$` args, `prompt_text`, `repeat`, `arg_lists` |
| `output.json` | Result bundle |

Child loads `.env` (`ACESTEP_BACKEND=native`, etc.) via `externals/bootstrap.py`.

## Overrides

| Variable | Effect |
|----------|--------|
| `AH_EXTERNAL_INPROCESS=all` | No subprocess |
| `AH_EXTERNAL_UV=0` | Subprocess uses `sys.executable`, not `uv run` |
| `AH_EXTERNAL_PYTHON_<name>` | Force interpreter path |
| `AH_EXTERNAL_VENV_<name>` | Force venv directory (e.g. `.venvs/media`) |
| `AH_UV_EXTRA_<name>` | Change `uv run --extra` list |
| `AH_EXTERNAL_TIMEOUT` | Subprocess timeout (seconds) |
| `AH_RELEASE_GPU_ON_RUN_END=0` | Keep warm workers (e.g. `$image2image`) after a run finishes |

When a `.ah` run finishes (success, error, or cancel), Anthill calls `release_gpu_resources()`:
stops the `$image2image` warm worker and any still-running external subprocesses, then
`torch.cuda.empty_cache()` in the orchestrator if torch is loaded. Default: on (`1`).

## Manual debug

```powershell
$env:AH_SESSION_BASE_DIR = "G:\_cur\_anthill\sessions\<session>"
.\.venvs\music\Scripts\python.exe -m externals.runner music "$env:AH_SESSION_BASE_DIR\11__music"
```

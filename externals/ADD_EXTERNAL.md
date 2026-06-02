# Adding a new model external

This guide walks through adding a new `$` external that loads weights from `models/`
and can be fetched via `tools/download_models.py` or on first use.

Use **`$pass`** as the smallest handler skeleton and **`$image2text`** / **`$voice_enhance`**
as reference implementations for model paths and downloads.

---

## Checklist

| Step | Where |
|------|--------|
| Create handler folder | `externals/<name>/` |
| Implement `run(ctx, inp) -> ArrayBundle` | `externals/<name>/run.py` |
| Register the external name | `externals/__init__.py` → `_KNOWN` |
| Optional: prompt / repeat behaviour | `_PROMPT_CONSUMING`, `_REPEAT_NATIVE` in same file |
| Named `$name(model='…')` profiles | `externals/<name>/model_list.py` |
| Resolve + download weights | `externals/<name>/model_paths.py` |
| Spot-check files for bulk download | `externals/anthill_models.py` → `CHECKS` + `PROFILE_GROUPS` |
| Optional upstream fallback | `tools/download_models.py` → `_run_upstream_fallback` |
| User-facing help | `externals/<name>/_description` |
| Heavy deps (torch, etc.) | `tools/setup_external_venvs.ps1` / `.sh`, see [`SUBPROCESS.md`](SUBPROCESS.md) |

---

## 1. Folder and handler interface

### Layout

```
externals/my_model/
  __init__.py       # re-export run
  run.py            # entry point (required)
  model_list.py     # optional — named model= profiles
  model_paths.py    # optional — paths + ensure_model()
  _description      # optional — syntax / examples for users
```

Minimal package stub:

```python
# externals/my_model/__init__.py
import importlib

run = importlib.import_module("externals.my_model.run").run
__all__ = ["run"]
```

### Handler signature

Every external implements one function:

```python
# externals/my_model/run.py
from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    # read inputs, write outputs …
    return out
```

### Input context (`ExternalInput`)

| Field | Meaning |
|-------|---------|
| `bundle` | Current data: `prompts`, `texts`, `images`, `sounds`, `videos`, `files`, `changes` (each is a list of session-relative links) |
| `args` | `$my_model(key=value, …)` arguments (all strings) |
| `prompt_text` | Combined prompt text when `prompts[]` is empty |
| `repeat` | From `$my_model(...)[n]` — only used if you handle variants internally |
| `arg_lists` | Expanded `@ref` args (Cartesian fan-out is done by the runtime) |

Helpers in `externals/api.py`:

- `read_prompt_texts(ctx, inp)` — prompts from bundle or `prompt_text`
- `read_bundle_texts(ctx, inp)` — all `texts[]` contents
- `read_arg_list(inp, "model")` — scalar or list arg (for `@ref` expansion)

### Output context (`ExternalContext`)

| Member | Use |
|--------|-----|
| `ctx.base_dir` | Session root (resolve relative links) |
| `ctx.op_dir` | This operation folder (scratch files, logs) |
| `ctx.read_link_text(link)` / `read_link_bytes(link)` | Read input link contents |
| `ctx.new_link(array, ext, content)` | Create a new session file; returns a link string for `out.<array>.append(...)` |
| `ctx.cancel_event` | Check for user cancel (long jobs) |

Typical pattern: copy the bundle, clear arrays you replace, append new links:

```python
out = inp.bundle.copy()
out.texts.clear()
link = ctx.new_link("texts", ".txt", "result\n")
out.texts.append(link)
return out
```

### Register the name

Add `"my_model"` to `_KNOWN` in `externals/__init__.py`.

If the external **reads prompts as model input** (like `$llm`, `$image`), also add it to
`_PROMPT_CONSUMING` so the runtime clears `prompts[]` on output.

If it handles **`$my_model(...)[n]` internally** (one call, `n` variants inside the handler),
add it to `_REPEAT_NATIVE`. Otherwise the runtime runs the external `n` times and joins results.

### Subprocess vs in-process

By default each `$` call runs in a **subprocess** (`python -m externals.runner <name> <op_dir>`).
The parent writes `input.json` + `invoke.json`; the child writes `output.json`.
See [`SUBPROCESS.md`](SUBPROCESS.md) for venv setup (`AH_EXTERNAL_VENV_<name>`, `AH_UV_EXTRA_<name>`).

For tests without heavy deps, support `AH_EMULATE_<NAME>=1` and return stub output (see `$llm`, `$ocr`).

---

## 2. Adding a model into `models/`

Anthill resolves files under **`models/`** (repo root) and optional extra roots from
`MODELS_PATH` (see `models_roots()` in `externals/image/model_paths.py`).

### Named profiles (`model=` argument)

Define profiles in `model_list.py` and resolve them in `run.py`:

```python
# externals/my_model/model_list.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class MyModel:
    name: str
    subdir: Path          # under models/, e.g. Path("my_model/v1")

_profiles = [
    MyModel("default", Path("my_model/v1")),
    MyModel("large", Path("my_model/v2")),
]
_by_name = {p.name: p for p in _profiles}

def get_model(name: str) -> MyModel:
    key = (name or "default").strip()
    if key not in _by_name:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(sorted(_by_name))}")
    return _by_name[key]
```

In `run.py`:

```python
model_name = inp.args.get("model", "default")
from externals.my_model.model_list import get_model
from externals.my_model.model_paths import ensure_model

profile = get_model(model_name)
weights_dir = ensure_model(profile)
```

Upload weights to the [anthill Hugging Face bundle](https://huggingface.co/skreling-eng/anthill)
under the same relative paths (e.g. `models/my_model/v1/…`) so bulk download and on-demand fetch
stay in sync.

### Path helpers (`model_paths.py`)

Keep all filesystem logic here:

```python
# externals/my_model/model_paths.py
from pathlib import Path
from externals.anthill_models import ensure_anthill_tree, upstream_fallback_enabled
from externals.image.model_paths import models_roots
from externals.my_model.model_list import MyModel, get_model

def model_dir(profile: MyModel | str | None = None) -> Path:
    m = profile if isinstance(profile, MyModel) else get_model(profile or "default")
    rel = m.subdir
    for root in models_roots():
        candidate = root / rel
        if (candidate / "config.json").is_file():   # pick your marker file(s)
            return candidate
    return models_roots()[0] / rel

def model_ready(profile: MyModel | str | None = None) -> bool:
    path = model_dir(profile)
    return (path / "config.json").is_file() and (path / "weights.bin").is_file()

def ensure_model(profile: MyModel | str | None = None, *, force: bool = False) -> Path:
    m = profile if isinstance(profile, MyModel) else get_model(profile or "default")
    if model_ready(m) and not force:
        return model_dir(m)

    ensure_anthill_tree(
        m.subdir.as_posix(),
        ready=lambda: model_ready(m),
        label="$my_model",
        force=force,
    )
    if model_ready(m):
        return model_dir(m)

    if upstream_fallback_enabled():
        # optional: snapshot_download from upstream HF repo
        ...

    raise FileNotFoundError(
        f"Model not ready under {model_dir(m)}. "
        "Run: uv run python tools/download_models.py"
    )
```

Use **`ensure_anthill_file`** / **`ensure_anthill_files`** when you need individual files
instead of a whole directory tree (see `externals/ocr/model_paths.py`).

Call **`ensure_model(...)` once per run** (or lazily on first inference), not on every item in a batch.

---

## 3. Bulk download (`download all models`)

Registration lives in **`externals/anthill_models.py`**.

### Add a CHECKS group

`CHECKS` maps a **group key** → list of **spot-check paths** (posix, relative to `models/`).
If every path exists locally, the group is considered ready.

```python
# externals/anthill_models.py
CHECKS: dict[str, list[str]] = {
    ...
    "my_model_v1": [
        "my_model/v1/config.json",
        "my_model/v1/weights.bin",
    ],
}
```

Pick paths that reliably indicate a complete install. For sharded models, list one shard plus
`config.json` (see existing `qwen2_vl`, `ocr_en` entries).

### Add to a download profile

```python
PROFILE_GROUPS: dict[str, frozenset[str]] = {
    "minimal": frozenset({..., "my_model_v1"}),
    "standard": frozenset(CHECKS.keys()) - {...},
    "full": frozenset(CHECKS.keys()),
}
```

`tools/download_models.py` uses these profiles:

```bash
uv run python tools/download_models.py              # standard profile
uv run python tools/download_models.py --profile minimal
uv run python tools/download_models.py --status     # readiness table
```

Behaviour:

1. For each missing group, fetch individual missing files via `ensure_anthill_file`.
2. Then sync the common directory prefix via `sync_anthill_tree` (derived by `group_tree_prefix`).

### Optional: upstream fallback in the download script

If anthill does not mirror every file yet, hook your `ensure_model` into
`_run_upstream_fallback` in `tools/download_models.py` (pattern used for OCR, voice enhance, code, etc.):

```python
if "my_model_v1" in missing:
    from externals.my_model.model_paths import ensure_model
    ensure_model("default")
```

Users run: `uv run python tools/download_models.py --upstream-fallback`

---

## 4. First-run download (on demand)

When a user runs `$my_model` without pre-downloading, call your `ensure_model()` from
`run.py` before loading weights. That delegates to **`externals/anthill_models`**:

| Function | When to use |
|----------|-------------|
| `ensure_anthill_file(rel)` | Single file |
| `ensure_anthill_files([rel, …])` | Several known files |
| `ensure_anthill_tree(rel_dir, ready=…)` | Whole subtree (snapshot `allow_patterns`) |
| `require_models_file(rel)` | Resolve locally or download one file |

Auto-download is **on by default** (`AH_ANTHILL_AUTO_DOWNLOAD=1`).
Set `AH_ANTHILL_AUTO_DOWNLOAD=0` to fail fast with a message pointing at
`uv run python tools/download_models.py`.

Optional upstream Hugging Face repos when anthill is incomplete:

- Set `AH_MODEL_UPSTREAM_FALLBACK=1`, or
- Implement fallback inside `ensure_model()` when `upstream_fallback_enabled()` is true
  (see `externals/image2text/model_paths.py`, `externals/voice_enhance/model_paths.py`).

Example in `run.py`:

```python
def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    if os.environ.get("AH_EMULATE_MY_MODEL", "").lower() in ("1", "true", "yes"):
        return _emulate(ctx, inp)

    from externals.my_model.model_paths import ensure_model
    from externals.my_model.model_list import get_model

    profile = get_model(inp.args.get("model", "default"))
    weights = ensure_model(profile)   # downloads on first use if missing
    ...
```

The anthill module deduplicates concurrent downloads and prints a line like:

```text
$my_model: downloading models/my_model/v1/** from skreling-eng/anthill
```

Requires **`huggingface-hub`** (`uv sync --extra media` or the relevant project extra).

---

## End-to-end example (`.ah`)

After implementing `$my_model`:

```ah
@caption: $file('photo.png') -> $my_model(model='default') -> $save('out.txt')
```

With emulation for CI / quick tests:

```powershell
$env:AH_EMULATE_MY_MODEL = "1"
uv run python run_ah.py examples/example_my_model.ah
```

---

## Related docs

- [`SUBPROCESS.md`](SUBPROCESS.md) — isolated venvs, IPC files, env overrides
- [`../_lang_desc`](../_lang_desc) — `$name(...)`, repeat `[n]`, parallel `( … )`
- Per-external user docs: `externals/<name>/_description`
- Model bundle: https://huggingface.co/skreling-eng/anthill

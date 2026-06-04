"""$code model registry — one dict entry per profile; add models here only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from externals.anthill_models import (
    require_models_file,
    resolve_models_file,
    upstream_fallback_enabled,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _REPO_ROOT / "models"


@dataclass(frozen=True)
class CodeModelProfile:
    """Full $code(model=…) profile: paths, aliases, optional HF instruct for QLoRA."""

    key: str
    subdir: str
    hf_gguf_repo: str
    hf_gguf_file: str
    aliases: tuple[str, ...]
    gguf_name: str = "model.gguf"
    anthill_gguf: str | None = None
    hf_instruct_repo: str | None = None
    hf_instruct_subdir: str | None = None
    chat_format: str = "chatml"
    is_default: bool = False
    allow_upstream_download: bool = True

    @property
    def anthill_bundle_path(self) -> str:
        if self.anthill_gguf:
            return self.anthill_gguf
        return f"code/{self.subdir}/{self.gguf_name}"

    @property
    def model_dir(self) -> Path:
        return MODELS_DIR / "code" / self.subdir

    @property
    def model_gguf(self) -> Path:
        return self.model_dir / self.gguf_name

    @property
    def hf_instruct_dir(self) -> Path | None:
        if not self.hf_instruct_repo:
            return None
        name = self.hf_instruct_subdir or f"{self.subdir}-HF"
        return MODELS_DIR / "code" / name


# ── Add new $code models here ──────────────────────────────────────────────
CODE_MODELS: dict[str, CodeModelProfile] = {
    "14b": CodeModelProfile(
        key="14b",
        subdir="Qwen2.5-Coder-14B-Instruct",
        hf_gguf_repo="bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
        hf_gguf_file="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        aliases=(
            "default",
            "14b",
            "Qwen2.5-Coder-14B-Instruct",
            "qwen2.5-coder-14b-instruct",
        ),
        is_default=True,
    ),
    "1.5b": CodeModelProfile(
        key="1.5b",
        subdir="Qwen2.5-Coder-1.5B-Instruct",
        hf_gguf_repo="bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        hf_gguf_file="Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
        aliases=(
            "1.5b",
            "15b",
            "1_5b",
            "Qwen2.5-Coder-1.5B-Instruct",
            "qwen2.5-coder-1.5b-instruct",
        ),
        hf_instruct_repo="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        hf_instruct_subdir="Qwen2.5-Coder-1.5B-Instruct-HF",
    ),
    "1.5b_ah_lora": CodeModelProfile(
        key="1.5b_ah_lora",
        subdir="Qwen2.5-Coder-1.5B-Instruct",
        gguf_name="model_lora.gguf",
        anthill_gguf="code/Qwen2.5-Coder-1.5B-Instruct/model_ah_lora.gguf",
        hf_gguf_repo="",
        hf_gguf_file="",
        aliases=("1.5b_ah_lora",),
        allow_upstream_download=False,
    ),
}


def _build_alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for profile in CODE_MODELS.values():
        for label in (profile.key, *profile.aliases):
            index[label.strip().lower()] = profile.key
            index[label.strip().lower().replace("-", "")] = profile.key
    return index


_ALIAS_TO_KEY = _build_alias_index()


def default_profile() -> CodeModelProfile:
    for profile in CODE_MODELS.values():
        if profile.is_default:
            return profile
    return next(iter(CODE_MODELS.values()))


def resolve_profile_key(name: str) -> str:
    """Map $code(model=…) name or alias to registry key."""
    raw = name.strip()
    for candidate in (raw.lower(), raw.lower().replace("-", "")):
        if candidate in _ALIAS_TO_KEY:
            return _ALIAS_TO_KEY[candidate]
    available = ", ".join(sorted(_list_model_names()))
    raise KeyError(f"Unknown $code model {name!r}. Available: {available}")


def get_code_profile(name: str) -> CodeModelProfile:
    return CODE_MODELS[resolve_profile_key(name)]


def code_model_paths(key: str = "14b") -> CodeModelProfile:
    """Back-compat alias for get_code_profile."""
    return get_code_profile(key)


def _list_model_names() -> list[str]:
    names: set[str] = set()
    for profile in CODE_MODELS.values():
        names.add(profile.key)
        names.update(profile.aliases)
    return sorted(names)


# Back-compat module-level defaults (14B)
_DEFAULT = default_profile()
MODEL_DIR = _DEFAULT.model_dir
MODEL_GGUF = _DEFAULT.model_gguf
ANTHILL_GGUF = _DEFAULT.anthill_bundle_path
HF_REPO = _DEFAULT.hf_gguf_repo
HF_GGUF = _DEFAULT.hf_gguf_file
HF_INSTRUCT_1_5B = CODE_MODELS["1.5b"].hf_instruct_repo or ""


def model_ready(key: str = "14b") -> bool:
    return get_code_profile(key).model_gguf.is_file()


def ensure_model(*, key: str = "14b", force: bool = False) -> Path:
    """Resolve Qwen2.5-Coder GGUF (anthill bundle or bartowski upstream)."""
    profile = get_code_profile(key)
    profile.model_dir.mkdir(parents=True, exist_ok=True)
    dest = profile.model_gguf
    if dest.is_file() and not force:
        return dest

    found = resolve_models_file(profile.anthill_bundle_path)
    if found is not None:
        if found.resolve() != dest.resolve():
            import shutil

            shutil.copy2(found, dest)
        return dest

    try:
        bundle = require_models_file(profile.anthill_bundle_path, label=f"$code({profile.key})")
        if bundle.resolve() != dest.resolve():
            import shutil

            shutil.copy2(bundle, dest)
        return dest
    except Exception:
        pass

    if profile.allow_upstream_download and upstream_fallback_enabled():
        if not profile.hf_gguf_repo or not profile.hf_gguf_file:
            raise FileNotFoundError(
                f"Model not found: {dest}. No upstream download configured for {profile.key!r}."
            )
        import shutil

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "$code needs huggingface-hub to download the model: "
                "uv pip install huggingface-hub"
            ) from exc

        print(
            f"$code: downloading {profile.hf_gguf_file} from {profile.hf_gguf_repo}",
            flush=True,
        )
        downloaded = Path(
            hf_hub_download(
                profile.hf_gguf_repo,
                profile.hf_gguf_file,
                local_dir=str(profile.model_dir),
            )
        )
        if downloaded.resolve() != dest.resolve():
            shutil.copy2(downloaded, dest)

    if not dest.is_file():
        hint = (
            "Run: uv run python tools/download_code_model.py --model 1.5b"
            if profile.key == "1.5b"
            else f"Place GGUF at {dest} or run $model_ah_merge_lora for LoRA profiles."
        )
        raise FileNotFoundError(f"Model not found: {dest}. {hint}")
    return dest


def ensure_instruct_hf(*, key: str = "1.5b", force: bool = False) -> Path:
    """Download full HF instruct weights (for QLoRA / transformers), not GGUF."""
    profile = get_code_profile(key)
    if not profile.hf_instruct_repo or profile.hf_instruct_dir is None:
        raise ValueError(f"HF instruct weights not configured for {profile.key!r}")
    dest = profile.hf_instruct_dir
    if dest.is_dir() and any(dest.iterdir()) and not force:
        return dest
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("uv pip install huggingface-hub") from exc
    print(f"Downloading {profile.hf_instruct_repo} -> {dest}", flush=True)
    snapshot_download(profile.hf_instruct_repo, local_dir=str(dest))
    return dest

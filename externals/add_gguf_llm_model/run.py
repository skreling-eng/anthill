"""$add_gguf_llm_model — fetch a GGUF into models/llm_user/ for $llm(model=…)."""

from __future__ import annotations

import os

from externals.api import ExternalContext, ExternalInput
from externals.llm.user_models import ensure_user_gguf, sanitize_model_name
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$add_gguf_llm_model(name='my_model', gguf='https://…/file.gguf')

Downloads into models/llm_user/<name>/<file>.gguf if not already present.
Then use: $llm(model='my_model')

Passes the input bundle through unchanged.
Set AH_EMULATE_ADD_GGUF_LLM_MODEL=1 to skip real download.
"""


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_ADD_GGUF_LLM_MODEL", "").lower() in (
        "1",
        "true",
        "yes",
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    name_raw = inp.args.get("name", "").strip()
    gguf_link = inp.args.get("gguf", "").strip()
    if not name_raw or not gguf_link:
        raise RuntimeError(_HELP.strip())

    safe_name = sanitize_model_name(name_raw)
    ensure_user_gguf(
        safe_name,
        gguf_link,
        emulate=_emulate_enabled(),
    )
    return inp.bundle.copy()

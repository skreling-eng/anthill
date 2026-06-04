"""$model_ah_merge_lora — merge LoRA adapter and export GGUF to files[]."""

from __future__ import annotations

import os
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.model_ah.merge import merge_lora_to_gguf
from externals.model_ah.paths import default_hf_model_dir, resolve_adapter_dir
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$model_ah_merge_lora — merge LoRA from files[] into base HF weights and write GGUF.

Input files[]: LoRA .zip from $model_ah_train_lora (or adapter_config.json path).

Example:
  @gguf: @lora -> $model_ah_merge_lora(model='1.5b', quant='Q4_K_M')

GGUF conversion auto-downloads llama.cpp tools on first use (or set AH_LLAMA_CPP).
Without llama-quantize, K-quants fall back to q8_0. Set AH_EMULATE_MODEL_AH_MERGE_LORA=1 for tests.
"""


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_MODEL_AH_MERGE_LORA", "").lower() in (
        "1",
        "true",
        "yes",
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    model_key = inp.args.get("model", "1.5b").strip() or "1.5b"
    quant = inp.args.get("quant", "Q4_K_M").strip() or "Q4_K_M"
    out_name = inp.args.get("output_name", "model").strip() or "model"
    work_dir = ctx.op_dir / "merge_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_gguf = ctx.op_dir / "files" / f"{out_name}.gguf"

    if _emulate_enabled():
        out_gguf.parent.mkdir(parents=True, exist_ok=True)
        out_gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    else:
        adapter_dir = resolve_adapter_dir(ctx, inp, work_dir=work_dir)
        model_dir = default_hf_model_dir(model_key)
        merge_lora_to_gguf(
            model_dir=model_dir,
            adapter_dir=adapter_dir,
            work_dir=work_dir,
            out_gguf=out_gguf,
            quant=quant,
        )

    rel = str(out_gguf.relative_to(ctx.base_dir)).replace("\\", "/")
    out = inp.bundle.copy()
    out.files.clear()
    out.files.append(rel)
    return out

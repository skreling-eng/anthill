"""$model_ah_train_lora — QLoRA fine-tune on JSONL from files[]."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.model_ah.paths import default_hf_model_dir, read_jsonl_path
from externals.model_ah.train import train_lora
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$model_ah_train_lora — QLoRA fine-tune Qwen2.5-Coder-1.5B on JSONL in files[].

Setup: uv sync --extra finetune
Base HF weights: uv run python tools/download_code_model.py --hf-instruct --hf-only --model 1.5b

Example:
  @jsonl: @scripts -> $model_ah_create_jsonl
  @lora: @jsonl -> $model_ah_train_lora(model='1.5b', epochs=2)
  @gguf: @lora -> $model_ah_merge_lora

Args: model=1.5b, epochs=, batch_size=, lr=, max_seq_len=, output_name=
Set AH_EMULATE_MODEL_AH_TRAIN_LORA=1 for a stub adapter (tests).
"""


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_MODEL_AH_TRAIN_LORA", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    return int(raw)


def _float_arg(args: dict[str, str], key: str, default: float) -> float:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    return float(raw)


def _write_emulate_adapter(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "emulated": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "adapter_model.safetensors").write_bytes(b"")


def _zip_adapter(adapter_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(adapter_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(adapter_dir).as_posix())


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    dataset_path = read_jsonl_path(ctx, inp)
    model_key = inp.args.get("model", "1.5b").strip() or "1.5b"
    out_name = inp.args.get("output_name", "lora_adapter").strip() or "lora_adapter"
    lora_dir = ctx.op_dir / "lora_out"
    zip_path = ctx.op_dir / "files" / f"{out_name}.zip"

    if _emulate_enabled():
        _write_emulate_adapter(lora_dir)
    else:
        model_dir = default_hf_model_dir(model_key)
        train_lora(
            model_dir=model_dir,
            dataset_path=dataset_path,
            output_dir=lora_dir,
            epochs=_float_arg(inp.args, "epochs", 2.0),
            batch_size=_int_arg(inp.args, "batch_size", 1),
            grad_accum=_int_arg(inp.args, "grad_accum", 8),
            lr=_float_arg(inp.args, "lr", 2e-4),
            max_seq_len=_int_arg(inp.args, "max_seq_len", 2048),
            lora_r=_int_arg(inp.args, "lora_r", 16),
            lora_alpha=_int_arg(inp.args, "lora_alpha", 32),
            save_steps=_int_arg(inp.args, "save_steps", 100),
            qlora=inp.args.get("qlora", "true").strip().lower()
            not in ("0", "false", "no", "off"),
        )

    _zip_adapter(lora_dir, zip_path)
    rel = str(zip_path.relative_to(ctx.base_dir)).replace("\\", "/")

    out = inp.bundle.copy()
    out.files.clear()
    out.files.append(rel)
    return out

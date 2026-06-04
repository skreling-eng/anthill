"""Merge LoRA into base HF weights and convert to GGUF."""

from __future__ import annotations

from pathlib import Path

from externals.model_ah.llama_cpp_tools import run_hf_to_gguf


def require_finetune_deps() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing merge deps. Install with:\n  uv sync --extra finetune"
        ) from exc


def merge_lora_to_hf(
    *,
    model_dir: Path,
    adapter_dir: Path,
    out_dir: Path,
) -> Path:
    """Merge LoRA adapter into base model; save full HF weights."""
    require_finetune_deps()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_dir = model_dir.resolve()
    adapter_dir = adapter_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (adapter_dir / "adapter_config.json").is_file():
        raise RuntimeError(f"LoRA adapter not found: {adapter_dir}")

    print(f"Merging {adapter_dir} into {base_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    print(f"Merged HF model saved: {out_dir}", flush=True)
    return out_dir


def hf_to_gguf(
    *,
    hf_dir: Path,
    out_gguf: Path,
    quant: str = "Q4_K_M",
) -> Path:
    """Convert merged HF weights to GGUF (auto-download llama.cpp tools if needed)."""
    return run_hf_to_gguf(hf_dir=hf_dir, out_gguf=out_gguf, quant=quant)


def merge_lora_to_gguf(
    *,
    model_dir: Path,
    adapter_dir: Path,
    work_dir: Path,
    out_gguf: Path,
    quant: str = "Q4_K_M",
) -> Path:
    """Merge LoRA, convert merged HF to GGUF."""
    merged_hf = work_dir / "merged_hf"
    merge_lora_to_hf(model_dir=model_dir, adapter_dir=adapter_dir, out_dir=merged_hf)
    return hf_to_gguf(hf_dir=merged_hf, out_gguf=out_gguf, quant=quant)

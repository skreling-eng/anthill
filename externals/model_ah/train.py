"""QLoRA fine-tune Qwen2.5-Coder on Anthill JSONL."""

from __future__ import annotations

from pathlib import Path

from externals.model_ah.dataset import load_jsonl


def require_finetune_deps() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import trl  # noqa: F401
        import datasets  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing training deps. Install with:\n  uv sync --extra finetune"
        ) from exc


def train_lora(
    *,
    model_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    epochs: float = 2.0,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-4,
    max_seq_len: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    save_steps: int = 100,
    qlora: bool = True,
) -> Path:
    """Train LoRA adapter; returns output_dir."""
    require_finetune_deps()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise RuntimeError(
            f"Model not found: {model_dir}\n"
            "Run: uv run python tools/download_code_model.py --hf-instruct --hf-only --model 1.5b"
        )

    rows = load_jsonl(dataset_path.resolve())
    if not rows:
        raise RuntimeError(f"Empty dataset: {dataset_path}")

    print(f"Loading tokenizer from {model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model from {model_dir} (qlora={qlora})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not qlora else None,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = Dataset.from_list(rows)

    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    out_dir = output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        max_length=max_seq_len,
        packing=False,
        dataset_text_field=None,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    print(f"Training on {len(ds)} examples -> {out_dir}", flush=True)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(out_dir)
    print(f"LoRA adapter saved: {out_dir}", flush=True)
    return out_dir

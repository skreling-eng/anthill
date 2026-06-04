#!/usr/bin/env python3
"""
QLoRA fine-tune Qwen2.5-Coder-1.5B-Instruct on Anthill .ah examples.

Setup (isolated venv recommended — torch + training deps):
  uv sync --extra finetune

Download base HF weights (not GGUF):
  uv run python tools/download_code_model.py --hf-instruct --hf-only --model 1.5b

Export dataset only:
  uv run python tools/finetune_coder_15b.py export \\
    --out data/anthill_code_train.jsonl

Train LoRA:
  uv run python tools/finetune_coder_15b.py train \\
    --dataset data/anthill_code_train.jsonl \\
    --output-dir models/code/lora-anthill-1.5b

Merge LoRA into base (optional, for HF export / later GGUF conversion):
  uv run python tools/finetune_coder_15b.py merge \\
    --adapter models/code/lora-anthill-1.5b \\
    --out models/code/Qwen2.5-Coder-1.5B-Instruct-Anthill-HF
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = REPO_ROOT / "models" / "code" / "Qwen2.5-Coder-1.5B-Instruct-HF"
DEFAULT_OUT = REPO_ROOT / "data" / "anthill_code_train.jsonl"
DEFAULT_LORA = REPO_ROOT / "models" / "code" / "lora-anthill-1.5b"


def cmd_export(args: argparse.Namespace) -> None:
    from externals.model_ah.dataset import build_dataset, write_jsonl

    folders = [Path(p).resolve() for p in args.examples]
    rows = build_dataset(folders, per_usecase=args.per_usecase)
    if not rows:
        raise SystemExit("No training rows (check --examples paths and .ah files).")
    out = Path(args.out).resolve()
    write_jsonl(rows, out)
    print(f"Wrote {len(rows)} examples -> {out}")


def cmd_train(args: argparse.Namespace) -> None:
    from externals.model_ah.train import train_lora

    train_lora(
        model_dir=Path(args.model_dir).resolve(),
        dataset_path=Path(args.dataset).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        save_steps=args.save_steps,
        qlora=args.qlora,
    )


def cmd_merge(args: argparse.Namespace) -> None:
    from externals.model_ah.merge import merge_lora_to_hf

    out_dir = merge_lora_to_hf(
        model_dir=Path(args.model_dir).resolve(),
        adapter_dir=Path(args.adapter).resolve(),
        out_dir=Path(args.out).resolve(),
    )
    print(f"Merged model saved: {out_dir}")
    print(
        "Convert to GGUF: $model_ah_merge_lora or llama.cpp convert_hf_to_gguf + quantize"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Build JSONL from .ah examples")
    p_export.add_argument(
        "--examples",
        nargs="+",
        default=["test_data/examples"],
        help="Folders with example_<usecase>_<n>.ah files (default: test_data/examples only)",
    )
    p_export.add_argument(
        "--per-usecase",
        type=int,
        default=3,
        help="Max numbered examples per usecase stem (test_data)",
    )
    p_export.add_argument("--out", default=str(DEFAULT_OUT))
    p_export.set_defaults(func=cmd_export)

    p_train = sub.add_parser("train", help="QLoRA fine-tune")
    p_train.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    p_train.add_argument("--dataset", default=str(DEFAULT_OUT))
    p_train.add_argument("--output-dir", default=str(DEFAULT_LORA))
    p_train.add_argument("--epochs", type=float, default=2.0)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--grad-accum", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--max-seq-len", type=int, default=2048)
    p_train.add_argument("--lora-r", type=int, default=16)
    p_train.add_argument("--lora-alpha", type=int, default=32)
    p_train.add_argument("--lora-dropout", type=float, default=0.05)
    p_train.add_argument("--save-steps", type=int, default=100)
    p_train.add_argument(
        "--qlora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="4-bit QLoRA (default: on; fits 16GB GPU)",
    )
    p_train.set_defaults(func=cmd_train)

    p_merge = sub.add_parser("merge", help="Merge LoRA into full HF weights")
    p_merge.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    p_merge.add_argument("--adapter", default=str(DEFAULT_LORA))
    p_merge.add_argument(
        "--out",
        default=str(REPO_ROOT / "models" / "code" / "Qwen2.5-Coder-1.5B-Instruct-Anthill-HF"),
    )
    p_merge.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

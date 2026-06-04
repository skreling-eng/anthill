"""$model_ah_create_jsonl — build training JSONL from .ah scripts in files[]."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from externals.model_ah.dataset import rows_from_ah_texts, write_jsonl
from externals.model_ah.paths import read_ah_files
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$model_ah_create_jsonl — convert .ah scripts in files[] to one JSONL training file.

Each input file should look like:

  # Request: Write a JavaScript snippet to retry an HTTP GET with timeout.

  @answer: $code
  Write a JavaScript snippet …
  run @answer

User message = # Request line text (optional request_prefix= prepended); assistant = rest of script.

Example:
  @scripts: $folder('test_data/examples')
  @jsonl: @scripts -> $model_ah_create_jsonl(
      request_prefix='Write an Anthill (.ah) script for this request:'
  )
  @jsonl -> $model_ah_train_lora
"""


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    items = read_ah_files(ctx, inp)
    if not items:
        raise RuntimeError(_HELP.strip())

    request_prefix = inp.args.get("request_prefix", "").strip()
    rows = rows_from_ah_texts(items, request_prefix=request_prefix)
    if not rows:
        raise RuntimeError(
            "No training rows from files[] (need # Request: … and run @… in each .ah file)."
        )

    out_path = ctx.op_dir / "files" / "train.jsonl"
    write_jsonl(rows, out_path)
    rel = str(out_path.relative_to(ctx.base_dir)).replace("\\", "/")

    out = inp.bundle.copy()
    out.files.clear()
    out.files.append(rel)
    return out

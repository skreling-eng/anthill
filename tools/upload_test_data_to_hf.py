#!/usr/bin/env python3
"""Upload local test_data/ to skreling-eng/anthill (incremental).

Example assets (images, wav, mp4) live under test_data/ on the Hub so clones
do not need large binaries in git. Skips files already present on the Hub.

Prefer the unified CLI:
  uv run python tools/upload_to_hf.py --token hf_...
  uv run python tools/upload_to_hf.py --bundle test-data --dry-run

This script is a thin alias for --bundle test-data.

Requires a Write HF token — see tools/upload_models_to_hf.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_tools = Path(__file__).resolve().parent
if str(_tools) not in sys.path:
    sys.path.insert(0, str(_tools))

from upload_to_hf import main  # noqa: E402

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["--bundle", "test-data", *sys.argv[1:]]))

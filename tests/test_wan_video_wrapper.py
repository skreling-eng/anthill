"""ComfyUI-WanVideoWrapper load smoke test (requires .venvs/comfy-wan)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path


class TestWanVideoWrapperLoad(unittest.TestCase):
    def test_load_registers_vace_node(self) -> None:
        venv_py = (
            Path(__file__).resolve().parents[1]
            / ".venvs"
            / "comfy-wan"
            / "Scripts"
            / "python.exe"
        )
        if not venv_py.is_file():
            self.skipTest(".venvs/comfy-wan not installed")

        repo = Path(__file__).resolve().parents[1]
        import subprocess
        import sys

        code = """
import os, sys
from pathlib import Path
repo = Path({repo!r})
sys.path.insert(0, str(repo))
os.environ.pop("AH_COMFY_WAN_WRAPPER", None)
from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs
ensure_comfy_import_stubs()
from externals.comfy_inprocess.bootstrap import bootstrap_comfy
from pathlib import Path as P
bootstrap_comfy(
    input_dir=P(repo) / "comfy_lib" / "input",
    output_dir=P(repo) / "comfy_lib" / "output",
    load_wan_wrapper=True,
)
import nodes
assert "WanVideoVACEStartToEndFrame" in nodes.NODE_CLASS_MAPPINGS
cls = nodes.NODE_CLASS_MAPPINGS["WanVideoVACEStartToEndFrame"]
assert getattr(cls, "FUNCTION", None) == "process"
print("ok")
""".format(
            repo=str(repo)
        )
        proc = subprocess.run(
            [str(venv_py), "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONPATH": str(repo)},
        )
        if proc.returncode != 0:
            self.fail(
                f"wrapper load failed (exit {proc.returncode}):\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        self.assertIn("ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()

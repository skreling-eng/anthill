"""comfy_api.latest exports required by execution.PromptExecutor."""

from __future__ import annotations

import unittest


class TestComfyApiLatest(unittest.TestCase):
    def test_io_export(self) -> None:
        from comfy_api.latest import io, _io

        self.assertIs(io, _io)
        self.assertEqual(io.Hidden.prompt.name, "PROMPT")
        self.assertEqual(io.Combo.io_type, "COMBO")

    def test_prompt_executor_imports(self) -> None:
        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

        ensure_comfy_import_stubs()
        from execution import PromptExecutor  # noqa: F401

    def test_prompt_executor_cache_args(self) -> None:
        from externals.comfy_inprocess.prompt_executor import _prompt_executor_cache_args

        args = _prompt_executor_cache_args()
        self.assertIn("ram", args)
        self.assertIn("ram_inactive", args)
        self.assertEqual(args["ram"], args["ram_inactive"])

    def test_strip_vhs_from_mega_workflow(self) -> None:
        from externals.image2video.comfy_workflow import load_i2v_workflow
        from externals.comfy_inprocess.executor import strip_skipped_workflow_nodes

        wf = load_i2v_workflow("Rapid-AIO-Mega__3_start_image.json")
        stripped = strip_skipped_workflow_nodes(wf)
        types = {n.get("class_type") for n in stripped.values()}
        self.assertNotIn("VHS_VideoCombine", types)
        self.assertIn("VAEDecode", types)

    def test_cache_entry_to_outputs(self) -> None:
        from externals.comfy_inprocess.prompt_executor import (
            _cache_entry_to_outputs,
            _unwrap_output_slot,
        )

        class TensorLike:
            def detach(self):
                return self

        t = TensorLike()
        self.assertIs(_unwrap_output_slot([t]), t)
        self.assertIs(_unwrap_output_slot([[t]]), t)

        class Entry:
            outputs = [[t]]

        self.assertEqual(_cache_entry_to_outputs(Entry()), (t,))

    def test_format_executor_failure(self) -> None:
        from externals.comfy_inprocess.prompt_executor import _format_executor_failure

        class FakeExecutor:
            status_messages = [
                (
                    "execution_error",
                    {
                        "node_id": "8",
                        "node_type": "KSampler",
                        "exception_type": "RuntimeError",
                        "exception_message": "CUDA out of memory",
                        "traceback": ["  File \"x.py\", line 1\n", "RuntimeError: CUDA OOM\n"],
                    },
                )
            ]

        text = _format_executor_failure(FakeExecutor())
        self.assertIn("KSampler", text)
        self.assertIn("CUDA out of memory", text)

    def test_memory_management_ram_cache_api(self) -> None:
        import comfy.memory_management as mm

        mm.set_ram_cache_release_state(None, 0)
        self.assertIsNone(mm.extra_ram_release_callback)
        self.assertEqual(mm.extra_ram_release(1024), 0)


if __name__ == "__main__":
    unittest.main()

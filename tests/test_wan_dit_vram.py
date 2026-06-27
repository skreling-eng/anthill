"""Tests for Wan DiT VRAM parking."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import torch

from externals.comfy_inprocess.wan_dit_vram import (
    _move_module_to_cpu_if_materialized,
    park_dit_for_vae_encode,
)


class _FakeWanVideoModel:
    def __init__(self) -> None:
        self.pipeline = {
            "sd": {
                "diffusion_model.blocks.0.weight": torch.randn(2, 2, device="cuda"),
            },
            "scale_weights": {
                "diffusion_model.blocks.0.scale_weight": torch.randn(2, device="cuda"),
            },
        }
        self.diffusion_model = mock.Mock()
        self.diffusion_model.patched_linear = False
        self.diffusion_model.parameters = lambda: iter(
            [torch.nn.Parameter(torch.randn(2, 2, device="cuda"))]
        )
        self.diffusion_model.to = mock.Mock()

    def __getitem__(self, key: str):
        return self.pipeline[key]


class _MetaTransformer(torch.nn.Module):
    patched_linear = True

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 2, device="meta"))


class TestWanDitVram(unittest.TestCase):
    def test_park_dit_handles_wan_video_model(self) -> None:
        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

        ensure_comfy_import_stubs()
        import comfy.model_management as mm

        model = _FakeWanVideoModel()
        patcher = mock.Mock(model=model)
        transformer = model.diffusion_model

        with mock.patch.object(mm, "unload_all_models"), mock.patch.object(
            mm, "soft_empty_cache"
        ), mock.patch(
            "externals.comfy_inprocess.wan_dit_vram.gpu_allocated_mib",
            side_effect=[1024.0, 128.0],
        ):
            before, after = park_dit_for_vae_encode(patcher, transformer)

        self.assertEqual(before, 1024.0)
        self.assertEqual(after, 128.0)
        sd = model.pipeline["sd"]["diffusion_model.blocks.0.weight"]
        self.assertEqual(sd.device.type, "cpu")
        transformer.to.assert_called_once()

    def test_park_dit_skips_meta_patched_linear_transformer(self) -> None:
        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

        ensure_comfy_import_stubs()
        import comfy.model_management as mm

        model = _FakeWanVideoModel()
        patcher = mock.Mock(model=model)
        transformer = _MetaTransformer()

        with mock.patch.object(mm, "unload_all_models"), mock.patch.object(
            mm, "soft_empty_cache"
        ), mock.patch(
            "externals.comfy_inprocess.wan_dit_vram.gpu_allocated_mib",
            side_effect=[512.0, 64.0],
        ):
            before, after = park_dit_for_vae_encode(patcher, transformer)

        self.assertEqual(before, 512.0)
        self.assertEqual(after, 64.0)
        self.assertEqual(
            model.pipeline["sd"]["diffusion_model.blocks.0.weight"].device.type,
            "cpu",
        )

    def test_move_module_to_cpu_if_materialized_skips_meta(self) -> None:
        transformer = _MetaTransformer()
        _move_module_to_cpu_if_materialized(transformer)
        self.assertEqual(transformer.weight.device.type, "meta")

    def test_log_sampling_vram_readiness_ready(self) -> None:
        from externals.comfy_inprocess.wan_dit_vram import log_sampling_vram_readiness

        mod = torch.nn.Linear(4, 4).cuda()
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_sampling_vram_readiness(mod)
        out = buf.getvalue()
        self.assertIn("sampling VRAM ready", out)
        self.assertIn("cuda=2", out)


if __name__ == "__main__":
    unittest.main()

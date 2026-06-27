"""Tests for avatar sampler perf patches."""

from __future__ import annotations

import sys
import types
import unittest

import externals.avatar.sampler_perf as sampler_perf
from externals.avatar.sampler_perf import begin_avatar_job, end_avatar_job, install_avatar_sampler_perf


class TestAvatarSamplerPerf(unittest.TestCase):
    def tearDown(self) -> None:
        end_avatar_job()
        sampler_perf._WRAPPED = None
        sampler_perf._BLOCKSWAP_WRAPPED = None

    def test_defer_skips_only_first_load(self) -> None:
        begin_avatar_job(defer_transformer_load=True)
        self.assertTrue(sampler_perf._should_skip_initial_load())
        sampler_perf._state.deferred_load_skipped = True
        self.assertFalse(sampler_perf._should_skip_initial_load())

    def test_defer_disabled(self) -> None:
        begin_avatar_job(defer_transformer_load=False)
        self.assertFalse(sampler_perf._should_skip_initial_load())

    def test_rebinds_importer_modules(self) -> None:
        calls: list[str] = []

        def orig(*args, **kwargs):
            calls.append("load")
            return "loaded"

        loading = types.ModuleType("ComfyUI_WanVideoWrapper.nodes_model_loading")
        loading.load_weights = orig
        sampler_mod = types.ModuleType("ComfyUI_WanVideoWrapper.nodes_sampler")
        sampler_mod.load_weights = orig
        sys.modules[loading.__name__] = loading
        sys.modules[sampler_mod.__name__] = sampler_mod
        try:
            sampler_perf._WRAPPED = None
            install_avatar_sampler_perf()
            begin_avatar_job(defer_transformer_load=True)
            self.assertIs(sampler_mod.load_weights, loading.load_weights)
            sampler_mod.load_weights()
            self.assertEqual(calls, [])
            sampler_mod.load_weights()
            self.assertEqual(calls, ["load"])
        finally:
            sys.modules.pop(loading.__name__, None)
            sys.modules.pop(sampler_mod.__name__, None)

    def test_defer_skips_only_first_blockswap(self) -> None:
        calls: list[str] = []

        def orig(*args, **kwargs):
            calls.append("blockswap")
            return None

        utils_mod = types.ModuleType("ComfyUI_WanVideoWrapper.utils")
        utils_mod.init_blockswap = orig
        sampler_mod = types.ModuleType("ComfyUI_WanVideoWrapper.nodes_sampler")
        sampler_mod.init_blockswap = orig
        sys.modules[utils_mod.__name__] = utils_mod
        sys.modules[sampler_mod.__name__] = sampler_mod
        try:
            sampler_perf._BLOCKSWAP_WRAPPED = None
            install_avatar_sampler_perf()
            begin_avatar_job(defer_transformer_load=True)
            sampler_mod.init_blockswap(None, None, None)
            self.assertEqual(calls, [])
            sampler_mod.init_blockswap(None, None, None)
            self.assertEqual(calls, ["blockswap"])
        finally:
            sys.modules.pop(utils_mod.__name__, None)
            sys.modules.pop(sampler_mod.__name__, None)


if __name__ == "__main__":
    unittest.main()

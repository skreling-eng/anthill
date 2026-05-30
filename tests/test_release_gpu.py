"""Tests for GPU release at end of .ah runs."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch


class TestReleaseGpu(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("AH_RELEASE_GPU_ON_RUN_END", None)

    @patch("externals.invoke.terminate_active_subprocesses")
    def test_skipped_when_disabled(self, terminate) -> None:
        os.environ["AH_RELEASE_GPU_ON_RUN_END"] = "0"
        from externals.invoke import release_gpu_resources

        release_gpu_resources(reason="test")
        terminate.assert_not_called()

    @patch("externals.invoke.terminate_active_subprocesses")
    def test_runs_by_default(self, terminate) -> None:
        from externals.invoke import release_gpu_resources

        release_gpu_resources(reason="test")
        terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()

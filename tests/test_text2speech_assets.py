"""Tests for Kokoro asset downloads into models/kokoro."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from externals.text2speech.assets import (
    ensure_config,
    ensure_model_assets,
    ensure_voice_pack,
    weights_filename,
)
from externals.text2speech.model_paths import KOKORO_DIR


class TestText2SpeechAssets(unittest.TestCase):
    @mock.patch("huggingface_hub.hf_hub_download")
    def test_download_uses_local_dir(self, hf_dl: mock.MagicMock) -> None:
        import shutil

        root = KOKORO_DIR / "_test_dl"
        root.mkdir(parents=True, exist_ok=True)
        def _fake_download(**kwargs):
            dest = Path(kwargs["local_dir"]) / kwargs["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("{}", encoding="utf-8")
            return str(dest)

        hf_dl.side_effect = _fake_download
        try:
            ensure_config("hexgrad/Kokoro-82M", root)
            hf_dl.assert_called_once()
            kwargs = hf_dl.call_args.kwargs
            self.assertEqual(kwargs["local_dir"], str(root.resolve()))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_weights_filename_is_v1_only(self) -> None:
        name = weights_filename("hexgrad/Kokoro-82M", KOKORO_DIR)
        self.assertEqual(name, "kokoro-v1_0.pth")

    def test_checkpoint_format_v0(self) -> None:
        v0 = KOKORO_DIR / "kokoro-v0_19.pth"
        if not v0.is_file():
            self.skipTest("kokoro-v0_19.pth not present")
        from externals.text2speech.assets import checkpoint_format

        self.assertEqual(checkpoint_format(v0), "v0")

    @mock.patch("externals.text2speech.assets.ensure_weights")
    @mock.patch("externals.text2speech.assets.ensure_config")
    def test_ensure_model_assets(self, cfg: mock.MagicMock, w: mock.MagicMock) -> None:
        cfg.return_value = KOKORO_DIR / "config.json"
        w.return_value = KOKORO_DIR / "kokoro-v1_0.pth"
        c, m = ensure_model_assets("hexgrad/Kokoro-82M", KOKORO_DIR)
        self.assertEqual(c, cfg.return_value)
        self.assertEqual(m, w.return_value)

    @mock.patch("externals.text2speech.assets._download")
    def test_ensure_voice_pack_path(self, dl: mock.MagicMock) -> None:
        import shutil

        root = KOKORO_DIR / "_test_voice"
        root.mkdir(parents=True, exist_ok=True)
        target = root / "voices" / "af_bella.pt"
        dl.return_value = target
        try:
            path = ensure_voice_pack("af_bella", root=root)
            dl.assert_called_once_with(
                "hexgrad/Kokoro-82M", "voices/af_bella.pt", root
            )
            self.assertEqual(path, target)
        finally:
            shutil.rmtree(root, ignore_errors=True)

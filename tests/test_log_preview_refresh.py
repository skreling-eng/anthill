"""Tests for live-regenerated media previews in the action log."""

from __future__ import annotations

import shutil
import time
import unittest
from pathlib import Path

from app import Interface, LinkApi
from externals.music_separation.audio_io import write_wav_bytes
import numpy as np


class TestLogPreviewRefresh(unittest.TestCase):
    def test_html_page_regenerates_audio_when_file_late(self) -> None:
        api = LinkApi(Path(".").resolve())
        ui = Interface(api)
        session = Path("sessions/test_log_preview_refresh_late")
        if session.exists():
            shutil.rmtree(session)
        session.mkdir(parents=True, exist_ok=True)
        ui.session_dir = session

        rel = "6__join_stems/sounds/0.wav"
        output_context = {"sounds": [rel], "prompts": [], "texts": [], "images": [], "videos": [], "files": [], "changes": []}
        ui.data.append(
            {
                "tm": "t1",
                "data": "<b>FINISH</b> $join_stems",
                "finish_preview": {
                    "action_name": "$join_stems",
                    "output_context": output_context,
                    "session_base_dir": str(session.resolve()),
                },
            }
        )

        html_before = ui.html_page()
        self.assertNotIn("<audio", html_before)

        path = session / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tone = (np.sin(np.linspace(0, 4 * np.pi, 4000)) * 0.1).astype(np.float32)
        stereo = np.stack([tone, tone])
        path.write_bytes(write_wav_bytes(stereo, 44100))
        time.sleep(0.05)

        html_after = ui.html_page()
        self.assertIn("<audio", html_after)
        self.assertNotIn("result-missing", html_after)

    def test_old_entries_keep_working_when_session_dir_changes(self) -> None:
        api = LinkApi(Path(".").resolve())
        ui = Interface(api)
        old_session = Path("sessions/test_log_preview_old").resolve()
        old_session.mkdir(parents=True, exist_ok=True)
        rel = "6__join_stems/sounds/0.wav"
        wav = old_session / rel
        wav.parent.mkdir(parents=True, exist_ok=True)
        tone = (np.sin(np.linspace(0, 4 * np.pi, 4000)) * 0.1).astype(np.float32)
        stereo = np.stack([tone, tone])
        wav.write_bytes(write_wav_bytes(stereo, 44100))

        frozen_uri = wav.as_uri()
        ui.data.append(
            {
                "tm": "t1",
                "data": (
                    f'<b>FINISH</b> $join_stems<br><audio src="{frozen_uri}" '
                    f'controls preload="metadata"></audio>'
                ),
                "input_json_ref": (
                    f"input_json('sessions/test_log_preview_old/6__join_stems/output.json')"
                ),
                "finish_preview": {
                    "action_name": "$join_stems",
                    "output_context": {
                        "sounds": [rel],
                        "prompts": [],
                        "texts": [],
                        "images": [],
                        "videos": [],
                        "files": [],
                        "changes": [],
                    },
                    "session_base_dir": str(old_session),
                },
            }
        )

        ui.session_dir = Path("sessions/test_log_preview_new_run")
        ui.session_dir.mkdir(parents=True, exist_ok=True)
        html = ui.html_page()
        self.assertIn(frozen_uri, html)
        self.assertNotIn("result-missing", html)

    def test_missing_session_falls_back_to_frozen_html(self) -> None:
        api = LinkApi(Path(".").resolve())
        ui = Interface(api)
        frozen = '<b>FINISH</b> $join_stems<br><audio src="file:///good.wav"></audio>'
        ui.data.append(
            {
                "tm": "t1",
                "data": frozen,
                "finish_preview": {
                    "action_name": "$join_stems",
                    "output_context": {"sounds": ["6__join_stems/sounds/0.wav"]},
                },
            }
        )
        ui.session_dir = Path("sessions/empty_new_session")
        self.assertEqual(ui._entry_body_html(ui.data[0]), frozen)


if __name__ == "__main__":
    unittest.main()

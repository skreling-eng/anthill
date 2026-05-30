"""Tests for waiting on output files before action_finish callbacks."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
from externals.music_separation.audio_io import write_wav_bytes
import numpy as np


class TestBundleFilesReady(unittest.TestCase):
    def test_waits_for_delayed_write(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("join_stems")
        rel = "sounds/0.wav"
        path = session.resolve_link_path(f"{op_dir.name}/{rel}")

        def _write_late() -> None:
            time.sleep(0.15)
            path.parent.mkdir(parents=True, exist_ok=True)
            tone = (np.sin(np.linspace(0, 4 * np.pi, 2000)) * 0.1).astype(np.float32)
            stereo = np.stack([tone, tone])
            data = write_wav_bytes(stereo, 44100)
            with open(path, "wb") as handle:
                handle.write(data)
                handle.flush()

        thread = threading.Thread(target=_write_late, daemon=True)
        thread.start()

        bundle = ArrayBundle(sounds=[f"{op_dir.name}/{rel}"])
        session.ensure_bundle_files_ready(bundle, timeout=5.0)
        thread.join(timeout=2.0)
        self.assertGreaterEqual(path.stat().st_size, 44)

    def test_rejects_empty_wav_placeholder(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("join_stems")
        rel = f"{op_dir.name}/sounds/0.wav"
        path = session.resolve_link_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

        bundle = ArrayBundle(sounds=[rel])
        with self.assertRaises(FileNotFoundError):
            session.ensure_bundle_files_ready(bundle, timeout=0.2)

    def test_materialize_skips_missing_wav(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("join_stems")
        rel = f"{op_dir.name}/sounds/0.wav"
        session.write_bundle(op_dir, ArrayBundle(sounds=[rel]), "output")
        path = session.resolve_link_path(rel)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

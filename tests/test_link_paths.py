"""Tests for bundle link path resolution (launch-dir output/ exports)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
from ahlib.link_paths import resolve_link_path


class TestLinkPaths(unittest.TestCase):
    def test_output_export_resolves_under_launch_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launch = Path(tmp)
            sessions_root = launch / "sessions"
            session_dir = create_session_dir(sessions_root)
            session = Session(session_dir, sessions_root=sessions_root)
            export_file = (
                launch
                / "output"
                / "20260606_040316_12372_877000"
                / "images"
                / "0000_test.png"
            )
            export_file.parent.mkdir(parents=True, exist_ok=True)
            export_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\r")

            os.environ["AH_LAUNCH_DIR"] = str(launch)
            try:
                link = "output/20260606_040316_12372_877000/images/0000_test.png"
                resolved = resolve_link_path(session, link)
                self.assertEqual(resolved, export_file.resolve())

                bundle = ArrayBundle(images=[link])
                session.ensure_bundle_files_ready(bundle)
            finally:
                os.environ.pop("AH_LAUNCH_DIR", None)


if __name__ == "__main__":
    unittest.main()

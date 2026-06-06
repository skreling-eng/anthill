"""Tests for $input_json external."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
from externals.api import ExternalContext, ExternalInput

run = importlib.import_module("externals.input_json.run").run


class TestInputJsonExternal(unittest.TestCase):
    def test_resolves_output_exports_from_launch_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launch = Path(tmp)
            sessions_root = launch / "sessions"
            src_session = create_session_dir(sessions_root)
            src_id = src_session.name
            export_file = launch / "output" / src_id / "images" / "0000_test.png"
            export_file.parent.mkdir(parents=True, exist_ok=True)
            export_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\r")

            op_dir = src_session / "1__output"
            op_dir.mkdir(parents=True)
            manifest = {
                "images": [f"../../output/{src_id}/images/0000_test.png"],
            }
            json_path = op_dir / "output.json"
            json_path.write_text(json.dumps(manifest), encoding="utf-8")

            cur_session = create_session_dir(sessions_root)
            session = Session(
                cur_session,
                sessions_root=sessions_root,
            )
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("input_json"))
            os.environ["AH_LAUNCH_DIR"] = str(launch)
            try:
                ref = f"sessions/{src_id}/1__output/output.json"
                out = run(
                    ctx,
                    ExternalInput(bundle=ArrayBundle(), args={"_path": ref}, prompt_text=""),
                )
            finally:
                os.environ.pop("AH_LAUNCH_DIR", None)

            resolved = Path(out.images[0])
            self.assertTrue(resolved.is_file())
            self.assertEqual(resolved, export_file.resolve())
            session.ensure_bundle_files_ready(out)

    def test_resolves_legacy_output_link_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launch = Path(tmp)
            sessions_root = launch / "sessions"
            src_session = create_session_dir(sessions_root)
            src_id = src_session.name
            export_file = launch / "output" / src_id / "images" / "0000_test.png"
            export_file.parent.mkdir(parents=True, exist_ok=True)
            export_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\r")

            cur_session = create_session_dir(sessions_root)
            session = Session(cur_session, sessions_root=sessions_root)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("input_json"))
            os.environ["AH_LAUNCH_DIR"] = str(launch)
            try:
                manifest_path = cur_session / "input.json"
                manifest_path.write_text(
                    json.dumps(
                        {"images": [f"output/{src_id}/images/0000_test.png"]}
                    ),
                    encoding="utf-8",
                )
                out = run(
                    ctx,
                    ExternalInput(
                        bundle=ArrayBundle(),
                        args={"_path": str(manifest_path)},
                        prompt_text="",
                    ),
                )
            finally:
                os.environ.pop("AH_LAUNCH_DIR", None)

            resolved = Path(out.images[0])
            self.assertEqual(resolved, export_file.resolve())
            self.assertNotIn("sessions", resolved.parts[resolved.parts.index("output") - 1 :])

    def test_resolves_session_relative_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launch = Path(tmp)
            sessions_root = launch / "sessions"
            src_session = create_session_dir(sessions_root)
            src_id = src_session.name
            image_dir = src_session / "2__image" / "images"
            image_dir.mkdir(parents=True)
            image_file = image_dir / "0.png"
            image_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\r")

            op_dir = src_session / "2__image"
            json_path = op_dir / "output.json"
            json_path.write_text(
                json.dumps({"images": ["2__image/images/0.png"]}),
                encoding="utf-8",
            )

            cur_session = create_session_dir(sessions_root)
            session = Session(cur_session, sessions_root=sessions_root)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("input_json"))
            os.environ["AH_LAUNCH_DIR"] = str(launch)
            try:
                out = run(
                    ctx,
                    ExternalInput(
                        bundle=ArrayBundle(),
                        args={"_path": f"sessions/{src_id}/2__image/output.json"},
                        prompt_text="",
                    ),
                )
            finally:
                os.environ.pop("AH_LAUNCH_DIR", None)

            self.assertEqual(Path(out.images[0]), image_file.resolve())


if __name__ == "__main__":
    unittest.main()

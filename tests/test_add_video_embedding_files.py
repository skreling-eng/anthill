"""Tests for $add_video_embedding_files external."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.add_video_embedding_files.run import run
from externals.image2embedding.embedding_format import EMBED_DIM, unpack_siglip_embedding
from externals.video_index.ahvemb import ahvemb_path_for_video
from externals.video2embedding.frames import read_video_frames, sample_fragment_frames
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestAddVideoEmbeddingFilesParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions(
            "$add_video_embedding_files(every=500, threshold=12, min_frames=80)"
        )
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "add_video_embedding_files")
        self.assertEqual(expr.args.get("every"), "500")
        self.assertEqual(expr.args.get("threshold"), "12")


    def test_parse_overwrite(self) -> None:
        expr = parse_actions("$add_video_embedding_files(overwrite=True)")
        self.assertEqual(expr.args.get("overwrite"), "True")


class TestSampleFragmentFrames(unittest.TestCase):
    def test_every_nth_for_frame_count(self) -> None:
        from externals.video2embedding.frames import every_nth_for_frame_count

        self.assertEqual(every_nth_for_frame_count(250), 50)
        self.assertEqual(every_nth_for_frame_count(1200), 240)
        self.assertEqual(every_nth_for_frame_count(5), 1)
        self.assertEqual(every_nth_for_frame_count(0), 1)

    def test_auto_samples_about_five_frames(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("opencv not installed")

        path = Path("sessions") / "_test_add_video_embedding_files_auto.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (16, 16),
        )
        try:
            for i in range(250):
                frame = np.full((16, 16, 3), i % 256, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

        try:
            frames = sample_fragment_frames(path, 0, 250)
            self.assertEqual(len(frames), 5)
        finally:
            path.unlink(missing_ok=True)

    def test_read_video_frames_by_index(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("opencv not installed")

        path = Path("sessions") / "_test_read_video_frames.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (16, 16),
        )
        try:
            for i in range(20):
                frame = np.full((16, 16, 3), i, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

        try:
            frames = read_video_frames(path, [3, 7, 11])
            self.assertEqual(len(frames), 3)
        finally:
            path.unlink(missing_ok=True)

    def test_samples_within_range(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("opencv not installed")

        path = Path("sessions") / "_test_add_video_embedding_files.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (16, 16),
        )
        try:
            for i in range(1200):
                frame = np.full((16, 16, 3), i % 256, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

        try:
            frames = sample_fragment_frames(path, 0, 1200, every_nth=500)
            self.assertEqual(len(frames), 2)
            frames_partial = sample_fragment_frames(path, 100, 700, every_nth=500)
            self.assertEqual(len(frames_partial), 1)
        finally:
            path.unlink(missing_ok=True)


class TestAddVideoEmbeddingFilesRun(unittest.TestCase):
    def test_emulate_writes_ahvemb_sidecar(self) -> None:
        os.environ["AH_EMULATE_ADD_VIDEO_EMBEDDING_FILES"] = "1"
        try:
            try:
                import cv2
                import numpy as np
            except ImportError:
                self.skipTest("opencv not installed")

            with tempfile.TemporaryDirectory() as tmp:
                video = Path(tmp) / "clip.mp4"
                writer = cv2.VideoWriter(
                    str(video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10.0,
                    (16, 16),
                )
                for i in range(250):
                    writer.write(np.full((16, 16, 3), i % 256, dtype=np.uint8))
                writer.release()

                session_dir = create_session_dir(Path("sessions"))
                session = Session(session_dir)
                ctx = ExternalContext(
                    session=session,
                    op_dir=session.next_op_dir("add_video_embedding_files"),
                )
                link = str(video.resolve()).replace("\\", "/")
                inp = ExternalInput(
                    bundle=ArrayBundle(videos=[link]),
                    args={"every": "500", "min_frames": "50", "threshold": "0"},
                    prompt_text="",
                )
                out = run(ctx, inp)
                self.assertEqual(out.videos, [link])

                sidecar = ahvemb_path_for_video(video)
                self.assertTrue(sidecar.is_file())
                lines = sidecar.read_text(encoding="utf-8").strip().splitlines()
                self.assertGreaterEqual(len(lines), 1)
                start, end, encoded = lines[0].split(" ", 2)
                self.assertTrue(start.isdigit())
                self.assertTrue(end.isdigit())
                self.assertEqual(len(unpack_siglip_embedding(encoded)), EMBED_DIM)
        finally:
            os.environ.pop("AH_EMULATE_ADD_VIDEO_EMBEDDING_FILES", None)

    def test_skips_existing_sidecar_by_default(self) -> None:
        os.environ["AH_EMULATE_ADD_VIDEO_EMBEDDING_FILES"] = "1"
        try:
            try:
                import cv2
                import numpy as np
            except ImportError:
                self.skipTest("opencv not installed")

            with tempfile.TemporaryDirectory() as tmp:
                video = Path(tmp) / "clip.mp4"
                writer = cv2.VideoWriter(
                    str(video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10.0,
                    (16, 16),
                )
                for i in range(250):
                    writer.write(np.full((16, 16, 3), i % 256, dtype=np.uint8))
                writer.release()

                sidecar = ahvemb_path_for_video(video)
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text("0 1 placeholder\n", encoding="utf-8")

                session_dir = create_session_dir(Path("sessions"))
                session = Session(session_dir)
                ctx = ExternalContext(
                    session=session,
                    op_dir=session.next_op_dir("add_video_embedding_files"),
                )
                link = str(video.resolve()).replace("\\", "/")
                inp = ExternalInput(
                    bundle=ArrayBundle(videos=[link]),
                    args={"every": "500", "min_frames": "50", "threshold": "0"},
                    prompt_text="",
                )
                run(ctx, inp)
                self.assertEqual(sidecar.read_text(encoding="utf-8"), "0 1 placeholder\n")
        finally:
            os.environ.pop("AH_EMULATE_ADD_VIDEO_EMBEDDING_FILES", None)

    def test_overwrite_rewrites_existing_sidecar(self) -> None:
        os.environ["AH_EMULATE_ADD_VIDEO_EMBEDDING_FILES"] = "1"
        try:
            try:
                import cv2
                import numpy as np
            except ImportError:
                self.skipTest("opencv not installed")

            with tempfile.TemporaryDirectory() as tmp:
                video = Path(tmp) / "clip.mp4"
                writer = cv2.VideoWriter(
                    str(video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10.0,
                    (16, 16),
                )
                for i in range(250):
                    writer.write(np.full((16, 16, 3), i % 256, dtype=np.uint8))
                writer.release()

                sidecar = ahvemb_path_for_video(video)
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text("0 1 placeholder\n", encoding="utf-8")

                session_dir = create_session_dir(Path("sessions"))
                session = Session(session_dir)
                ctx = ExternalContext(
                    session=session,
                    op_dir=session.next_op_dir("add_video_embedding_files"),
                )
                link = str(video.resolve()).replace("\\", "/")
                inp = ExternalInput(
                    bundle=ArrayBundle(videos=[link]),
                    args={
                        "every": "500",
                        "min_frames": "50",
                        "threshold": "0",
                        "overwrite": "True",
                    },
                    prompt_text="",
                )
                run(ctx, inp)
                lines = sidecar.read_text(encoding="utf-8").strip().splitlines()
                self.assertGreaterEqual(len(lines), 1)
                _, _, encoded = lines[0].split(" ", 2)
                self.assertEqual(len(unpack_siglip_embedding(encoded)), EMBED_DIM)
        finally:
            os.environ.pop("AH_EMULATE_ADD_VIDEO_EMBEDDING_FILES", None)

    def test_no_videos_passthrough(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        ctx = ExternalContext(
            session=session,
            op_dir=session.next_op_dir("add_video_embedding_files"),
        )
        bundle = ArrayBundle(videos=["missing.mp4"])
        inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
        out = run(ctx, inp)
        self.assertEqual(out.videos, bundle.videos)


if __name__ == "__main__":
    unittest.main()

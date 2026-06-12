"""Tests for $video2embedding external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.image2embedding.embedding_format import EMBED_DIM, unpack_siglip_embedding
from externals.video2embedding.frames import sample_video_frames
from externals.video2embedding.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestVideo2EmbeddingParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$video2embedding(every=10, gpu=True)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "video2embedding")
        self.assertEqual(expr.args.get("every"), "10")


class TestVideo2EmbeddingFrames(unittest.TestCase):
    def test_auto_samples_about_five_frames(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("opencv not installed")

        path = Path("sessions") / "_test_video2embedding_auto.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (16, 16),
        )
        try:
            for i in range(100):
                frame = np.full((16, 16, 3), i % 256, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

        try:
            frames = sample_video_frames(path)
            self.assertEqual(len(frames), 5)
        finally:
            path.unlink(missing_ok=True)

    def test_sample_every_nth(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("opencv not installed")

        path = Path("sessions") / "_test_video2embedding.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (16, 16),
        )
        try:
            for i in range(25):
                frame = np.full((16, 16, 3), i * 10, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

        try:
            frames = sample_video_frames(path, every_nth=10)
            self.assertEqual(len(frames), 2)
        finally:
            path.unlink(missing_ok=True)

    def test_short_video_uses_first_frame(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("opencv not installed")

        path = Path("sessions") / "_test_video2embedding_short.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (16, 16),
        )
        try:
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        finally:
            writer.release()

        try:
            frames = sample_video_frames(path, every_nth=20)
            self.assertEqual(len(frames), 1)
        finally:
            path.unlink(missing_ok=True)


class TestVideo2EmbeddingRun(unittest.TestCase):
    def test_emulate_one_embedding_per_video(self) -> None:
        os.environ["AH_EMULATE_VIDEO2EMBEDDING"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("video2embedding")
            )
            bundle = ArrayBundle()
            bundle.videos.append(
                ctx.new_link("videos", ".mp4", b"\x00\x00\x00\x18ftypisom\x00")
            )
            inp = ExternalInput(bundle=bundle, args={"every": "20"}, prompt_text="")
            out = run(ctx, inp)
            self.assertEqual(len(out.videos), 1)
            self.assertEqual(len(out.embeddings), 1)
            self.assertEqual(len(unpack_siglip_embedding(out.embeddings[0])), EMBED_DIM)
        finally:
            os.environ.pop("AH_EMULATE_VIDEO2EMBEDDING", None)

    def test_no_videos_clears_embeddings(self) -> None:
        os.environ["AH_EMULATE_VIDEO2EMBEDDING"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("video2embedding")
            )
            inp = ExternalInput(
                bundle=ArrayBundle(embeddings=["abc"]),
                args={},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(out.embeddings, [])
        finally:
            os.environ.pop("AH_EMULATE_VIDEO2EMBEDDING", None)

    def test_average_path(self) -> None:
        from externals.image2embedding.embedding_format import pack_averaged_siglip_embeddings
        import numpy as np

        v1 = np.ones(768, dtype=np.float32)
        v1 /= np.linalg.norm(v1)
        v2 = np.full(768, 2.0, dtype=np.float32)
        v2 /= np.linalg.norm(v2)
        encoded = pack_averaged_siglip_embeddings([v1, v2])
        self.assertEqual(len(unpack_siglip_embedding(encoded)), EMBED_DIM)


if __name__ == "__main__":
    unittest.main()

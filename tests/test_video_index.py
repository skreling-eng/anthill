"""Tests for $create_video_index and $search_local_video."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.create_video_index.run import run as create_index_run
from externals.search_local_video.run import _output_name, run as search_run
from externals.video_index.ahvemb import ahvemb_path_for_video, parse_ahvemb_file
from externals.video_index.store import (
    FAISS_SUFFIX,
    INDEX_STEM,
    MAP_SUFFIX,
    brute_force_search,
    build_combined_mapping,
    build_faiss_index,
    load_index_pair,
    save_index_pair,
)
from externals.image2embedding.embedding_format import (
    EMBED_DIM,
    emulated_siglip_embedding,
    unpack_siglip_embedding,
)
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


def _write_test_video(path: Path, frames: int = 120) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (16, 16),
    )
    for i in range(frames):
        writer.write(np.full((16, 16, 3), i % 256, dtype=np.uint8))
    writer.release()


def _write_sidecar(video: Path, content: str) -> Path:
    sidecar = ahvemb_path_for_video(video)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(content, encoding="utf-8")
    return sidecar


class TestCreateVideoIndexParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$create_video_index")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "create_video_index")


class TestSearchLocalVideoParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$search_local_video(n=5)")
        self.assertEqual(expr.name, "search_local_video")
        self.assertEqual(expr.args.get("n"), "5")


class TestAhvembAndIndex(unittest.TestCase):
    def test_ahvemb_path_for_video(self) -> None:
        video = Path("test_data/videos/clip.mp4")
        expected = video.resolve().parent / ".ahvemb" / "clip.ahvemb"
        self.assertEqual(ahvemb_path_for_video(video), expected)

    def test_parse_ahvemb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.ahvemb"
            path.write_text(
                f"0 100 {emulated_siglip_embedding('demo')}\n", encoding="utf-8"
            )
            rows = parse_ahvemb_file(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][:2], (0, 100))
            self.assertEqual(len(rows[0][2]), EMBED_DIM)

    def test_combined_index(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")

        vec_a = unpack_siglip_embedding(emulated_siglip_embedding("a"))
        vec_b = unpack_siglip_embedding(emulated_siglip_embedding("b"))
        with tempfile.TemporaryDirectory() as tmp:
            video_a = Path(tmp) / "a.mp4"
            video_b = Path(tmp) / "b.mp4"
            ahvemb_a = ahvemb_path_for_video(video_a)
            ahvemb_b = ahvemb_path_for_video(video_b)
            video_a.write_bytes(b"fake-a")
            video_b.write_bytes(b"fake-b")
            ahvemb_a.parent.mkdir(parents=True, exist_ok=True)
            ahvemb_b.parent.mkdir(parents=True, exist_ok=True)
            ahvemb_a.write_text(f"0 50 {emulated_siglip_embedding('a')}\n", encoding="utf-8")
            ahvemb_b.write_text(f"0 80 {emulated_siglip_embedding('b')}\n", encoding="utf-8")
            rows = [
                (video_a, ahvemb_a, 25.0, 0, 50, vec_a),
                (video_b, ahvemb_b, 30.0, 0, 80, vec_b),
            ]
            meta = build_combined_mapping(rows)
            index = build_faiss_index(rows)
            index_path = Path(tmp) / f"{INDEX_STEM}{FAISS_SUFFIX}"
            save_index_pair(index_path, index, meta)
            loaded, loaded_meta = load_index_pair(index_path)
            self.assertEqual(len(loaded_meta["fragments"]), 2)
            hits = brute_force_search(
                loaded_meta, [vec_a, vec_b], vec_b, k=1
            )
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["video"], str(video_b.resolve()).replace("\\", "/"))


class TestCreateVideoIndexRun(unittest.TestCase):
    def test_emulate_outputs_single_index_link(self) -> None:
        os.environ["AH_EMULATE_CREATE_VIDEO_INDEX"] = "1"
        try:
            try:
                import cv2  # noqa: F401
            except ImportError:
                self.skipTest("opencv not installed")

            with tempfile.TemporaryDirectory() as tmp:
                video_a = Path(tmp) / "a.mp4"
                video_b = Path(tmp) / "b.mp4"
                _write_test_video(video_a)
                _write_test_video(video_b)
                _write_sidecar(
                    video_a, f"0 120 {emulated_siglip_embedding('a')}\n"
                )
                _write_sidecar(
                    video_b, f"0 120 {emulated_siglip_embedding('b')}\n"
                )

                session_dir = create_session_dir(Path("sessions"))
                session = Session(session_dir)
                ctx = ExternalContext(
                    session=session, op_dir=session.next_op_dir("create_video_index")
                )
                links = [
                    str(video_a.resolve()).replace("\\", "/"),
                    str(video_b.resolve()).replace("\\", "/"),
                ]
                inp = ExternalInput(
                    bundle=ArrayBundle(videos=links, prompts=["keep"]),
                    args={},
                    prompt_text="",
                )
                out = create_index_run(ctx, inp)
                self.assertEqual(out.prompts, [])
                self.assertEqual(out.videos, [])
                self.assertEqual(len(out.files), 1)
                self.assertTrue(
                    out.files[0].endswith(MAP_SUFFIX) or out.files[0].endswith(FAISS_SUFFIX)
                )
        finally:
            os.environ.pop("AH_EMULATE_CREATE_VIDEO_INDEX", None)


class TestSearchLocalVideoOutputName(unittest.TestCase):
    def test_truncates_long_source_stem(self) -> None:
        long_stem = "a" * 300
        name = _output_name(long_stem, 0, 120)
        self.assertLessEqual(len(name), 260)
        self.assertTrue(name.endswith("_search000_f120.mp4"))
        self.assertIn("a" * 150, name)
        self.assertNotIn("a" * 151, name)

    def test_preserves_unicode_stem(self) -> None:
        name = _output_name("视频 clip (demo)", 1, 50)
        self.assertIn("视频", name)
        self.assertIn("clip", name)
        self.assertLessEqual(len(name), 260)

    def test_replaces_invalid_path_characters(self) -> None:
        name = _output_name('bad:name*test', 0, 10)
        self.assertNotIn(":", name)
        self.assertNotIn("*", name)


class TestSearchLocalVideoRun(unittest.TestCase):
    def test_emulate_returns_labeled_fragments(self) -> None:
        os.environ["AH_EMULATE_CREATE_VIDEO_INDEX"] = "1"
        os.environ["AH_EMULATE_SEARCH_LOCAL_VIDEO"] = "1"
        try:
            try:
                import cv2  # noqa: F401
            except ImportError:
                self.skipTest("opencv not installed")

            with tempfile.TemporaryDirectory() as tmp:
                video = Path(tmp) / "clip.mp4"
                _write_test_video(video)
                _write_sidecar(
                    video, f"0 120 {emulated_siglip_embedding('0:120')}\n"
                )

                session_dir = create_session_dir(Path("sessions"))
                session = Session(session_dir)
                create_ctx = ExternalContext(
                    session=session, op_dir=session.next_op_dir("create_video_index")
                )
                video_link = str(video.resolve()).replace("\\", "/")
                index_out = create_index_run(
                    create_ctx,
                    ExternalInput(
                        bundle=ArrayBundle(videos=[video_link]),
                        args={},
                        prompt_text="",
                    ),
                )

                search_ctx = ExternalContext(
                    session=session, op_dir=session.next_op_dir("search_local_video")
                )
                out = search_run(
                    search_ctx,
                    ExternalInput(
                        bundle=ArrayBundle(files=list(index_out.files)),
                        args={"n": "1"},
                        prompt_text="beach scene",
                    ),
                )
                self.assertEqual(len(out.videos), 1)
                self.assertEqual(len(out.labels), 1)
                self.assertEqual(out.labels[0][0], "search_results")
                meta = out.labels[0][2]
                self.assertIn("closeness", meta)
                self.assertIn("start", meta)
                self.assertIn("end", meta)
                self.assertIn("ahvemb", meta)
        finally:
            os.environ.pop("AH_EMULATE_CREATE_VIDEO_INDEX", None)
            os.environ.pop("AH_EMULATE_SEARCH_LOCAL_VIDEO", None)

    def test_emulate_searches_each_prompt(self) -> None:
        os.environ["AH_EMULATE_CREATE_VIDEO_INDEX"] = "1"
        os.environ["AH_EMULATE_SEARCH_LOCAL_VIDEO"] = "1"
        try:
            try:
                import cv2  # noqa: F401
            except ImportError:
                self.skipTest("opencv not installed")

            with tempfile.TemporaryDirectory() as tmp:
                video = Path(tmp) / "clip.mp4"
                _write_test_video(video)
                _write_sidecar(
                    video, f"0 120 {emulated_siglip_embedding('0:120')}\n"
                )

                session_dir = create_session_dir(Path("sessions"))
                session = Session(session_dir)
                create_ctx = ExternalContext(
                    session=session, op_dir=session.next_op_dir("create_video_index")
                )
                video_link = str(video.resolve()).replace("\\", "/")
                index_out = create_index_run(
                    create_ctx,
                    ExternalInput(
                        bundle=ArrayBundle(videos=[video_link]),
                        args={},
                        prompt_text="",
                    ),
                )

                search_ctx = ExternalContext(
                    session=session, op_dir=session.next_op_dir("search_local_video")
                )
                prompts = [
                    search_ctx.new_link("prompts", ".txt", "beach scene\n"),
                    search_ctx.new_link("prompts", ".txt", "mountain scene\n"),
                ]
                out = search_run(
                    search_ctx,
                    ExternalInput(
                        bundle=ArrayBundle(
                            files=list(index_out.files),
                            prompts=prompts,
                        ),
                        args={"n": "1"},
                        prompt_text="",
                    ),
                )
                self.assertEqual(len(out.videos), 2)
                self.assertEqual(len(out.labels), 2)
                queries = {label[2]["query"] for label in out.labels}
                self.assertEqual(queries, {"beach scene", "mountain scene"})
        finally:
            os.environ.pop("AH_EMULATE_CREATE_VIDEO_INDEX", None)
            os.environ.pop("AH_EMULATE_SEARCH_LOCAL_VIDEO", None)

    def test_emulate_searches_by_video(self) -> None:
        os.environ["AH_EMULATE_CREATE_VIDEO_INDEX"] = "1"
        os.environ["AH_EMULATE_SEARCH_LOCAL_VIDEO"] = "1"
        try:
            try:
                import cv2  # noqa: F401
            except ImportError:
                self.skipTest("opencv not installed")

            with tempfile.TemporaryDirectory() as tmp:
                video = Path(tmp) / "clip.mp4"
                query_video = Path(tmp) / "query.mp4"
                _write_test_video(video)
                _write_test_video(query_video)
                _write_sidecar(
                    video, f"0 120 {emulated_siglip_embedding('0:120')}\n"
                )

                session_dir = create_session_dir(Path("sessions"))
                session = Session(session_dir)
                create_ctx = ExternalContext(
                    session=session, op_dir=session.next_op_dir("create_video_index")
                )
                video_link = str(video.resolve()).replace("\\", "/")
                index_out = create_index_run(
                    create_ctx,
                    ExternalInput(
                        bundle=ArrayBundle(videos=[video_link]),
                        args={},
                        prompt_text="",
                    ),
                )

                search_ctx = ExternalContext(
                    session=session, op_dir=session.next_op_dir("search_local_video")
                )
                query_link = str(query_video.resolve()).replace("\\", "/")
                out = search_run(
                    search_ctx,
                    ExternalInput(
                        bundle=ArrayBundle(
                            files=list(index_out.files),
                            videos=[query_link],
                        ),
                        args={"n": "1"},
                        prompt_text="",
                    ),
                )
                self.assertEqual(len(out.videos), 1)
                self.assertEqual(out.labels[0][2]["query"], query_link)
        finally:
            os.environ.pop("AH_EMULATE_CREATE_VIDEO_INDEX", None)
            os.environ.pop("AH_EMULATE_SEARCH_LOCAL_VIDEO", None)


if __name__ == "__main__":
    unittest.main()

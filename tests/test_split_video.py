"""Tests for $split_video fragment logic."""

from __future__ import annotations

import os
import random
import unittest
from dataclasses import dataclass
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.split_video.run import run
from externals.split_video.split import (
    VideoFragment,
    merge_sample_images,
    merge_small_fragments,
)
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


@dataclass(frozen=True)
class _Hash:
    value: int

    def __sub__(self, other: "_Hash") -> int:
        return self.value - other.value


class TestSplitVideoMerge(unittest.TestCase):
    def _frag(self, start: int, end: int, first: int, last: int) -> VideoFragment:
        return VideoFragment(start, end, _Hash(first), _Hash(last))

    def test_merge_two_adjacent_short_fragments(self) -> None:
        fragments = [
            self._frag(0, 40, 1, 2),
            self._frag(40, 70, 3, 4),
            self._frag(70, 200, 5, 6),
        ]
        merged = merge_small_fragments(fragments, min_frames=100)
        # 40+30 -> 70 still short, then joins the larger next neighbor
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].start, 0)
        self.assertEqual(merged[0].end, 200)

    def test_merge_pair_of_short_fragments_to_reach_min(self) -> None:
        fragments = [
            self._frag(0, 50, 1, 2),
            self._frag(50, 120, 3, 4),
        ]
        merged = merge_small_fragments(fragments, min_frames=100)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].size, 120)

    def test_merge_short_into_neighbor_with_smaller_boundary_diff(self) -> None:
        fragments = [
            self._frag(0, 200, 1, 10),
            self._frag(200, 240, 11, 12),
            self._frag(240, 500, 50, 60),
        ]
        merged = merge_small_fragments(fragments, min_frames=100)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].start, 0)
        self.assertEqual(merged[0].end, 240)

    def test_no_merge_when_fragment_is_long_enough(self) -> None:
        fragments = [self._frag(0, 150, 1, 2), self._frag(150, 300, 3, 4)]
        merged = merge_small_fragments(fragments, min_frames=100)
        self.assertEqual(len(merged), 2)


class TestFragmentSampleImages(unittest.TestCase):
    def test_merge_sample_images_shuffles_combined_pool(self) -> None:
        random.seed(0)
        left = tuple(f"L{i}" for i in range(5))
        right = tuple(f"R{i}" for i in range(5))
        merged = merge_sample_images(left, right, 5)
        self.assertEqual(len(merged), 5)
        self.assertEqual(len(set(merged)), 5)

    def test_merge_small_fragments_recombines_sample_images(self) -> None:
        random.seed(0)
        fragments = [
            VideoFragment(0, 40, _Hash(1), _Hash(2), tuple(f"A{i}" for i in range(5))),
            VideoFragment(40, 70, _Hash(3), _Hash(4), tuple(f"B{i}" for i in range(5))),
            VideoFragment(70, 200, _Hash(5), _Hash(6), tuple(f"C{i}" for i in range(5))),
        ]
        merged = merge_small_fragments(fragments, min_frames=100, sample_count=5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].sample_images), 5)

    def test_detect_fragments_collects_sample_images(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("opencv not installed")

        from externals.split_video.split import detect_fragments

        path = Path("sessions") / "_test_detect_fragment_images.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (16, 16),
        )
        try:
            for i in range(250):
                writer.write(np.full((16, 16, 3), i % 256, dtype=np.uint8))
        finally:
            writer.release()

        try:
            fragments, _fps = detect_fragments(
                path,
                threshold=0,
                hash_size=8,
                min_frames=50,
                sample_count=5,
            )
            self.assertGreaterEqual(len(fragments), 1)
            self.assertEqual(len(fragments[0].sample_images), 5)
        finally:
            path.unlink(missing_ok=True)


class TestSplitVideoParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$split_video(threshold=12, min_frames=80)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "split_video")
        self.assertEqual(expr.args.get("threshold"), "12")


class TestSplitVideoRun(unittest.TestCase):
    def test_emulate_outputs_labeled_fragments(self) -> None:
        os.environ["AH_EMULATE_SPLIT_VIDEO"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("split_video")
            )
            bundle = ArrayBundle()
            bundle.videos.append(
                ctx.new_link("videos", ".mp4", b"\x00\x00\x00\x18ftypisom\x00")
            )
            inp = ExternalInput(bundle=bundle, args={"threshold": "10"}, prompt_text="")
            out = run(ctx, inp)
            self.assertEqual(len(out.videos), 1)
            self.assertEqual(len(out.labels), 1)
            self.assertEqual(out.labels[0][0], "fragment")
            self.assertEqual(out.labels[0][2]["frame"], 0)
            self.assertIn("src", out.labels[0][2])
        finally:
            os.environ.pop("AH_EMULATE_SPLIT_VIDEO", None)


if __name__ == "__main__":
    unittest.main()

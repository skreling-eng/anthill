"""Tests for structured labels, $label, $add_label, and zip(label=...)."""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, ZipAction, parse_actions
from ahlib.ah_runtime import ArrayBundle
from ahlib.label_utils import (
    add_label_for_elements,
    exclude_by_label_name,
    filter_by_label_name,
    normalize_label_entry,
    propagate_labels,
)

from tests.test_prompt_merge import _run


def _png_bytes() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return b"\x89PNG\r\n\x1a\n"


class TestLabelUtils(unittest.TestCase):
    def test_normalize_label_entry(self) -> None:
        entry = ["good", [("images", "sessions/x/1.png"), ("texts", "sessions/x/t.txt")]]
        parsed = normalize_label_entry(entry)
        self.assertEqual(
            parsed,
            ("good", [("images", "sessions/x/1.png"), ("texts", "sessions/x/t.txt")]),
        )

    def test_add_label_for_elements(self) -> None:
        bundle = ArrayBundle(
            images=["a.png", "b.png"],
            texts=["t.txt"],
            labels=[["old", [("images", "x.png")]]],
        )
        out = add_label_for_elements(bundle, "tag")
        self.assertEqual(len(out.labels), 4)
        self.assertEqual(out.labels[0], ["old", [("images", "x.png")]])
        self.assertEqual(out.labels[1], ["tag", [("texts", "t.txt")]])
        self.assertEqual(out.labels[2], ["tag", [("images", "a.png")]])
        self.assertEqual(out.labels[3], ["tag", [("images", "b.png")]])

    def test_filter_by_label_name(self) -> None:
        bundle = ArrayBundle(
            images=["keep.png", "drop.png"],
            labels=[
                ["good", [("images", "keep.png")]],
                ["bad", [("images", "drop.png")]],
                ["good", [("texts", "note.txt")]],
            ],
        )
        out = filter_by_label_name(bundle, "good")
        self.assertEqual(out.images, ["keep.png"])
        self.assertEqual(out.texts, ["note.txt"])
        self.assertEqual(len(out.labels), 2)

    def test_exclude_by_label_name(self) -> None:
        bundle = ArrayBundle(
            images=["a.png", "b.png", "c.png"],
            labels=[
                ["good", [("images", "a.png")]],
                ["bad", [("images", "b.png")]],
            ],
        )
        out = exclude_by_label_name(bundle, "good")
        self.assertEqual(out.images, ["b.png", "c.png"])
        self.assertEqual(len(out.labels), 1)
        self.assertEqual(out.labels[0][0], "bad")

    def test_exclude_includes_unlabeled_and_other_labels(self) -> None:
        bundle = ArrayBundle(
            images=["orig.png", "plain.png"],
            labels=[
                ["original", [("images", "orig.png")]],
                ["flipped", [("images", "flip.png")]],
            ],
        )
        out = exclude_by_label_name(bundle, "original")
        self.assertEqual(out.images, ["plain.png", "flip.png"])
        self.assertEqual(len(out.labels), 1)
        self.assertEqual(out.labels[0][0], "flipped")

    def test_propagate_labels_keeps_matching_links(self) -> None:
        input_bundle = ArrayBundle(
            images=["keep.png", "drop.png"],
            labels=[
                ["original", [("images", "keep.png")]],
                ["tag", [("images", "drop.png")]],
            ],
        )
        output = ArrayBundle(images=["keep.png", "new.png"])
        out = propagate_labels(input_bundle, output)
        self.assertEqual(len(out.labels), 1)
        self.assertEqual(out.labels[0], ["original", [("images", "keep.png")]])

    def test_propagate_labels_skips_missing_links(self) -> None:
        input_bundle = ArrayBundle(
            images=["gone.png"],
            labels=[["original", [("images", "gone.png")]]],
        )
        output = ArrayBundle(images=["other.png"])
        out = propagate_labels(input_bundle, output)
        self.assertEqual(out.labels, [])


class TestLabelParse(unittest.TestCase):
    def test_parse_label_not(self) -> None:
        expr = parse_actions("$label(not 'good')")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "label")
        self.assertEqual(expr.args, {"not": "good"})

    def test_parse_zip_label_block(self) -> None:
        expr = parse_actions("zip(label='good'){ $pass }")
        self.assertIsInstance(expr, ZipAction)
        self.assertEqual(expr.label_name, "good")
        self.assertEqual(expr.array_keys, [])


class TestLabelRuntime(unittest.TestCase):
    def test_add_label_and_label_filter(self) -> None:
        source = """
@img1: $file('test_label_a.png')
@img2: $file('test_label_b.png')
@pair: (@img1, @img2)
@tagged: @pair -> $add_label('good') -> $add_label('bad')
@run: @tagged -> $label('good')
"""
        repo = Path(__file__).resolve().parents[1]
        (repo / "test_label_a.png").write_bytes(_png_bytes())
        (repo / "test_label_b.png").write_bytes(_png_bytes())
        os.environ["AH_EXTERNAL_INPROCESS"] = "file,add_label,label"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            result, _session = _run(source, "run")
            self.assertEqual(len(result.images), 2)
            for entry in result.labels:
                self.assertEqual(entry[0], "good")
        finally:
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)
            (repo / "test_label_a.png").unlink(missing_ok=True)
            (repo / "test_label_b.png").unlink(missing_ok=True)

    def test_label_not_includes_unlabeled_elements(self) -> None:
        source = """
@img1: $file('test_label_unlabeled_a.png')
@img2: $file('test_label_unlabeled_b.png')
@img3: $file('test_label_unlabeled_c.png')
@all: (@img1, @img2, @img3)
@tagged: @img1 -> $add_label('original')
@run: (@tagged, @img2, @img3) -> $label(not 'original')
"""
        repo = Path(__file__).resolve().parents[1]
        for name in (
            "test_label_unlabeled_a.png",
            "test_label_unlabeled_b.png",
            "test_label_unlabeled_c.png",
        ):
            (repo / name).write_bytes(_png_bytes())
        os.environ["AH_EXTERNAL_INPROCESS"] = "file,add_label,label"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            result, _session = _run(source, "run")
            self.assertEqual(len(result.images), 2)
            for entry in result.labels:
                self.assertNotEqual(entry[0], "original")
        finally:
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)
            for name in (
                "test_label_unlabeled_a.png",
                "test_label_unlabeled_b.png",
                "test_label_unlabeled_c.png",
            ):
                (repo / name).unlink(missing_ok=True)

    def test_label_not_excludes_named_elements(self) -> None:
        source = """
@img1: $file('test_label_not_a.png')
@img2: $file('test_label_not_b.png')
@pair: (@img1, @img2)
@run: @pair -> $add_label('good') -> $add_label('bad') -> $label(not 'good')
"""
        repo = Path(__file__).resolve().parents[1]
        (repo / "test_label_not_a.png").write_bytes(_png_bytes())
        (repo / "test_label_not_b.png").write_bytes(_png_bytes())
        os.environ["AH_EXTERNAL_INPROCESS"] = "file,add_label,label"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            result, _session = _run(source, "run")
            self.assertEqual(result.images, [])
            self.assertEqual(len(result.labels), 2)
            for entry in result.labels:
                self.assertEqual(entry[0], "bad")
        finally:
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)
            (repo / "test_label_not_a.png").unlink(missing_ok=True)
            (repo / "test_label_not_b.png").unlink(missing_ok=True)

    def test_label_filter_keeps_only_named_elements(self) -> None:
        source = """
@img1: $file('test_label_only_a.png')
@img2: $file('test_label_only_b.png')
@pair: (@img1, @img2)
@run: @pair -> $add_label('good') -> $add_label('bad') -> $label('bad')
"""
        repo = Path(__file__).resolve().parents[1]
        (repo / "test_label_only_a.png").write_bytes(_png_bytes())
        (repo / "test_label_only_b.png").write_bytes(_png_bytes())
        os.environ["AH_EXTERNAL_INPROCESS"] = "file,add_label,label"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            result, _session = _run(source, "run")
            self.assertEqual(len(result.images), 2)
            for entry in result.labels:
                self.assertEqual(entry[0], "bad")
        finally:
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)
            (repo / "test_label_only_a.png").unlink(missing_ok=True)
            (repo / "test_label_only_b.png").unlink(missing_ok=True)

    def test_zip_label_runs_body_per_entry(self) -> None:
        source = """
@img1: $file('test_zip_label_a.png')
@img2: $file('test_zip_label_b.png')
@pair: (@img1, @img2)
@tagged: @pair -> $add_label('item')
@run: @tagged -> zip(label='item'){ $pass }
"""
        repo = Path(__file__).resolve().parents[1]
        (repo / "test_zip_label_a.png").write_bytes(_png_bytes())
        (repo / "test_zip_label_b.png").write_bytes(_png_bytes())
        os.environ["AH_EXTERNAL_INPROCESS"] = "file,add_label,pass"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            result, _session = _run(source, "run")
            self.assertEqual(len(result.images), 2)
        finally:
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)
            (repo / "test_zip_label_a.png").unlink(missing_ok=True)
            (repo / "test_zip_label_b.png").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

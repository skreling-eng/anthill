"""Tests for ahlib.ah_bundle_compact."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ahlib.ah_bundle_compact import bundle_compact_dict, bundle_compact_str
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestBundleCompact(unittest.TestCase):
    def setUp(self) -> None:
        self.session_dir = create_session_dir(Path("sessions"))
        self.session = Session(self.session_dir)
        self.op_dir = self.session.next_op_dir("compact")

    def test_omits_empty_arrays_and_inlines_text(self) -> None:
        bundle = ArrayBundle()
        bundle.prompts.append(
            self.session.new_link(self.op_dir, "prompts", ".txt", "line one\n")
        )
        bundle.texts.append(
            self.session.new_link(self.op_dir, "texts", ".txt", "meta\n")
        )
        bundle.images.append(
            self.session.new_link(self.op_dir, "images", ".png", b"\x89PNG")
        )

        compact = bundle_compact_dict(bundle, self.session_dir)
        self.assertEqual(list(compact.keys()), ["images", "prompts", "texts"])
        self.assertEqual(compact["prompts"], ["line one"])
        self.assertEqual(compact["texts"], ["meta"])
        self.assertEqual(len(compact["images"]), 1)

        parsed = json.loads(bundle_compact_str(bundle, self.session_dir))
        self.assertEqual(parsed, compact)

    def test_empty_bundle_is_empty_object(self) -> None:
        self.assertEqual(bundle_compact_str(ArrayBundle(), self.session_dir), "{}")


if __name__ == "__main__":
    unittest.main()

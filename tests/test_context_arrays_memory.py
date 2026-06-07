from __future__ import annotations

import json
import unittest
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Runtime, Session, create_session_dir


class TestContextArraysMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.session_dir = create_session_dir(Path("sessions"))
        self.session = Session(self.session_dir)
        self.op_dir = self.session.next_op_dir("ctx_mem")

    def test_write_bundle_keeps_embeddings_labels_in_manifest_only(self) -> None:
        bundle = ArrayBundle(
            embeddings=[[0.1, 0.2], {"vec": [1, 2, 3]}],
            labels=[["cat", [("images", "x.png")]], ["style", [("texts", "y.txt")]]],
        )
        self.session.write_bundle(self.op_dir, bundle, "output")

        manifest = json.loads((self.op_dir / "output.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["labels"],
            [["cat", [["images", "x.png"]]], ["style", [["texts", "y.txt"]]]],
        )
        self.assertEqual(manifest["embeddings"], [[0.1, 0.2], {"vec": [1, 2, 3]}])
        self.assertFalse((self.op_dir / "labels").exists())
        self.assertFalse((self.op_dir / "embeddings").exists())

    def test_input_key_accepts_nested_embedding_structures(self) -> None:
        bundle = ArrayBundle(
            embeddings=[[0.1, 0.2], {"name": "e1", "vec": [1, 2, 3]}],
            labels=["a", "b"],
        )
        key = Runtime._input_key(bundle)
        # Ensure key is hashable for instruction cache usage.
        _ = hash(key)


if __name__ == "__main__":
    unittest.main()


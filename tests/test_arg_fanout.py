"""Runtime fan-out: key=@ref list args ≡ parallel $ branches (Cartesian product)."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Runtime, Session, create_session_dir
from ahlib.ah_parser import parse_ah_source


def _run(source: str, target: str, *, inprocess: str) -> tuple[ArrayBundle, Path]:
    os.environ["AH_EXTERNAL_INPROCESS"] = inprocess
    os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
    program = parse_ah_source(source)
    session_dir = create_session_dir(Path("sessions"))
    result = Runtime(program, Session(session_dir)).run(target)
    return result, session_dir


def _pass_invokes(session_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(session_dir.rglob("invoke.json")):
        if "__pass" not in path.parent.name.replace("\\", "/"):
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _read_texts(session_dir: Path, bundle: ArrayBundle) -> list[str]:
    out: list[str] = []
    for link in bundle.texts:
        path = Path(link)
        if not path.is_absolute():
            path = session_dir / link
        out.append(path.read_text(encoding="utf-8"))
    return out


class TestArgFanoutVariants(unittest.TestCase):
    def test_cartesian_product(self) -> None:
        variants = Runtime._external_arg_variants(
            {"model": ["a", "b"], "tag": ["x", "y"]}
        )
        self.assertEqual(
            variants,
            [
                {"model": "a", "tag": "x"},
                {"model": "a", "tag": "y"},
                {"model": "b", "tag": "x"},
                {"model": "b", "tag": "y"},
            ],
        )

    def test_empty_lists_returns_empty(self) -> None:
        self.assertEqual(Runtime._external_arg_variants({}), [])


class TestArgFanoutRuntime(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EMULATE_IMAGE2TEXT"] = "1"
        os.environ["AH_EMULATE_FILE"] = "1"
        os.environ["AH_EMULATE_IMAGE"] = "1"

    def tearDown(self) -> None:
        for key in (
            "AH_EMULATE_IMAGE2TEXT",
            "AH_EMULATE_FILE",
            "AH_EMULATE_IMAGE",
            "AH_EMULATE_IMAGE2VIDEO",
            "AH_EXTERNAL_INPROCESS",
            "AH_EXTERNAL_SUBPROCESS",
        ):
            os.environ.pop(key, None)

    def test_image_native_repeat_single_invoke(self) -> None:
        source = """
@run: $image[5]
a cat
"""
        _, session_dir = _run(source, "run", inprocess="image,list")
        invokes = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(session_dir.rglob("invoke.json"))
            if "__image" in p.parent.name.replace("\\", "/")
        ]
        self.assertEqual(len(invokes), 1)
        self.assertEqual(invokes[0].get("repeat"), 5)

    def test_image2video_native_repeat_single_invoke(self) -> None:
        os.environ["AH_EMULATE_IMAGE2VIDEO"] = "1"
        try:
            source = """
@img: $file('photo.png')

@run: @img -> $image2video[10]
Animate gently
"""
            out, session_dir = _run(source, "run", inprocess="file,image2video")
            invokes = [
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(session_dir.rglob("invoke.json"))
                if "__image2video" in p.parent.name.replace("\\", "/")
            ]
            self.assertEqual(len(invokes), 1)
            self.assertEqual(invokes[0].get("repeat"), 10)
            self.assertEqual(len(out.videos), 10)
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2VIDEO", None)

    def test_image2text_model_at_ref_matches_parallel(self) -> None:
        source = """
@models: $list
qwen2, qwen3

@img: $file('photo.png')

@via_list: @img -> $image2text(model=@models)
Describe the image

@via_parallel: @img -> (
$image2text(model='qwen2'),
$image2text(model='qwen3')
)
Describe the image
"""
        via_list, session_list = _run(
            source, "via_list", inprocess="file,image2text,list"
        )
        via_parallel, session_parallel = _run(
            source, "via_parallel", inprocess="file,image2text,list"
        )

        self.assertEqual(len(via_list.texts), 2)
        self.assertEqual(len(via_parallel.texts), 2)
        list_texts = _read_texts(session_list, via_list)
        parallel_texts = _read_texts(session_parallel, via_parallel)
        self.assertIn("model=qwen2", list_texts[0])
        self.assertIn("model=qwen3", list_texts[1])
        self.assertIn("model=qwen2", parallel_texts[0])
        self.assertIn("model=qwen3", parallel_texts[1])

    def test_cartesian_product_via_pass(self) -> None:
        source = """
@items: $list
a, b

@tags: $list
x, y

@run: $pass(model=@items, tag=@tags, mode='fast')
"""
        _, session_dir = _run(source, "run", inprocess="pass,list")
        invokes = _pass_invokes(session_dir)
        self.assertEqual(len(invokes), 4)
        combos = sorted(
            (
                inv["args"]["model"],
                inv["args"]["tag"],
                inv["args"]["mode"],
            )
            for inv in invokes
        )
        self.assertEqual(
            combos,
            [
                ("a", "x", "fast"),
                ("a", "y", "fast"),
                ("b", "x", "fast"),
                ("b", "y", "fast"),
            ],
        )
        for inv in invokes:
            self.assertEqual(inv.get("arg_lists") or {}, {})
            self.assertEqual(inv.get("repeat"), 1)

    def test_repeat_applies_to_each_branch(self) -> None:
        source = """
@items: $list
a, b

@run: $pass(model=@items)[3]
"""
        _, session_dir = _run(source, "run", inprocess="pass,list")
        invokes = _pass_invokes(session_dir)
        self.assertEqual(len(invokes), 6)
        self.assertTrue(all(inv.get("repeat") == 1 for inv in invokes))

    def test_image2text_repeat_five_outputs(self) -> None:
        source = """
@img: $file('photo.png')

@run: @img -> $image2text(model='qwen2')[5]
Count fingers
"""
        out, _ = _run(source, "run", inprocess="file,image2text,list")
        self.assertEqual(len(out.texts), 5)

    def test_single_ref_value_collapses_to_scalar_arg(self) -> None:
        source = """
@items: $list
only_one

@run: $pass(model=@items)
"""
        _, session_dir = _run(source, "run", inprocess="pass,list")
        invokes = _pass_invokes(session_dir)
        self.assertEqual(len(invokes), 1)
        self.assertEqual(invokes[0]["args"]["model"], "only_one")
        self.assertEqual(invokes[0].get("arg_lists") or {}, {})


if __name__ == "__main__":
    unittest.main()

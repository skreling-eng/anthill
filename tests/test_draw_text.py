"""Tests for $draw_text external."""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.draw_text.run import add_text, run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


def _png_bytes(width: int = 64, height: int = 64, color: tuple[int, int, int] = (40, 80, 120)) -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (width, height), color).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00@\x00\x00\x00@"
            b"\x08\x02\x00\x00\x00%\x0b\xe6\x89\x00\x00\x00\x0cIDATx\x9cc``\x00"
            b"\x00\x00\x04\x00\x01\x5c\xcd\xff\x69\x00\x00\x00\x00IEND\xaeB`\x82"
        )


class TestDrawTextExternal(unittest.TestCase):
    def _ctx_and_input(
        self,
        *,
        images: list[str] | None = None,
        prompts: list[str] | None = None,
        args: dict[str, str] | None = None,
    ) -> tuple[ExternalContext, ExternalInput]:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("draw_text")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle()
        if images:
            bundle.images = list(images)
        if prompts:
            for i, text in enumerate(prompts):
                bundle.prompts.append(
                    session.new_link(op_dir, "prompts", ".txt", text + "\n")
                )
        inp = ExternalInput(
            bundle=bundle,
            args=args or {},
            prompt_text="\n".join(prompts or []),
        )
        return ctx, inp

    def test_emulate_replaces_images(self) -> None:
        os.environ["AH_EMULATE_DRAW_TEXT"] = "1"
        try:
            ctx, inp = self._ctx_and_input(prompts=["Hello^World"])
            img_link = ctx.new_link("images", ".png", _png_bytes())
            inp.bundle.images = [img_link]
            out = run(ctx, inp)
            self.assertEqual(len(out.images), 1)
            self.assertNotEqual(out.images[0], img_link)
            data = (ctx.base_dir / out.images[0]).read_bytes()
            self.assertIn(b"[emulated $draw_text]", data)
        finally:
            os.environ.pop("AH_EMULATE_DRAW_TEXT", None)

    def test_caret_becomes_newline_for_long_text(self) -> None:
        from externals.draw_text import run as draw_mod

        self.assertEqual(
            draw_mod._prepare_text("abcdefghij^k"),
            "abcdefghij\nk",
        )
        self.assertEqual(draw_mod._prepare_text("short^x"), "short^x")

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("PIL") is not None,
        "Pillow not installed",
    )
    def test_draw_text_writes_output_image(self) -> None:
        os.environ.pop("AH_EMULATE_DRAW_TEXT", None)
        ctx, inp = self._ctx_and_input(prompts=["Overlay text"])
        img_link = ctx.new_link("images", ".png", _png_bytes(120, 120))
        inp.bundle.images = [img_link]
        out = run(ctx, inp)
        self.assertEqual(len(out.images), 1)
        out_path = ctx.base_dir / out.images[0]
        self.assertTrue(out_path.is_file())
        from PIL import Image

        with Image.open(out_path) as img:
            self.assertGreater(img.size[0], 0)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("PIL") is not None,
        "Pillow not installed",
    )
    def test_add_text_function(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        src = session_dir / "in.png"
        dst = session_dir / "out.png"
        src.write_bytes(_png_bytes(200, 200))
        add_text(
            src,
            dst,
            "Test^Line",
            font_path=None,
            font_size=20,
            text_left=10,
            text_top=10,
            spacing=4,
            stroke_width=1,
        )
        self.assertTrue(dst.is_file())


if __name__ == "__main__":
    unittest.main()

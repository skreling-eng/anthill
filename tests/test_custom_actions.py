"""Tests for & custom actions (parse, cache, emulate execution)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ahlib.ah_actions import CustomActionExpr, parse_actions
from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import ArrayBundle, Runtime, Session
from ahlib.custom_action_codegen import (
    CUSTOM_CODE_LOG_NAME,
    _FailedAttempt,
    _build_generation_prompt,
    apply_run_py_fixups,
    check_generated_code,
    generate_and_store,
    prompt_hash,
    rejected_attempt_log_name,
    smoke_execution_issues,
    write_custom_code_log,
    write_rejected_attempt_log,
)
from ahlib.custom_action_env import (
    modules_to_pypi_packages,
    parse_third_party_imports,
)


class TestStaticCodegenChecks(unittest.TestCase):
    def test_rejects_double_uuid_append(self) -> None:
        from ahlib.custom_action_codegen import _static_codegen_issues

        bad = (
            "dest = root / f'sounds/x_{uuid.uuid4()}.wav'\n"
            "wavfile.write(dest, sr, data)\n"
            "new_sounds.append(f'sounds/x_{uuid.uuid4()}.wav')\n"
        )
        reason = _static_codegen_issues(bad)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("uuid.uuid4()", reason)


class TestSmokeExecution(unittest.TestCase):
    _BAD_TEXT2PROMPTS = """
def run(bundle, base_dir, op_dir):
    out = {k: list(bundle.get(k, [])) for k in
           ("prompts", "texts", "images", "sounds", "videos", "files", "changes")}
    new_prompts = []
    for text_link in bundle.get("texts", []):
        src = Path(base_dir) / text_link
        with open(src, encoding="utf-8") as file:
            new_prompts.extend(file.read().split("\\n\\n"))
    out["prompts"] = new_prompts
    return out
"""

    _GOOD_SHUFFLE = """
import random
from pathlib import Path

def run(bundle, base_dir, op_dir):
    out = {k: list(bundle.get(k, [])) for k in
           ("prompts", "texts", "images", "sounds", "videos", "files", "changes")}
    random.shuffle(out["texts"])
    return out
"""

    def test_rejects_missing_path_import(self) -> None:
        reason = smoke_execution_issues(self._BAD_TEXT2PROMPTS)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("NameError", reason)

    def test_rejects_inline_prompt_text(self) -> None:
        code = """
from pathlib import Path

def run(bundle, base_dir, op_dir):
    out = {k: list(bundle.get(k, [])) for k in
           ("prompts", "texts", "images", "sounds", "videos", "files", "changes")}
    new_prompts = []
    for text_link in bundle.get("texts", []):
        src = Path(base_dir) / text_link
        new_prompts.extend(src.read_text(encoding="utf-8").split("\\n\\n"))
    out["prompts"] = new_prompts
    return out
"""
        reason = smoke_execution_issues(code)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("inline text", reason)

    def test_text2prompts_writes_prompt_links(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        code = (repo / "custom_actions/text2prompts/run.py").read_text(encoding="utf-8")
        self.assertIsNone(smoke_execution_issues(code))

    def test_accepts_valid_handler(self) -> None:
        self.assertIsNone(smoke_execution_issues(self._GOOD_SHUFFLE))

    def test_rejects_non_dict_return(self) -> None:
        code = "def run(bundle, base_dir, op_dir):\n    return []\n"
        reason = smoke_execution_issues(code)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("expected dict", reason)

    def test_check_generated_code_runs_smoke_in_emulate_mode(self) -> None:
        os.environ["AH_EMULATE_CODE"] = "1"
        try:
            ok, reason, _ = check_generated_code(
                self._BAD_TEXT2PROMPTS,
                "split texts by blank lines into prompts",
            )
            self.assertFalse(ok)
            self.assertIn("NameError", reason)
        finally:
            os.environ.pop("AH_EMULATE_CODE", None)

    def test_checked_in_handlers_pass_smoke(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        for run_py in sorted((repo / "custom_actions").glob("*/run.py")):
            with self.subTest(handler=run_py.parent.name):
                reason = smoke_execution_issues(
                    run_py.read_text(encoding="utf-8")
                )
                if reason and reason.startswith("smoke load failed: ImportError"):
                    self.skipTest(reason)
                self.assertIsNone(reason, msg=f"{run_py}: {reason}")


class TestCodegenRetries(unittest.TestCase):
    def test_prompt_includes_prior_failures(self) -> None:
        prompt = _build_generation_prompt(
            "fff",
            "make sounds louder",
            [
                _FailedAttempt(
                    attempt=1,
                    code="def run(b, d):\n    return b\n",
                    reason="copies only",
                )
            ],
        )
        self.assertIn("PREVIOUS FAILED ATTEMPTS", prompt)
        self.assertIn("REJECTION REASON: copies only", prompt)
        self.assertIn("def run(b, d)", prompt)

    def test_emulate_check_accepts_save_image_links(self) -> None:
        os.environ["AH_EMULATE_CODE"] = "1"
        try:
            code = Path(__file__).resolve().parents[1].joinpath(
                "custom_actions/double_image/run.py"
            ).read_text(encoding="utf-8")
            ok, reason, _ = check_generated_code(
                code,
                "for images, make these changes:\n"
                "- return ab original image\n"
                "- return a flipped horizontally image",
            )
            self.assertTrue(ok, reason)
        finally:
            os.environ.pop("AH_EMULATE_CODE", None)

    def test_emulate_check_rejects_subprocess(self) -> None:
        os.environ["AH_EMULATE_CODE"] = "1"
        try:
            ok, reason, _ = check_generated_code(
                "import subprocess\ndef run(b, d):\n    return b\n",
                "make sounds louder",
            )
            self.assertFalse(ok)
            self.assertIn("subprocess", reason)
        finally:
            os.environ.pop("AH_EMULATE_CODE", None)


class TestCustomCodeLog(unittest.TestCase):
    def test_each_rejected_attempt_gets_own_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            op = Path(tmp) / "op"
            op.mkdir()
            write_rejected_attempt_log(
                op,
                1,
                5,
                "code1",
                name="loud",
                spec="make sounds louder",
                reason="copies only",
            )
            write_rejected_attempt_log(
                op,
                2,
                5,
                "code2",
                name="loud",
                spec="make sounds louder",
                reason="still no gain",
                validator_response='{"ok": false}',
            )
            t1 = (op / rejected_attempt_log_name(1)).read_text(encoding="utf-8")
            t2 = (op / rejected_attempt_log_name(2)).read_text(encoding="utf-8")
            self.assertTrue(t1.startswith("REJECTION REASON: copies only"))
            self.assertTrue(t2.startswith("REJECTION REASON: still no gain"))
            self.assertIn("code1", t1)
            self.assertIn("code2", t2)
            self.assertFalse((op / CUSTOM_CODE_LOG_NAME).exists())

    def test_writes_custome_code_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            op = Path(tmp) / "op"
            op.mkdir()
            write_custom_code_log(
                op,
                "def run(bundle, base_dir):\n    return bundle\n",
                name="fff",
                spec="crop images",
            )
            log = op / CUSTOM_CODE_LOG_NAME
            self.assertTrue(log.is_file())
            text = log.read_text(encoding="utf-8")
            self.assertIn("&fff", text)
            self.assertIn("crop images", text)
            self.assertIn("def run", text)


class TestCustomActionIo(unittest.TestCase):
    def test_save_image_returns_link_under_op_dir(self) -> None:
        from PIL import Image

        from ahlib.custom_action_io import save_image

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "sess"
            op = base / "1__crop"
            op.mkdir(parents=True)
            im = Image.new("RGB", (10, 20))
            link = save_image(base, op, "out.png", im)
            self.assertEqual(link, "1__crop/images/out.png")
            self.assertTrue((base / link).is_file())

    def test_apply_db_gain_handles_float_pcm(self) -> None:
        import numpy as np

        from ahlib.custom_action_io import apply_db_gain, float_to_int16

        data = np.array([0.5, -0.25], dtype=np.float32)
        louder = float_to_int16(apply_db_gain(data, 10))
        self.assertGreater(int(np.max(np.abs(louder))), 1000)

    def test_save_wav_returns_link_under_op_dir(self) -> None:
        import numpy as np

        from ahlib.custom_action_io import save_wav

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "sess"
            op = base / "1__loud"
            op.mkdir(parents=True)
            data = np.zeros(100, dtype=np.int16)
            link = save_wav(base, op, "out.wav", 44100, data)
            self.assertEqual(link, "1__loud/sounds/out.wav")
            self.assertTrue((base / link).is_file())


class TestSaveMkdirFixup(unittest.TestCase):
    def test_inserts_mkdir_before_save(self) -> None:
        code = (
            "        rel = f'custom/x.png'\n"
            "        im.save(root / rel)\n"
        )
        fixed = apply_run_py_fixups(code)
        self.assertIn("mkdir", fixed)
        self.assertIn("dest = root / rel", fixed)
        self.assertIn("im.save(dest)", fixed)


class TestImportParsing(unittest.TestCase):
    def test_parse_pil_and_pathlib(self) -> None:
        code = "from pathlib import Path\nfrom PIL import Image\n"
        mods = parse_third_party_imports(code)
        self.assertEqual(mods, ["PIL"])
        self.assertEqual(modules_to_pypi_packages(mods), ["Pillow"])

    def test_parse_import_probes(self) -> None:
        from ahlib.custom_action_env import parse_import_probes

        code = "import moviepy.editor as mp\nfrom PIL import Image\n"
        probes = parse_import_probes(code)
        self.assertIn("import moviepy.editor", probes)
        self.assertIn("from PIL import Image", probes)

    def test_parse_import_probes_skips_except_fallbacks(self) -> None:
        from ahlib.custom_action_env import parse_import_probes

        code = (
            "try:\n"
            "    from moviepy import VideoFileClip\n"
            "except ImportError:\n"
            "    import moviepy.editor as mp\n"
        )
        probes = parse_import_probes(code)
        self.assertEqual(probes, ["from moviepy import VideoFileClip"])

    def test_packages_needing_install_detects_missing(self) -> None:
        from ahlib.custom_action_env import packages_needing_install
        from unittest.mock import patch

        py = Path("fake/python.exe")
        with patch(
            "ahlib.custom_action_env._package_importable",
            side_effect=lambda _py, pkg: pkg == "numpy",
        ):
            missing = packages_needing_install(py, ["numpy", "moviepy"])
        self.assertEqual(missing, ["moviepy"])


class TestCustomActionParse(unittest.TestCase):
    SOURCE = """
&fff:
for images, delete bottom 100 pixels

@test: $clear -> &fff
body line

>>>
"""

    def test_parses_definition_and_pipeline(self) -> None:
        prog = parse_ah_source(self.SOURCE)
        self.assertIn("fff", prog.custom_actions)
        self.assertIn("bottom 100", prog.custom_actions["fff"].body)
        self.assertEqual(prog.run_target, "test")
        from ahlib.ah_actions import SequenceAction

        expr = parse_actions(prog.instructions["test"].actions)
        self.assertIsInstance(expr, SequenceAction)
        self.assertIsInstance(expr.steps[-1], CustomActionExpr)
        self.assertEqual(expr.steps[-1].name, "fff")


class TestCustomActionCache(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EMULATE_CODE"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("AH_EMULATE_CODE", None)

    def test_skips_regen_when_prompt_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = "for images, delete bottom 100 pixels"
            p1 = generate_and_store("fff", spec, root)
            mtime = p1.stat().st_mtime
            p2 = generate_and_store("fff", spec, root)
            self.assertEqual(p1, p2)
            self.assertEqual(p2.stat().st_mtime, mtime)
            meta = json.loads((p1.parent / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["prompt_hash"], prompt_hash(spec))


class TestCustomActionEmulateRun(unittest.TestCase):
    SOURCE = """
&crop:
for images, delete bottom 100 pixels

@go: &crop

run @go
"""

    def setUp(self) -> None:
        os.environ["AH_EMULATE_CODE"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("AH_EMULATE_CODE", None)

    def test_crop_images_in_session(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_dir = root / "sessions" / "test_sess"
            session_dir.mkdir(parents=True)
            img_path = session_dir / "in.png"
            Image.new("RGB", (40, 120), color=(255, 0, 0)).save(img_path)

            program = parse_ah_source(self.SOURCE)
            session = Session(session_dir)
            runtime = Runtime(program, session, repo_root=root)
            bundle = ArrayBundle(images=["in.png"])
            out = runtime._execute_instruction("go", bundle)

            self.assertEqual(len(out.images), 1)
            out_path = session_dir / out.images[0]
            self.assertTrue(out_path.is_file())
            with Image.open(out_path) as im:
                self.assertEqual(im.size[1], 20)

    def test_custom_action_propagates_labels_for_kept_links(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")

        source = """
@img: $file('test_double_label.png')

@go: @img -> $add_label('original') -> &double_image

&double_image:
for images, return original and flipped horizontally image

run @go
"""
        repo = Path(__file__).resolve().parents[1]
        img_path = repo / "test_double_label.png"
        buf = __import__("io").BytesIO()
        Image.new("RGB", (20, 20), color=(0, 128, 255)).save(buf, format="PNG")
        img_path.write_bytes(buf.getvalue())

        os.environ["AH_EXTERNAL_INPROCESS"] = "file,add_label"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            from tests.test_prompt_merge import _run

            result, _session = _run(source, "go")
            self.assertGreaterEqual(len(result.images), 2)
            original_entries = [
                entry for entry in result.labels if entry[0] == "original"
            ]
            self.assertTrue(original_entries, result.labels)
            labeled_paths = {path for _k, path in original_entries[0][1]}
            self.assertTrue(labeled_paths.intersection(result.images))
        finally:
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)
            os.environ.pop("AH_EXTERNAL_SUBPROCESS", None)
            img_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

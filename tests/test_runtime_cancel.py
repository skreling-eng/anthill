"""Tests for cooperative runtime cancellation."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import Runtime, RuntimeCancelled, Session, create_session_dir


class TestRuntimeCancel(unittest.TestCase):
    def test_cancel_before_run_raises(self) -> None:
        source = """
@a:
  x

@run: @a -> @a -> @a

run @run
"""
        cancel = threading.Event()
        cancel.set()
        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        runtime = Runtime(
            program, Session(session_dir), cancel_event=cancel
        )
        with self.assertRaises(RuntimeCancelled):
            runtime.run()

    def test_cancel_between_steps(self) -> None:
        source = """
@a:
  step

@run: @a -> @a -> @a -> @a

run @run
"""
        cancel = threading.Event()

        class _CancelAfterTwo:
            def __init__(self) -> None:
                self._finished = 0

            def action_start(self, action_name: str) -> None:
                pass

            def action_finish(
                self,
                action_name: str,
                output_context: dict,
                output_json_path: str | None = None,
                session_base_dir: str | None = None,
            ) -> None:
                self._finished += 1
                if self._finished >= 2:
                    cancel.set()

            def action_error(self, action_name: str, error_message: str) -> None:
                pass

        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        runtime = Runtime(
            program,
            Session(session_dir),
            callback=_CancelAfterTwo(),
            cancel_event=cancel,
        )
        with self.assertRaises(RuntimeCancelled):
            runtime.run()

    def test_ref_infinite_repeat_until_cancelled(self) -> None:
        source = """
@tick:
  n

@loop: @tick[inf]

run @loop
"""
        cancel = threading.Event()
        timer = threading.Timer(0.05, cancel.set)
        timer.start()
        try:
            program = parse_ah_source(source)
            session_dir = create_session_dir(Path("sessions"))
            runtime = Runtime(
                program,
                Session(session_dir),
                cancel_event=cancel,
            )
            with self.assertRaises(RuntimeCancelled):
                runtime.run()
        finally:
            timer.cancel()

    def test_ref_infinite_repeat_finishes_quickly(self) -> None:
        source = """
@tick:
  n

@loop: @tick[inf]

run @loop
"""
        cancel = threading.Event()
        timer = threading.Timer(0.05, cancel.set)
        timer.start()
        started = time.monotonic()
        try:
            program = parse_ah_source(source)
            session_dir = create_session_dir(Path("sessions"))
            runtime = Runtime(
                program,
                Session(session_dir),
                cancel_event=cancel,
            )
            with self.assertRaises(RuntimeCancelled):
                runtime.run()
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 2.0)

    def test_ref_infinite_repeat_rate_limited(self) -> None:
        source = """
@tick:
  n

@loop: @tick[inf]

run @loop
"""
        cancel = threading.Event()
        timer = threading.Timer(3.0, cancel.set)
        timer.start()
        try:
            program = parse_ah_source(source)
            session_dir = create_session_dir(Path("sessions"))
            runtime = Runtime(
                program,
                Session(session_dir),
                cancel_event=cancel,
            )
            with self.assertRaises(RuntimeCancelled):
                runtime.run()
        finally:
            timer.cancel()

        tick_dirs = [
            p
            for p in session_dir.iterdir()
            if p.is_dir() and p.name.endswith("_tick")
        ]
        self.assertEqual(len(tick_dirs), 2)


if __name__ == "__main__":
    unittest.main()

"""Action log updates must be polled from the WebView GUI thread."""

from __future__ import annotations

import time
import unittest
from pathlib import Path

from app import Interface, LinkApi


class TestLogGuiThread(unittest.TestCase):
    def test_add_action_results_queues_poll_instead_of_direct_draw(self) -> None:
        api = LinkApi(Path(".").resolve())
        ui = Interface(api)
        api.set_callback_obj(ui)
        ui.add_action_results("<b>START</b> test")
        self.assertTrue(ui._log_draw_pending)
        self.assertEqual(ui._log_generation, 1)
        html = api.poll_log_refresh()
        self.assertIsNotNone(html)
        self.assertIn("START", html)
        self.assertFalse(ui._log_draw_pending)
        self.assertEqual(ui._log_painted_generation, 1)
        self.assertIsNone(api.poll_log_refresh())
        ui.add_action_results("<b>STEP</b> two")
        time.sleep(ui._log_paint_min_interval_s + 0.05)
        html2 = api.poll_log_refresh()
        self.assertIsNotNone(html2)
        self.assertIn("STEP", html2)

    def test_queue_log_draw_coalesces(self) -> None:
        api = LinkApi(Path(".").resolve())
        ui = Interface(api)
        ui._queue_log_draw()
        ui._queue_log_draw()
        self.assertTrue(ui._log_draw_pending)


if __name__ == "__main__":
    unittest.main()

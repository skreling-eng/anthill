"""Chat input enable/disable must be polled from the WebView GUI thread."""

from __future__ import annotations

import unittest
from pathlib import Path

from chat import ChatInterface, LinkApi


class TestChatInputPoll(unittest.TestCase):
    def test_user_input_queues_enable_for_poll(self) -> None:
        api = LinkApi(Path(".").resolve())
        ui = ChatInterface(api)
        api.set_callback_obj(ui)
        ui._request_chat_input_enabled(True)
        self.assertIs(api.poll_chat_input_state(), True)
        self.assertIsNone(api.poll_chat_input_state())

    def test_disable_after_submit(self) -> None:
        api = LinkApi(Path(".").resolve())
        ui = ChatInterface(api)
        api.set_callback_obj(ui)
        ui._request_chat_input_enabled(False)
        self.assertIs(api.poll_chat_input_state(), False)


if __name__ == "__main__":
    unittest.main()

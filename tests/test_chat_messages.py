"""Chat messages as JSON; compaction via @messages_summary."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from chat import (
    CHAT_MAX_MESSAGES,
    CHAT_MAX_TOKENS,
    ChatInterface,
    LinkApi,
)
from ahlib.ah_runtime import ArrayBundle, create_session_dir


class TestChatMessages(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EMULATE_LLM"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "clear,llm"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

    def tearDown(self) -> None:
        os.environ.pop("AH_EMULATE_LLM", None)

    def _ui_with_session(self) -> ChatInterface:
        api = LinkApi(Path(".").resolve())
        ui = ChatInterface(api)
        ui.session_dir = create_session_dir(Path("sessions"))
        return ui

    def test_store_message_json_user(self) -> None:
        ui = self._ui_with_session()
        link = ui._new_text_link("Hello from user\n")
        ui._action_store_message(ArrayBundle(texts=[link]))
        self.assertEqual(ui._messages[0]["role"], "user")
        self.assertEqual(ui._messages[0]["text"], "Hello from user")

    def test_answer_json_bot_with_media(self) -> None:
        ui = self._ui_with_session()
        bundle = ArrayBundle(
            texts=[ui._new_text_link("Here is the clip.\n")],
            images=["3__gen/out.png"],
            sounds=["3__gen/out.wav"],
            videos=["3__gen/out.mp4"],
        )
        with mock.patch.object(ui, "_append_chat_pane"):
            ui._action_answer(bundle)
        msg = ui._messages[0]
        self.assertEqual(msg["role"], "bot")
        self.assertEqual(msg["images"], ["3__gen/out.png"])

    def test_get_messages_returns_summaries_then_messages(self) -> None:
        ui = self._ui_with_session()
        ui._summaries = [{"role": "summary", "text": "Earlier chat", "message_count": 5}]
        ui._messages = [{"role": "user", "text": "latest"}]
        out = ui._action_get_messages()
        self.assertEqual(len(out.texts), 2)
        first = json.loads(ui._chat_read_link_text(out.texts[0]))
        second = json.loads(ui._chat_read_link_text(out.texts[1]))
        self.assertEqual(first["role"], "summary")
        self.assertEqual(second["role"], "user")

    def test_maybe_summarize_over_message_count(self) -> None:
        ui = self._ui_with_session()
        ui._messages = [
            {"role": "user" if i % 2 == 0 else "bot", "text": f"msg {i}"}
            for i in range(CHAT_MAX_MESSAGES + 5)
        ]
        ui._action_maybe_summarize(ArrayBundle())
        self.assertGreaterEqual(len(ui._summaries), 1)
        self.assertLess(len(ui._messages), CHAT_MAX_MESSAGES + 5)
        self.assertIn("summary", ui._summaries[0]["role"])
        self.assertGreater(ui._summaries[0].get("message_count", 0), 0)

    def test_maybe_summarize_over_token_limit(self) -> None:
        ui = self._ui_with_session()
        big = "x" * 90_000
        ui._messages = [{"role": "user", "text": big}]
        self.assertGreater(ui._estimate_message_tokens(ui._messages), CHAT_MAX_TOKENS)
        ui._action_maybe_summarize(ArrayBundle())
        self.assertGreaterEqual(len(ui._summaries), 1)

    def test_clear_resets_summaries_and_messages(self) -> None:
        ui = self._ui_with_session()
        ui._messages = [{"role": "user", "text": "x"}]
        ui._summaries = [{"role": "summary", "text": "y", "message_count": 1}]
        ui.act("clear")
        self.assertEqual(ui._messages, [])
        self.assertEqual(ui._summaries, [])
        out = ui._action_get_messages()
        self.assertEqual(out.texts, [])
        self.assertEqual(ui._answer_parts, [])


if __name__ == "__main__":
    unittest.main()

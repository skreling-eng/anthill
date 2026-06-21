"""Brain UI controller — runs agent in background, renders chat thread."""

from __future__ import annotations

import html
import json
import threading
from datetime import datetime

import webview

from brain.agent.orchestrator import AgentOrchestrator, AgentResult
from brain.config import BrainConfig, load_config
from brain.conversation.history import ConversationStore, HistoryTurn
from brain.ui.api import BrainApi


class BrainInterface:
    def __init__(self, api: BrainApi, config: BrainConfig | None = None):
        self.api = api
        self.config = config or load_config()
        self._agent_thread: threading.Thread | None = None
        self._orchestrator: AgentOrchestrator | None = None
        self._cancel_event = threading.Event()
        self._shutting_down = False
        self._lock = threading.Lock()
        self._output_generation = 0
        self._output_painted = -1
        self._busy = False
        self._pending_request = ""
        self._working_status = ""
        self._conversation = ConversationStore(self.config)
        self._conversation.load()

    def begin_shutdown(self) -> None:
        self._shutting_down = True
        self.stop_agent()

    def wait_for_agent(self, timeout: float = 15.0) -> None:
        thread = self._agent_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def act(self, action: str) -> None:
        if action == "analyze":
            self.submit_request(self.api.get_request_text())
        elif action == "stop":
            self.stop_agent()

    def submit_request(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            self._refresh_output()
            return
        if self._agent_thread is not None and self._agent_thread.is_alive():
            with self._lock:
                self._working_status = "Agent already running."
                self._output_generation += 1
            return

        self._cancel_event.clear()
        self._busy = True
        with self._lock:
            self._pending_request = text
            self._working_status = "Analyzing request…"
            self._output_generation += 1

        self._agent_thread = threading.Thread(
            target=self._run_worker,
            args=(text,),
            daemon=True,
        )
        self._agent_thread.start()

    def stop_agent(self) -> None:
        self._cancel_event.set()
        if self._orchestrator is not None:
            self._orchestrator.cancel()
        with self._lock:
            self._working_status = "Stop requested."
            self._output_generation += 1

    def clear_conversation(self) -> None:
        if self._busy:
            return
        self._conversation.clear()
        with self._lock:
            self._pending_request = ""
            self._working_status = ""
            self._output_generation += 1

    def _run_worker(self, text: str) -> None:
        def on_status(msg: str) -> None:
            with self._lock:
                self._working_status = msg
                self._output_generation += 1

        try:
            prior = self._conversation.recent_for_model(
                self.config.max_conversation_turns
            )
            self._orchestrator = AgentOrchestrator(
                self.config,
                on_status=on_status,
            )
            result = self._orchestrator.run(text, conversation=prior)
            self._record_history(result)
        except Exception as exc:
            turn = HistoryTurn(
                tm=str(datetime.now()),
                request=text,
                report="",
                error=str(exc),
            )
            self._conversation.append(turn)
            with self._lock:
                self._output_generation += 1
        finally:
            self._orchestrator = None
            self._busy = False
            self._pending_request = ""
            self._working_status = ""
            self._agent_thread = None
            with self._lock:
                self._output_generation += 1

    def _record_history(self, result: AgentResult) -> None:
        turn = ConversationStore.turn_from_result(result)
        self._conversation.append(turn)

    def save_conversation(self) -> None:
        self._conversation.save()

    def load_conversation(self) -> None:
        self._conversation.load()
        with self._lock:
            self._output_generation += 1

    def _refresh_output(self) -> None:
        with self._lock:
            self._output_generation += 1

    def _render_thread(self) -> str:
        parts: list[str] = ['<div class="chat-thread">']
        turns = self._conversation.turns()
        if not turns and not self._pending_request:
            parts.append(
                "<p class='chat-empty'>Ask about the codebase or request a change. "
                "Follow-up messages keep context from earlier turns.</p>"
            )
        for turn in turns:
            parts.append(self._user_bubble(turn.request))
            parts.append(self._assistant_bubble(turn))
        if self._pending_request:
            parts.append(self._user_bubble(self._pending_request))
            parts.append(self._working_bubble())
        parts.append("</div>")
        return "\n".join(parts)

    def _user_bubble(self, text: str) -> str:
        body = html.escape(text).replace("\n", "<br>")
        return f'<div class="chat-msg user"><div class="chat-label">You</div>{body}</div>'

    def _assistant_bubble(self, turn: HistoryTurn) -> str:
        meta: list[str] = []
        if turn.elapsed_s:
            meta.append(f"{turn.elapsed_s:.1f}s")
        if turn.plan_summary:
            meta.append(html.escape(turn.plan_summary))
        header = ""
        if meta:
            header = f"<div class='chat-meta'>{' · '.join(meta)}</div>"
        if turn.error:
            body = f"<p class='error'>{html.escape(turn.error)}</p>"
        else:
            body = self._markdown_to_html(turn.report or "")
        return (
            f'<div class="chat-msg assistant">'
            f'<div class="chat-label">Brain</div>{header}{body}</div>'
        )

    def _working_bubble(self) -> str:
        status = html.escape(self._working_status or "Working…")
        return (
            f'<div class="chat-msg assistant working">'
            f'<div class="chat-label">Brain</div>'
            f'<p class="status-line">{status}</p></div>'
        )

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        out = html.escape(text)
        import re

        out = re.sub(
            r"```diff\n([\s\S]*?)```",
            lambda m: f"<pre class='diff-block'>{m.group(1)}</pre>",
            out,
        )
        out = re.sub(
            r"```([\s\S]*?)```",
            lambda m: f"<pre>{m.group(1)}</pre>",
            out,
        )
        out = re.sub(r"^## (.+)$", r"<h3>\1</h3>", out, flags=re.MULTILINE)
        out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
        out = out.replace("\n", "<br>")
        return out

    def poll_output(self) -> dict | None:
        with self._lock:
            html_body = self._render_thread()
            generation = self._output_generation
            busy = self._busy
            if generation == self._output_painted:
                return {"busy": busy}
        self._output_painted = generation
        return {"html": html_body, "busy": busy, "scroll_bottom": True}

    def paint_output_now(self) -> None:
        if self._shutting_down or not webview.windows:
            return
        data = self.poll_output()
        if not data or not data.get("html"):
            return
        try:
            webview.windows[0].evaluate_js(
                f"setChatOutput({json.dumps(data['html'])}, {json.dumps(bool(data.get('scroll_bottom')))});"
            )
        except Exception as exc:
            print(f"brain: paint failed: {exc}", flush=True)

"""PyWebView JS API for the brain application."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import webview

from brain.config import load_config
from brain.tools.codebase import CodebaseIndex


class BrainApi:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.callback_obj = None
        self._index = CodebaseIndex(load_config())
        self._request_text = ""
        self._file_list: list[dict] | None = None
        self._index_error: str | None = None
        self._index_thread: threading.Thread | None = None
        self._index_lock = threading.Lock()

    def set_callback_obj(self, obj) -> None:
        self.callback_obj = obj

    def start_index_build(self) -> None:
        """Build the file catalog in a background thread (non-blocking UI)."""
        with self._index_lock:
            if self._index_thread is not None and self._index_thread.is_alive():
                return
            self._file_list = None
            self._index_error = None
            self._index_thread = threading.Thread(
                target=self._build_index_worker,
                daemon=True,
            )
            self._index_thread.start()

    def _build_index_worker(self) -> None:
        try:
            entries = [
                {"rel_path": e.rel_path, "size": e.size, "kind": e.kind}
                for e in self._index.entries()
            ]
            with self._index_lock:
                self._file_list = entries
                self._index_error = None
        except Exception as exc:
            with self._index_lock:
                self._file_list = []
                self._index_error = f"{exc}\n{traceback.format_exc()}"

    def poll_file_tree(self) -> dict:
        """Return index status for the left-panel file tree."""
        with self._index_lock:
            building = self._index_thread is not None and self._index_thread.is_alive()
            if self._index_error:
                return {
                    "status": "error",
                    "entries": [],
                    "message": self._index_error,
                }
            if self._file_list is not None:
                return {
                    "status": "ready",
                    "entries": self._file_list,
                    "message": f"{len(self._file_list)} files",
                }
            if building:
                root = str(self._index.root)
                return {
                    "status": "loading",
                    "entries": [],
                    "message": f"Indexing {root}…",
                }
        return {
            "status": "loading",
            "entries": [],
            "message": "Starting index…",
        }

    def list_code_files(self) -> list[dict]:
        with self._index_lock:
            if self._file_list is not None:
                return list(self._file_list)
        self.start_index_build()
        with self._index_lock:
            if self._file_list is not None:
                return list(self._file_list)
        return []

    def read_code_file(self, rel_path: str) -> str:
        try:
            return self._index.read(rel_path)
        except (OSError, ValueError) as exc:
            return f"[error reading {rel_path}: {exc}]"

    def refresh_index(self) -> None:
        self._index.refresh()
        self.start_index_build()

    def update_request_text(self, text: str) -> None:
        if isinstance(text, str):
            self._request_text = text

    def get_request_text(self) -> str:
        if not webview.windows:
            return self._request_text
        try:
            value = webview.windows[0].evaluate_js(
                "document.getElementById('change-request') "
                "? document.getElementById('change-request').value : ''"
            )
            if isinstance(value, str):
                self._request_text = value
                return value
        except Exception:
            pass
        return self._request_text

    def submit_request(self, text: str) -> dict:
        if self.callback_obj is not None:
            self.callback_obj.submit_request(text)
        return {"ok": True}

    def stop_agent(self) -> dict:
        if self.callback_obj is not None:
            self.callback_obj.stop_agent()
        return {"ok": True}

    def poll_output(self) -> dict | None:
        ui = self.callback_obj
        if ui is None:
            return None
        return ui.poll_output()

    def get_conversation_history(self) -> dict:
        ui = self.callback_obj
        if ui is None:
            return {"turns": 0}
        return {"turns": len(ui._conversation.turns())}

    def clear_conversation(self) -> dict:
        ui = self.callback_obj
        if ui is not None:
            ui.clear_conversation()
        return {"ok": True}

    def on_link_click(self, link_id: str, link_type: str) -> dict:
        if self.callback_obj is not None:
            self.callback_obj.act(link_id)
        return {"id": link_id, "type": link_type}

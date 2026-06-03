
from __future__ import annotations

import atexit
import html
import json
import threading
from pathlib import Path

import webview

from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import ArrayBundle, Runtime, RuntimeCancelled, Session, create_session_dir
from ahlib.run_ah import _bootstrap_env
from app import Interface, session_root_from_input_json_ref
from externals.api import ExternalInput
from externals.invoke import release_gpu_resources, terminate_active_subprocesses

CHAT_SCRIPT = Path("chat") / "chat.ah"


def chat_script_path(base_dir: Path) -> Path:
    return base_dir / CHAT_SCRIPT


def load_chat_script(base_dir: Path) -> str:
    path = chat_script_path(base_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Missing chat script: {path}")
    return path.read_text(encoding="utf-8")


def actions_save_path(base_dir: Path) -> Path:
    return base_dir / "saves" / "chat_actions.json"


def load_saved_actions(base_dir: Path) -> list[dict[str, str]]:
    path = actions_save_path(base_dir)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    entries: list[dict] = []
    for item in data:
        if isinstance(item, dict) and "tm" in item and "data" in item:
            entry: dict = {"tm": str(item["tm"]), "data": str(item["data"])}
            if item.get("input_json_ref"):
                entry["input_json_ref"] = str(item["input_json_ref"])
            preview = item.get("finish_preview")
            if isinstance(preview, dict):
                if not preview.get("session_base_dir"):
                    root = session_root_from_input_json_ref(
                        base_dir, entry.get("input_json_ref")
                    )
                    if root is not None:
                        preview = dict(preview)
                        preview["session_base_dir"] = str(root)
                entry["finish_preview"] = preview
            entries.append(entry)
    return entries


def save_actions(entries: list[dict[str, str]], base_dir: Path) -> None:
    path = actions_save_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class LinkApi:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.counter = 0
        self.callback_obj = None

    def set_callback_obj(self, obj) -> None:
        self.callback_obj = obj

    def set_layout_fragments(
        self,
        top_left_html: str,
        top_right_html: str,
        bottom_left_html: str,
        bottom_right_html: str,
    ) -> bool:
        if not webview.windows:
            return False
        js_call = (
            "window.setTemplateFragments("
            f"{json.dumps(top_left_html)},"
            f"{json.dumps(top_right_html)},"
            f"{json.dumps(bottom_left_html)},"
            f"{json.dumps(bottom_right_html)}"
            ");"
        )
        webview.windows[0].evaluate_js(js_call)
        return True

    def set_html(self, place: str, html_data: str) -> None:
        if place not in ("top-left", "top-right", "bottom-left", "bottom-right"):
            return
        if not webview.windows:
            return
        js_call = (
            "window.setTemplateFragment("
            f"'{place}',"
            f"{json.dumps(html_data)}"
            ");"
        )
        try:
            webview.windows[0].evaluate_js(js_call)
        except Exception:
            pass

    def copy_text(self, text: str) -> bool:
        if not isinstance(text, str) or not text:
            return False
        if not webview.windows:
            return False
        try:
            webview.windows[0].evaluate_js(
                f"navigator.clipboard.writeText({json.dumps(text)})"
            )
            return True
        except Exception:
            return False

    def poll_chat_input_state(self) -> bool | None:
        """Apply chat input enable/disable from the WebView GUI thread."""
        ui = self.callback_obj
        if ui is None:
            return None
        return ui.take_chat_input_enabled_pending()

    def poll_log_refresh(self) -> str | None:
        ui = self.callback_obj
        if ui is None or not ui._log_draw_pending:
            return None
        ui._log_draw_pending = False
        return ui.html_page()

    def submit_chat_input(self, text: str) -> dict[str, str | int]:
        if self.callback_obj is not None:
            self.callback_obj.submit_user_input(text)
        self.counter += 1
        return {"counter": self.counter, "status": "ok"}

    def on_link_click(self, link_id: str, link_type: str) -> dict[str, str | int]:
        if self.callback_obj is not None:
            self.callback_obj.act(link_id)
        self.counter += 1
        return {"counter": self.counter, "id": link_id, "type": link_type}


class ChatInterface(Interface):
    def __init__(self, api: LinkApi):
        super().__init__(api)
        self._messages: list[str] = []
        self._answer_parts: list[str] = []
        self._code_html = ""
        self._waiting_for_input = False
        self._user_input_event = threading.Event()
        self._user_input_lock = threading.Lock()
        self._user_input_result = ""
        self._input_ui_lock = threading.Lock()
        self._chat_input_enabled_pending: bool | None = None

    def save_actions_now(self) -> str:
        save_actions(self.data, self.linkapi.base_dir)
        return str(actions_save_path(self.linkapi.base_dir))

    def load_actions_from_disk(self) -> None:
        self.data = load_saved_actions(self.linkapi.base_dir)
        if not self._shutting_down:
            self._queue_log_draw()

    def _request_chat_input_enabled(self, enabled: bool) -> None:
        with self._input_ui_lock:
            self._chat_input_enabled_pending = enabled

    def take_chat_input_enabled_pending(self) -> bool | None:
        with self._input_ui_lock:
            pending = self._chat_input_enabled_pending
            self._chat_input_enabled_pending = None
            return pending

    def request_stop_run(self) -> None:
        super().request_stop_run()
        self._release_user_input_wait()
        self._request_chat_input_enabled(True)

    def _release_user_input_wait(self) -> None:
        with self._user_input_lock:
            self._waiting_for_input = False
            self._user_input_event.set()

    def act(self, action: str) -> None:
        if action == "stop":
            if self._run_thread is not None and self._run_thread.is_alive():
                self.request_stop_run()
                self.add_action_results("<b>Stop requested</b>")
            else:
                self._request_chat_input_enabled(True)
                self.add_action_results("<b>No run in progress</b>")
        elif action == "clear":
            self.data = []
            self._answer_parts = []
            self.linkapi.set_html("top-right", "")
            self.add_action_results("<b>Clear</b>")
            try:
                self.save_actions_now()
            except Exception as exc:
                print(f"Save after clear failed: {exc}", flush=True)

    def submit_user_input(self, text: str) -> None:
        with self._user_input_lock:
            if not self._waiting_for_input:
                return
            self._user_input_result = text if isinstance(text, str) else ""
            self._waiting_for_input = False
            self._user_input_event.set()
        stripped = self._user_input_result.strip()
        if stripped:
            self._append_chat_pane(stripped, role="user")
        self._clear_chat_input_field()
        self._request_chat_input_enabled(False)

    def _append_chat_pane(self, text: str, *, role: str = "bot") -> None:
        css = "chat-user" if role == "user" else "chat-bot"
        block = f'<div class="chat-text {css}">{html.escape(text)}</div>'
        self._answer_parts.append(block)
        self.linkapi.set_html("top-right", '<hr>'.join(self._answer_parts))

    def _clear_chat_input_field(self) -> None:
        if not webview.windows:
            return
        try:
            webview.windows[0].evaluate_js(
                "var el=document.getElementById('chat-input'); if(el) el.value='';"
            )
        except Exception:
            pass

    def ah_action(
        self,
        name: str,
        bundle: ArrayBundle,
        inp: ExternalInput,
        args: dict[str, str],
        repeat: int = 1,
    ) -> ArrayBundle:
        if name == "user_input":
            return self._action_user_input()
        if name == "store_message":
            return self._action_store_message(bundle)
        if name == "get_messages":
            return self._action_get_messages()
        if name == "answer":
            return self._action_answer(bundle)
        if name == "show_code":
            return self._action_show_code(bundle)
        raise KeyError(f"Unknown callback action: ^{name}")

    def _session(self) -> Session:
        if self.session_dir is None:
            raise RuntimeError("Chat session is not active")
        return Session(self.session_dir)

    def _new_text_link(self, content: str) -> str:
        session = self._session()
        op_dir = session.next_op_dir("chat_text")
        text = content if content.endswith("\n") else content + "\n"
        return session.new_link(op_dir, "texts", ".txt", text)

    def _chat_read_link_text(self, link: str) -> str:
        if self.session_dir is None:
            return link
        path = Path(link)
        if not path.is_absolute():
            path = self.session_dir / link
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def _action_user_input(self) -> ArrayBundle:
        self._request_chat_input_enabled(True)
        with self._user_input_lock:
            self._waiting_for_input = True
            self._user_input_result = ""
            self._user_input_event.clear()
        while True:
            if self._cancel_event is not None and self._cancel_event.is_set():
                self._request_chat_input_enabled(True)
                raise RuntimeCancelled("Execution cancelled")
            if self._user_input_event.wait(timeout=0.2):
                break
        with self._user_input_lock:
            text = self._user_input_result
            self._waiting_for_input = False
        self._request_chat_input_enabled(False)
        out = ArrayBundle()
        if text.strip():
            out.texts.append(self._new_text_link(text.strip()))
        return out

    def _action_store_message(self, bundle: ArrayBundle) -> ArrayBundle:
        for link in bundle.texts:
            text = self._chat_read_link_text(link).strip()
            if text:
                self._messages.append(text)
        return bundle.copy()

    def _action_get_messages(self) -> ArrayBundle:
        out = ArrayBundle()
        for msg in self._messages:
            out.texts.append(self._new_text_link(msg))
        return out

    def _action_answer(self, bundle: ArrayBundle) -> ArrayBundle:
        for link in bundle.texts:
            text = self._chat_read_link_text(link)
            if text.strip():
                self._append_chat_pane(text.strip(), role="bot")
        return bundle.copy()

    def _action_show_code(self, bundle: ArrayBundle) -> ArrayBundle:
        chunks = [
            self._chat_read_link_text(link)
            for link in bundle.texts
            if self._chat_read_link_text(link).strip()
        ]
        joined = "\n\n".join(chunks)
        self._code_html = (
            f'<pre class="chat-code">{html.escape(joined)}</pre>'
            if joined
            else ""
        )
        self.linkapi.set_html("bottom-left", self._code_html)
        return bundle.copy()

    def start_chat_script(self) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            return
        source = load_chat_script(self.linkapi.base_dir)
        self._cancel_event = threading.Event()
        self._run_thread = threading.Thread(
            target=self._run_script_worker,
            args=(source,),
            daemon=True,
        )
        self._run_thread.start()

    def _run_script_worker(self, source: str) -> None:
        was_cancelled = False
        self.add_action_results("<b>Chat started</b>")
        self.session_dir = create_session_dir(Path("sessions"))
        try:
            program = parse_ah_source(source)
            runtime = Runtime(
                program,
                Session(self.session_dir),
                callback=self,
                cancel_event=self._cancel_event,
            )
            result = runtime.run()
            if self._cancel_event is not None and self._cancel_event.is_set():
                was_cancelled = True
                self.add_action_results("<b>Chat cancelled</b>")
                return
            meta = {
                "session": str(self.session_dir),
                "run": program.run_target,
                "output": result.as_dict(),
            }
            output_json = (
                str(runtime.last_output_json_path.resolve())
                if runtime.last_output_json_path is not None
                else None
            )
            self.add_action_results(
                (
                    f"<b>Chat finished</b><br>"
                    f"<small>{html.escape(str(self.session_dir))}</small>"
                    f"<br>{self._format_result_block(meta['output'], session_root=self.session_dir)}"
                    f"<br>{self._format_json_block(meta)}"
                ),
                input_json_ref=output_json,
                finish_preview={
                    "action_name": "Run finished",
                    "output_context": meta["output"],
                    "session_base_dir": str(self.session_dir.resolve()),
                },
            )
            self._queue_log_draw()
        except RuntimeCancelled:
            was_cancelled = True
            self.add_action_results("<b>Chat cancelled</b>")
        except Exception as exc:
            self.add_action_results(
                f"<b>Chat failed</b><br><pre>{html.escape(str(exc))}</pre>"
            )
        finally:
            release_gpu_resources(reason="chat run finished")
            self._cancel_event = None
            self._run_thread = None
            self._request_chat_input_enabled(True)
            with self._user_input_lock:
                self._waiting_for_input = False
                self._user_input_event.set()
            if was_cancelled and not self._shutting_down:
                self.add_action_results("<b>Restarting chat</b>")
                self.start_chat_script()


def main() -> None:
    _bootstrap_env()

    base_dir = Path(__file__).resolve().parent
    html_file = base_dir / "html" / "chat_page.html"
    api = LinkApi(base_dir)
    ui = ChatInterface(api)
    api.set_callback_obj(ui)

    def _atexit_save() -> None:
        try:
            ui.save_actions_now()
        except Exception as exc:
            print(f"Exit save failed: {exc}", flush=True)

    atexit.register(_atexit_save)

    webview.create_window(
        title="Anthill Chat",
        url=html_file.as_uri(),
        js_api=api,
        width=1000,
        height=700,
        text_select=True,
    )

    input_area = """
<style>
  .btn {
    display: inline-block;
    margin-top: 4px;
    margin-left: 6px;
    padding: 4px 10px;
    background-color: #007bff;
    color: white;
    text-decoration: none;
    text-align: center;
    border-radius: 4px;
    font-size: 14px;
    font-family: Arial, Helvetica, sans-serif;
    width: fit-content;
    cursor: pointer;
    user-select: none;
  }
  .btn:disabled, .btn.disabled {
    opacity: 0.45;
    pointer-events: none;
    cursor: default;
  }
  .btn-stop { background-color: #c9302c; }
  .btn-stop:hover { background-color: #a94442; }
</style>
<form style="width:100%; height:100%; margin:0; font-family:Arial, Helvetica, sans-serif;"
      onsubmit="return false;">
  <textarea
    id="chat-input"
    disabled
    style="
      width:100%;
      height:calc(100% - 40px);
      box-sizing:border-box;
      resize:none;
      display:block;
      margin:0;
      padding:6px;
    "
  ></textarea>
  <div style="text-align: right; margin-top: 8px;">
    <a class="btn btn-stop" href="#" onclick="event.preventDefault(); if(confirm('Stop chat?')){ window.pywebview.api.on_link_click('stop',''); }">Stop</a>
    <a class="btn" href="#" onclick="event.preventDefault(); if(confirm('Clear log and chat?')){ window.pywebview.api.on_link_click('clear',''); }">Clear</a>
    <a class="btn disabled" id="chat-run" href="#" onclick="event.preventDefault(); window.submitChatInput();">Run</a>
  </div>
</form>
    """

    def on_loaded() -> None:
        api.set_layout_fragments(
            "<h3>Action log</h3>",
            "<h3>Chat</h3>",
            "<h3>Code</h3>",
            input_area,
        )
        ui.load_actions_from_disk()
        ui.paint_log_now()
        ui.start_chat_script()

    def on_closing() -> bool:
        ui.begin_shutdown()
        try:
            ui.save_actions_now()
        except Exception as exc:
            print(f"Failed to save on exit: {exc}", flush=True)
        ui.wait_for_run(timeout=2.0)
        return True

    webview.windows[0].events.loaded += on_loaded
    webview.windows[0].events.closing += on_closing
    webview.start(debug=False)


if __name__ == "__main__":
    main()

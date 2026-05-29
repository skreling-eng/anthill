
from __future__ import annotations

import atexit
import html
import json
import threading
from datetime import datetime
from pathlib import Path

import webview

from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import Runtime, RuntimeCancelled, Session, create_session_dir
from ahlib.run_ah import _bootstrap_env
from externals.invoke import terminate_active_subprocesses

DEFAULT_SCRIPT = """@hello:
hello

@run: @hello -> $clear

run @run
"""


def script_save_path(base_dir: Path) -> Path:
    return base_dir / "saves" / "default.ah"


def load_saved_script(base_dir: Path) -> str:
    path = script_save_path(base_dir)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return DEFAULT_SCRIPT


def save_script(source: str, base_dir: Path) -> None:
    path = script_save_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def actions_save_path(base_dir: Path) -> Path:
    return base_dir / "saves" / "default_actions.json"


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
    entries: list[dict[str, str]] = []
    for item in data:
        if isinstance(item, dict) and "tm" in item and "data" in item:
            entry = {"tm": str(item["tm"]), "data": str(item["data"])}
            if item.get("input_json_ref"):
                entry["input_json_ref"] = str(item["input_json_ref"])
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
    _AUTOSAVE_DELAY_S = 1.0

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.counter = 0
        self.callback_obj = None
        self._script_text = ""
        self._save_timer: threading.Timer | None = None
        self._save_lock = threading.Lock()

    def set_callback_obj(self, obj):
        self.callback_obj = obj

    def set_script_buffer(self, source: str) -> None:
        self._script_text = source if isinstance(source, str) else ""

    def save_script_now(self) -> str:
        """Write saves/default.ah immediately; return path written."""
        with self._save_lock:
            save_script(self._script_text, self.base_dir)
            return str(script_save_path(self.base_dir))

    def save_all_now(self) -> None:
        """Persist script and action log."""
        self.save_script_now()
        if self.callback_obj is not None:
            self.callback_obj.save_actions_now()

    def _schedule_autosave(self) -> None:
        with self._save_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
            self._save_timer = threading.Timer(
                self._AUTOSAVE_DELAY_S, self._autosave_tick
            )
            self._save_timer.daemon = True
            self._save_timer.start()

    def _autosave_tick(self) -> None:
        try:
            self.save_all_now()
        except Exception as exc:
            print(f"Autosave failed: {exc}", flush=True)

    def update_script_text(self, text: str) -> None:
        """Called from the textarea on input/blur to keep a Python-side copy."""
        if isinstance(text, str):
            self._script_text = text
            self._schedule_autosave()

    def get_script_source(self) -> str:
        if not webview.windows:
            return self._script_text
        try:
            value = webview.windows[0].evaluate_js(
                "document.getElementById('ah-script') "
                "? document.getElementById('ah-script').value "
                ": ''"
            )
            if isinstance(value, str):
                self._script_text = value
                return value
        except Exception:
            pass
        return self._script_text

    def script_text_for_save(self) -> str:
        """Best-effort read for shutdown; never blocks on a dying webview."""
        return self.get_script_source()

    def set_script_source(self, source: str) -> None:
        self._script_text = source if isinstance(source, str) else ""
        if not webview.windows:
            return
        try:
            webview.windows[0].evaluate_js(
                f"document.getElementById('ah-script').value = {json.dumps(source)};"
            )
        except Exception:
            pass

    def set_layout_fragments(
        self,
        top_left_html: str,
        top_right_html: str,
        bottom_left_html: str,
        bottom_right_html: str,
    ) -> bool:
        """Set 4 HTML fragments into the page template areas."""
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

    def set_html(self, place, html_data):
        if place not in ('top-left', 'top-right', 'bottom-left', 'bottom-right'):
            print('WRONG PLACE')
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

    def on_link_click(self, link_id: str, link_type: str) -> dict[str, str | int]:
        """Handle any link click callback from HTML and return updated state."""
        if self.callback_obj is not None:
            self.callback_obj.act(link_id)

        self.counter += 1
        return {"counter": self.counter, "id": link_id, "type": link_type}


class Interface:
    _TEXT_PREVIEW_LIMIT = 200
    _MEDIA_PREVIEW_COUNT = 3
    _MEDIA_HEIGHT = 100

    def __init__(self, api):
        self.linkapi = api
        self.data = []
        self.session_dir: Path | None = None
        self._cancel_event: threading.Event | None = None
        self._run_thread: threading.Thread | None = None
        self._shutting_down = False
        self._actions_save_timer: threading.Timer | None = None
        self._actions_save_lock = threading.Lock()

    def begin_shutdown(self) -> None:
        """Stop UI updates and signal any in-flight run to exit."""
        self._shutting_down = True
        self.request_stop_run()

    def request_stop_run(self) -> None:
        """Signal an in-flight script run to stop and kill child processes."""
        if self._cancel_event is not None:
            self._cancel_event.set()
        terminate_active_subprocesses()

    def wait_for_run(self, timeout: float = 10.0) -> None:
        thread = self._run_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def save_actions_now(self) -> str:
        with self._actions_save_lock:
            save_actions(self.data, self.linkapi.base_dir)
            return str(actions_save_path(self.linkapi.base_dir))

    def load_actions_from_disk(self) -> None:
        self.data = load_saved_actions(self.linkapi.base_dir)
        if not self._shutting_down:
            self.draw("top-left", self.html_page())

    def _schedule_actions_save(self) -> None:
        if self._shutting_down:
            return
        with self._actions_save_lock:
            if self._actions_save_timer is not None:
                self._actions_save_timer.cancel()
            self._actions_save_timer = threading.Timer(
                LinkApi._AUTOSAVE_DELAY_S, self._actions_save_tick
            )
            self._actions_save_timer.daemon = True
            self._actions_save_timer.start()

    def _actions_save_tick(self) -> None:
        try:
            self.save_actions_now()
        except Exception as exc:
            print(f"Action log autosave failed: {exc}", flush=True)

    def act(self, action):
        if action == 'run':
            self._run_script()
        elif action == 'clear':
            self.data = []
            self.add_action_results('<b>Clear</b>')
            try:
                self.save_actions_now()
            except Exception as exc:
                print(f"Save after clear failed: {exc}", flush=True)

    @staticmethod
    def _compact_text(text: str, limit: int) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if len(normalized) <= limit:
            return html.escape(normalized)
        return html.escape(normalized[:limit]) + "…"

    def _media_uri(self, link: str) -> str | None:
        if self.session_dir is None:
            return None
        path = (self.session_dir / link).resolve()
        if not path.is_file():
            return None
        return html.escape(path.as_uri())

    def _read_link_text(self, link: str) -> str:
        if self.session_dir is None:
            return link
        path = self.session_dir / link
        if not path.is_file():
            return f"[missing: {link}]"
        return path.read_text(encoding="utf-8", errors="replace")

    def _gallery_img_tag(self, uri: str, label: str, css_class: str, index: int) -> str:
        return (
            f'<img class="{css_class} gallery-img" src="{uri}" '
            f'data-gallery-index="{index}" '
            f'height="{self._MEDIA_HEIGHT}" alt="{label}" title="{label}">'
        )

    def _format_images_block(self, links: list[str]) -> str:
        count = len(links)
        uri_entries: list[tuple[str, str, str]] = []
        for link in links:
            uri = self._media_uri(link)
            label = html.escape(Path(link).name)
            if uri:
                uri_entries.append((uri, label, link))

        if not uri_entries:
            return (
                '<details class="result-fold result-images">'
                f'<summary><span class="result-title">Images [{count}]:</span> '
                '<span class="result-preview-row"><span class="result-missing">'
                "(no files)</span></span>"
                "</summary></details>"
            )

        uris_json = html.escape(
            json.dumps([uri for uri, _, _ in uri_entries]), quote=True
        )
        preview_imgs = []
        for index, (uri, label, _) in enumerate(
            uri_entries[: self._MEDIA_PREVIEW_COUNT]
        ):
            preview_imgs.append(
                self._gallery_img_tag(uri, label, "result-thumb", index)
            )

        all_imgs = [
            self._gallery_img_tag(uri, label, "result-media", index)
            for index, (uri, label, _) in enumerate(uri_entries)
        ]

        suffix = " ..." if count > self._MEDIA_PREVIEW_COUNT else ""
        return (
            '<details class="result-fold result-images" '
            f'data-images="{uris_json}">'
            f'<summary><span class="result-title">Images [{count}]:</span> '
            f'<span class="result-preview-row">{"".join(preview_imgs)}{suffix}</span>'
            f"</summary>"
            f'<div class="result-list result-list-images">{"".join(all_imgs)}</div>'
            "</details>"
        )

    def _format_videos_block(self, links: list[str]) -> str:
        count = len(links)
        preview_links = links[: self._MEDIA_PREVIEW_COUNT]

        def _video_tag(link: str, css_class: str) -> str:
            uri = self._media_uri(link)
            label = html.escape(Path(link).name)
            if not uri:
                return f"<span class='result-missing'>{label}</span>"
            return (
                f'<video class="{css_class}" src="{uri}" '
                f'height="{self._MEDIA_HEIGHT}" controls preload="metadata"></video>'
            )

        preview_videos = [_video_tag(link, "result-thumb") for link in preview_links]
        all_videos = [_video_tag(link, "result-media") for link in links]
        suffix = " ..." if count > self._MEDIA_PREVIEW_COUNT else ""
        return (
            '<details class="result-fold result-videos">'
            f'<summary><span class="result-title">Videos [{count}]:</span> '
            f'<span class="result-preview-row">{"".join(preview_videos)}{suffix}</span>'
            f"</summary>"
            f'<div class="result-list result-list-videos">{"".join(all_videos)}</div>'
            "</details>"
        )

    def _format_sounds_block(self, links: list[str]) -> str:
        count = len(links)
        preview_links = links[: self._MEDIA_PREVIEW_COUNT]

        def _audio_tag(link: str, css_class: str) -> str:
            uri = self._media_uri(link)
            label = html.escape(Path(link).name)
            if not uri:
                return f"<span class='result-missing'>{label}</span>"
            return (
                f'<audio class="{css_class}" src="{uri}" '
                f'controls preload="metadata"></audio>'
            )

        preview_sounds = [_audio_tag(link, "result-thumb") for link in preview_links]
        all_sounds = [_audio_tag(link, "result-media") for link in links]
        suffix = " ..." if count > self._MEDIA_PREVIEW_COUNT else ""
        return (
            '<details class="result-fold result-sounds">'
            f'<summary><span class="result-title">Sounds [{count}]:</span> '
            f'<span class="result-preview-row">{"".join(preview_sounds)}{suffix}</span>'
            f"</summary>"
            f'<div class="result-list result-list-sounds">{"".join(all_sounds)}</div>'
            "</details>"
        )

    def _format_text_item(self, title: str, link: str) -> str:
        full_text = self._read_link_text(link)
        compact = self._compact_text(full_text, self._TEXT_PREVIEW_LIMIT)
        return (
            '<details class="result-fold result-text-item">'
            f'<summary><span class="result-title">{html.escape(title)}:</span> {compact}</summary>'
            f'<pre class="result-text-full">{html.escape(full_text)}</pre>'
            "</details>"
        )

    def _format_output_preview(self, output_context: dict) -> str:
        if self.session_dir is None:
            return ""

        parts: list[str] = []
        images = output_context.get("images") or []
        if images:
            parts.append(self._format_images_block(list(images)))

        videos = output_context.get("videos") or []
        if videos:
            parts.append(self._format_videos_block(list(videos)))

        sounds = output_context.get("sounds") or []
        if sounds:
            parts.append(self._format_sounds_block(list(sounds)))

        texts = output_context.get("texts") or []
        for index, link in enumerate(texts, start=1):
            parts.append(self._format_text_item(f"Text{index}", link))

        prompts = output_context.get("prompts") or []
        for index, link in enumerate(prompts, start=1):
            parts.append(self._format_text_item(f"Prompt{index}", link))

        if not parts:
            return ""
        return '<div class="result-preview">' + "".join(parts) + "</div>"

    def _format_result_block(self, output_context: dict) -> str:
        preview = self._format_output_preview(output_context)
        json_block = self._format_json_block(output_context)
        if preview:
            return preview + json_block
        return json_block

    def _format_json_block(self, data) -> str:
        compact = html.escape(
            json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        )
        pretty = html.escape(json.dumps(data, indent=2, ensure_ascii=False))
        return (
            '<details class="json-fold">'
            f"<summary><code>{compact}</code></summary>"
            f"<pre>{pretty}</pre>"
            "</details>"
        )

    def action_start(self, action_name: str) -> None:
        self.add_action_results(f"<b>START</b> {html.escape(action_name)}")

    def _input_json_ref(self, json_path: str | Path) -> str:
        path = Path(json_path)
        if not path.is_absolute():
            return f"input_json('{path.as_posix()}')"
        try:
            rel = path.resolve().relative_to(self.linkapi.base_dir.resolve())
        except ValueError:
            rel = path.resolve()
        return f"input_json('{rel.as_posix()}')"

    def action_finish(
        self,
        action_name: str,
        output_context: dict,
        output_json_path: str | None = None,
    ) -> None:
        self.add_action_results(
            f"<b>FINISH</b> {html.escape(action_name)}"
            f"<br>{self._format_result_block(output_context)}",
            input_json_ref=output_json_path,
        )

    def action_error(self, action_name: str, error_message: str) -> None:
        self.add_action_results(
            f"<b>ERROR</b> {html.escape(action_name)}"
            f"<br><pre>{html.escape(error_message)}</pre>"
        )

    def _run_script(self) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            self.add_action_results("<b>Run already in progress</b>")
            return

        source = self.linkapi.get_script_source()
        if not source.strip():
            self.add_action_results("<b>Empty script</b>")
            return

        try:
            self.linkapi.save_all_now()
        except Exception as exc:
            print(f"Save before run failed: {exc}", flush=True)

        self._cancel_event = threading.Event()
        self._run_thread = threading.Thread(
            target=self._run_script_worker,
            args=(source,),
            daemon=True,
        )
        self._run_thread.start()

    def _run_script_worker(self, source: str) -> None:
        self.add_action_results("<b>Run started</b>")
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
                self.add_action_results("<b>Run cancelled</b>")
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
                f"<b>Run finished</b><br>"
                f"<small>{html.escape(str(self.session_dir))}</small>"
                f"<br>{self._format_result_block(meta['output'])}"
                f"<br>{self._format_json_block(meta)}",
                input_json_ref=output_json,
            )
        except RuntimeCancelled:
            self.add_action_results("<b>Run cancelled</b>")
        except Exception as exc:
            self.add_action_results(
                f"<b>Run failed</b><br><pre>{html.escape(str(exc))}</pre>"
            )
        finally:
            self._cancel_event = None
            self._run_thread = None

    def _format_log_header(self, entry: dict) -> str:
        tm = f"<small>{html.escape(entry['tm'])}</small>"
        copy_text = entry.get("input_json_ref")
        if not copy_text:
            return tm
        attr = copy_text.replace("&", "&amp;").replace('"', "&quot;")
        btn = (
            '<a href="#" class="copy-json-ref" '
            f'data-copy="{attr}" '
            "onclick=\"event.preventDefault();"
            "window.pywebview.api.copy_text(this.getAttribute('data-copy'));"
            '">Copy the path to buffer</a>'
        )
        return f'<div class="log-head">{tm}{btn}</div>'

    def html_page(self):
        res = []
        for e in self.data[::-1]:
            res.append(f"{self._format_log_header(e)}<br>{e['data']}")
        dlm = '<hr>'
        return dlm.join(res)

    def add_action_results(self, data, *, input_json_ref: str | Path | None = None):
        entry: dict[str, str] = {
            'tm': str(datetime.now()),
            'data': data,
        }
        if input_json_ref:
            raw = str(input_json_ref)
            entry["input_json_ref"] = (
                raw if raw.startswith("input_json(") else self._input_json_ref(raw)
            )
        self.data.append(entry)
        if not self._shutting_down:
            self.draw('top-left', self.html_page())
        self._schedule_actions_save()

    def draw(self, place, htmldata):
        if self._shutting_down:
            return
        self.linkapi.set_html(place, htmldata)


def main() -> None:
    _bootstrap_env()

    base_dir = Path(__file__).resolve().parent
    saved_script = load_saved_script(base_dir)
    save_path = script_save_path(base_dir)
    if not save_path.is_file():
        save_script(saved_script, base_dir)

    html_file = base_dir / "html" / "test_page.html"
    api = LinkApi(base_dir)
    ui = Interface(api)
    api.set_callback_obj(ui)

    def _atexit_save() -> None:
        try:
            api.save_all_now()
        except Exception as exc:
            print(f"Exit save failed: {exc}", flush=True)

    atexit.register(_atexit_save)

    webview.create_window(
        title="Anthill",
        url=html_file.as_uri(),
        js_api=api,
        width=900,
        height=600,
        text_select=True,
    )

    inpit_area = """
<style>
  .btn {
    display: inline-block;
    margin-top: 4px;
    padding: 4px 6px;
    background-color: #007bff;
    color: white;
    text-decoration: none;
    text-align: center;
    border-radius: 4px;
    font-size: 14px;
    font-family: Arial, Helvetica, sans-serif;
    width: fit-content;
    cursor: pointer;
    -webkit-user-select: none;
    user-select: none;
  }
</style>
<form style="width:100%; height:100%; margin:0; font-family:Arial, Helvetica, sans-serif;">
  <textarea
    id="ah-script"
    oninput="window.pywebview && window.pywebview.api && window.pywebview.api.update_script_text(this.value)"
    onblur="window.pywebview && window.pywebview.api && window.pywebview.api.update_script_text(this.value)"
    style="
      width:100%;
      height:calc(100% - 40px);
      box-sizing:border-box;
      resize:none;
      display:block;
      margin: 0px 0px;
      padding: 3px 3px;
    "
></textarea>
  <div style="text-align: right; margin-top: 8px;">
    <a class="btn" id='' href="#" onclick="if(confirm('Are you sure you want to clear?')){ window.pywebview.api.on_link_click('clear', ''); }">Clear</a>
    <a class="btn" id='run' href="#">Run</a>
  </div>
</form>
    """

    def on_loaded() -> None:
        api.set_script_buffer(saved_script)
        api.set_layout_fragments(
            "<h2>Action results</h2>",
            "<h3>Настройки</h3>",
            inpit_area,
            "<h3>Параметры</h3>",
        )
        api.set_script_source(saved_script)
        ui.load_actions_from_disk()
        try:
            api.save_all_now()
        except Exception as exc:
            print(f"Initial save failed: {exc}", flush=True)

    def on_closing() -> bool:
        ui.begin_shutdown()
        try:
            api.save_all_now()
        except Exception as exc:
            print(f"Failed to save on exit: {exc}", flush=True)
        ui.wait_for_run(timeout=2.0)
        return True

    webview.windows[0].events.loaded += on_loaded
    webview.windows[0].events.closing += on_closing
    webview.start(debug=False)


if __name__ == "__main__":
    main()

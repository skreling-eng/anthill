"""Brain desktop application — AH codebase analyzer with diff proposals."""

from __future__ import annotations

import atexit
import sys
from pathlib import Path

import webview

from brain.config import load_config
from brain.ui.api import BrainApi
from brain.ui.interface import BrainInterface

_BRAIN_DIR = Path(__file__).resolve().parent
_HTML = _BRAIN_DIR / "html" / "page.html"


def main() -> None:
    config = load_config()
    config.ensure_sessions_dir()
    (config.brain_dir / "saves").mkdir(parents=True, exist_ok=True)

    if not _HTML.is_file():
        print(f"brain: missing UI template {_HTML}", file=sys.stderr)
        sys.exit(1)

    api = BrainApi(config.brain_dir)
    ui = BrainInterface(api, config)
    api.set_callback_obj(ui)

    def _atexit_save() -> None:
        try:
            ui.save_conversation()
        except Exception as exc:
            print(f"brain: exit save failed: {exc}", flush=True)

    atexit.register(_atexit_save)

    webview.create_window(
        title="Brain — AH Code Analyzer",
        url=_HTML.as_uri(),
        js_api=api,
        width=1100,
        height=720,
        text_select=True,
    )

    def on_loaded() -> None:
        api.refresh_index()
        ui.load_conversation()
        ui.paint_output_now()
        try:
            webview.windows[0].evaluate_js(
                "typeof pollFileTree === 'function' && pollFileTree();"
            )
        except Exception:
            pass

    def on_closing() -> bool:
        ui.begin_shutdown()
        ui.wait_for_agent(timeout=3.0)
        return True

    webview.windows[0].events.loaded += on_loaded
    webview.windows[0].events.closing += on_closing
    webview.start(debug=False)


if __name__ == "__main__":
    main()

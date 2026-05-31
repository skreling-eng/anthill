"""Minimal ComfyUI ``server`` stub for headless comfy_lib + vendored custom nodes."""

from __future__ import annotations

import uuid
from typing import Any


class _PromptQueue:
    currently_running: dict = {}

    def put(self, item: Any) -> None:
        _ = item


class _RouteTable:
    def get(self, _path: str):
        def decorator(fn):
            return fn

        return decorator


class _WebApp:
    routes = _RouteTable()


class PromptServer:
    """Subset of ComfyUI PromptServer used by VHS / WanVideoWrapper."""

    instance: PromptServer | None = None
    number = 0
    client_id: str | None = None
    last_node_id: str | None = None
    sockets_metadata: dict = {}

    def __init__(self) -> None:
        self.prompt_queue = _PromptQueue()
        self.web = _WebApp()
        self.routes = self.web.routes

    def send_sync(self, *_args, **_kwargs) -> None:
        pass

    @classmethod
    def _ensure_instance(cls) -> PromptServer:
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance


# ComfyUI modules access ``server.PromptServer.instance`` at import time.
_ps = PromptServer._ensure_instance()
web = _ps.web


class Response:
    """Placeholder for aiohttp Response in headless mode."""

    pass


web.Response = Response  # type: ignore[attr-defined]

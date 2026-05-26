"""Common API for external ($) action handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ahlib.ah_runtime import ArrayBundle, Session


@dataclass
class ExternalContext:
    """Session-scoped context passed to every external handler."""

    session: Session
    op_dir: Path

    @property
    def base_dir(self) -> Path:
        return self.session.base_dir

    def new_link(self, array_name: str, ext: str, content: str | bytes) -> str:
        return self.session.new_link(self.op_dir, array_name, ext, content)

    def read_link_text(self, link: str) -> str:
        path = self.base_dir / link
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace").strip()
        return ""

    def read_link_bytes(self, link: str) -> bytes:
        path = self.base_dir / link
        return path.read_bytes() if path.exists() else b""


@dataclass
class ExternalInput:
    """Inputs for an external action invocation."""

    bundle: ArrayBundle
    args: dict[str, str]
    prompt_text: str
    repeat: int = 1  # from $name(...)[n] — variants per prompt when multiple prompts
    arg_lists: dict[str, list[str]] = field(
        default_factory=dict
    )  # from key=@instruction — values from that instruction's output texts


def read_arg_list(inp: ExternalInput, key: str, default: str = "default") -> list[str]:
    """Scalar list for an external arg; expands key=@ref via arg_lists."""
    if key in inp.arg_lists and inp.arg_lists[key]:
        return list(inp.arg_lists[key])
    if key in inp.args and inp.args[key]:
        return [inp.args[key]]
    return [default]


def read_bundle_texts(ctx: ExternalContext, inp: ExternalInput) -> list[str]:
    """All strings from input texts[] links."""
    texts: list[str] = []
    for link in inp.bundle.texts:
        text = ctx.read_link_text(link)
        if text:
            texts.append(text)
    return texts


def read_prompt_texts(ctx: ExternalContext, inp: ExternalInput) -> list[str]:
    """All input prompts from bundle links, else a single combined prompt_text."""
    texts: list[str] = []
    for link in inp.bundle.prompts:
        text = ctx.read_link_text(link)
        if text:
            texts.append(text)
    if texts:
        return texts
    if inp.prompt_text.strip():
        return [inp.prompt_text.strip()]
    return []


class ExternalHandler(Protocol):
    def run(self, ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle: ...

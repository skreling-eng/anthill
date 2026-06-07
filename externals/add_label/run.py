"""$add_label — pass input through and tag every element with a label entry."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle
from ahlib.label_utils import add_label_for_elements


def _label_name(inp: ExternalInput) -> str:
    name = (inp.args.get("label") or inp.args.get("_path") or "").strip()
    if not name:
        raise ValueError(
            "$add_label(...) requires a label name, e.g. $add_label('important_change')"
        )
    return name


def run(_ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    return add_label_for_elements(inp.bundle, _label_name(inp))

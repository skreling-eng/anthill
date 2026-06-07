"""$label — filter bundle elements by structured label entries."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle
from ahlib.label_utils import exclude_by_label_name, filter_by_label_name


def _label_spec(inp: ExternalInput) -> tuple[str, bool]:
    if "not" in inp.args:
        name = inp.args["not"].strip()
        if not name:
            raise ValueError(
                "$label(not 'name') requires a label name, e.g. $label(not 'good')"
            )
        return name, True
    name = (inp.args.get("label") or inp.args.get("_path") or "").strip()
    if not name:
        raise ValueError("$label(...) requires a label name, e.g. $label('good')")
    return name, False


def run(_ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    name, inverted = _label_spec(inp)
    if inverted:
        return exclude_by_label_name(inp.bundle, name)
    return filter_by_label_name(inp.bundle, name)

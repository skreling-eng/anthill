"""Minimal comfy_api.latest._io stubs for vendored execution.py (legacy Wan workflows)."""

from __future__ import annotations

from typing import Any, TypedDict


class V3Data(TypedDict, total=False):
    hidden_inputs: dict[str, Any]
    dynamic_paths: dict[str, Any]
    dynamic_paths_default_value: dict[str, Any]
    create_dynamic_tuple: bool


class _HiddenEnum:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class Hidden:
    unique_id = _HiddenEnum("UNIQUE_ID")
    prompt = _HiddenEnum("PROMPT")
    extra_pnginfo = _HiddenEnum("EXTRA_PNGINFO")
    dynprompt = _HiddenEnum("DYNPROMPT")
    auth_token_comfy_org = _HiddenEnum("AUTH_TOKEN_COMFY_ORG")
    api_key_comfy_org = _HiddenEnum("API_KEY_COMFY_ORG")


class Combo:
    io_type = "COMBO"


class _ComfyNodeBaseInternal:
    INPUT_TYPES = lambda: {"required": {}, "optional": {}, "hidden": {}}
    ACCEPT_ALL_INPUTS = False


def get_finalized_class_inputs(
    class_inputs: Any,
    inputs: Any,
    v3_data: V3Data | None = None,
) -> tuple[Any, dict[str, Any], V3Data]:
    return class_inputs, {}, v3_data if v3_data is not None else {}


def build_nested_inputs(inputs: dict[str, Any], _v3_data: dict | None) -> dict[str, Any]:
    return inputs

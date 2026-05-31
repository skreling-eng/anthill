"""Minimal internal API markers for execution.py (legacy nodes only)."""

from __future__ import annotations

import inspect


class _ComfyNodeInternal:
    """V3 Comfy nodes subclass this; Anthill MEGA/I2V workflows use legacy nodes."""


class _NodeOutputInternal:
    pass


def first_real_override(cls, name: str):
    for base in cls.__mro__:
        if name in base.__dict__ and getattr(base, name) is not None:
            return getattr(base, name)
    return None


def is_class(obj) -> bool:
    return inspect.isclass(obj)


def make_locked_method_func(type_obj, func_name: str, instance):
    return getattr(instance, func_name)

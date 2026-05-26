"""Shared pytest hooks for Anthill tests."""

from __future__ import annotations

import os

import pytest

_EMULATE_ENV_KEYS = (
    "AH_EMULATE_MUSIC",
    "AH_EMULATE_LLM",
    "AH_EMULATE_IMAGE",
    "AH_EMULATE_IMAGE_CLIP",
    "AH_EMULATE_VIDEO_CLIP",
    "AH_EMULATE_MUSIC_SEPARATION",
    "AH_EMULATE_CHANGE_VOICE",
    "AH_EMULATE_JOIN_STEMS",
    "AH_EMULATE_VOICE_ENHANCE",
    "AH_EMULATE_COMFY",
    "AH_EMULATE_DRAW_TEXT",
    "AH_EMULATE_FILE",
    "AH_EMULATE_FOLDER",
    "AH_EXTERNAL_INPROCESS",
    "AH_EXTERNAL_SUBPROCESS",
)


@pytest.fixture(autouse=True)
def _isolate_emulate_env():
    saved = {key: os.environ.get(key) for key in _EMULATE_ENV_KEYS}
    for key in _EMULATE_ENV_KEYS:
        os.environ.pop(key, None)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

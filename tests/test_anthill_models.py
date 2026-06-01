"""Tests for unified models/ resolution and anthill on-demand fetch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from externals import anthill_models as am


def test_resolve_models_file_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = tmp_path / "models"
    target = models / "kokoro" / "config.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(am, "models_roots", lambda: (models,))
    found = am.resolve_models_file("kokoro/config.json")
    assert found == target.resolve()


def test_files_ready_false_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(am, "models_roots", lambda: (models,))
    assert not am.files_ready(am.CHECKS["kokoro"])


def test_ensure_anthill_file_downloads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    dest = models / "demucs-openvino" / "htdemucs_v4" / "htdemucs_v4.xml"
    dest.parent.mkdir(parents=True)
    dest.write_text("<xml/>", encoding="utf-8")

    calls: list[str] = []

    def fake_download(rel: str) -> Path:
        calls.append(rel)
        out = models.joinpath(*rel.split("/"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("ok", encoding="utf-8")
        return out

    monkeypatch.setattr(am, "models_roots", lambda: (models,))
    monkeypatch.setattr(am, "primary_models_dir", lambda: models)
    monkeypatch.setattr(am, "_hf_hub_download_file", fake_download)
    monkeypatch.setattr(am, "auto_download_enabled", lambda: True)
    am._downloaded.clear()

    path = am.ensure_anthill_file("demucs-openvino/htdemucs_v4/htdemucs_v4.xml")
    assert path == dest.resolve()
    assert calls == []

    am._downloaded.clear()
    kokoro = models / "kokoro" / "config.json"
    path2 = am.ensure_anthill_file("kokoro/config.json")
    assert path2 == kokoro.resolve()
    assert calls == ["kokoro/config.json"]


def test_auto_download_disabled_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(am, "models_roots", lambda: (models,))
    monkeypatch.setattr(am, "auto_download_enabled", lambda: False)
    with pytest.raises(FileNotFoundError, match="AH_ANTHILL_AUTO_DOWNLOAD"):
        am.ensure_anthill_file("kokoro/config.json")


def test_group_ready_uses_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    for rel in am.CHECKS["kokoro"]:
        path = models.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    monkeypatch.setattr(am, "models_roots", lambda: (models,))
    assert am.group_ready("kokoro")

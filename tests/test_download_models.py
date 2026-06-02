"""Tests for incremental tools/download_models.py behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from externals import anthill_models as am


def test_missing_group_names_partial_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    for rel in am.CHECKS["kokoro"]:
        path = models.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    monkeypatch.setattr(am, "models_roots", lambda: (models,))

    missing = am.missing_group_names("minimal")
    assert "kokoro" not in missing
    assert "demucs_openvino" in missing


def test_group_tree_prefix_multi_file() -> None:
    assert am.group_tree_prefix("qwen2_vl") == "qwen-vl/Qwen2-VL-2B-Instruct"
    assert am.group_tree_prefix("ocr_latin") == "ocr/PP-OCRv4/latin"


def test_group_tree_prefix_single_file() -> None:
    assert am.group_tree_prefix("flux_dev") == "FLUX.1-dev"


def _write_checks(models: Path, group: str) -> None:
    for rel in am.CHECKS[group]:
        path = models.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_download_anthill_skips_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    models = repo / "models"
    for group in am.PROFILE_GROUPS["minimal"]:
        _write_checks(models, group)

    import tools.download_models as dm

    monkeypatch.chdir(repo)
    monkeypatch.setattr(dm, "REPO_ROOT", repo)
    monkeypatch.setattr(dm, "MODELS_DIR", models)
    monkeypatch.setattr(am, "models_roots", lambda: (models,))

    with patch.object(dm, "_download_full_snapshot") as full, patch.object(
        dm, "_download_missing_groups"
    ) as inc:
        dm.download_anthill(profile="minimal", dry_run=False)
        full.assert_not_called()
        inc.assert_not_called()

    out = capsys.readouterr().out
    assert "all groups ready" in out


def test_download_anthill_incremental_when_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    models = repo / "models"
    _write_checks(models, "kokoro")

    import tools.download_models as dm

    monkeypatch.chdir(repo)
    monkeypatch.setattr(dm, "REPO_ROOT", repo)
    monkeypatch.setattr(dm, "MODELS_DIR", models)
    monkeypatch.setattr(am, "models_roots", lambda: (models,))

    with patch.object(dm, "_download_full_snapshot") as full, patch.object(
        dm, "_download_missing_groups"
    ) as inc:
        dm.download_anthill(profile="minimal", dry_run=False)
        full.assert_not_called()
        inc.assert_called_once_with(profile="minimal", dry_run=False)

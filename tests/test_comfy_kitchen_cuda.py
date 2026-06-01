"""comfy_kitchen cuda backend version gate (cu129 reports torch.version.cuda 12.9)."""

from __future__ import annotations

from comfy.quant_ops import _kitchen_cuda_version_unsupported


def test_cu129_not_unsupported():
    assert _kitchen_cuda_version_unsupported((12, 9)) is False


def test_cu128_not_unsupported():
    assert _kitchen_cuda_version_unsupported((12, 8)) is False


def test_cu130_supported():
    assert _kitchen_cuda_version_unsupported((13, 0)) is False


def test_old_cuda_unsupported():
    assert _kitchen_cuda_version_unsupported((12, 7)) is True
    assert _kitchen_cuda_version_unsupported((11, 8)) is True

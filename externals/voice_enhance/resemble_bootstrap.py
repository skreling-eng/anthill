"""Bootstrap resemble-enhance for inference without installing deepspeed."""

from __future__ import annotations

import pathlib
import sys
from types import ModuleType

_BOOTSTRAPPED = False


def install_omegaconf_path_fix() -> None:
    """Map YAML PosixPath tags to pathlib.Path (resemble hparams.yaml on Windows)."""
    import omegaconf._utils as oc_utils

    _orig = oc_utils.get_yaml_loader

    def _get_yaml_loader():
        loader = _orig()

        def _posix_as_path(loader, node):
            return pathlib.Path(*loader.construct_sequence(node))

        loader.add_constructor(
            "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
            _posix_as_path,
        )
        return loader

    oc_utils.get_yaml_loader = _get_yaml_loader


def install_deepspeed_stub() -> None:
    """resemble-enhance 0.0.1 imports deepspeed in train.py; stub it for inference."""
    if "deepspeed" in sys.modules:
        return

    ds = ModuleType("deepspeed")

    class DeepSpeedConfig:
        def __init__(self, *args, **kwargs):
            pass

    ds.DeepSpeedConfig = DeepSpeedConfig

    accelerator = ModuleType("deepspeed.accelerator")

    class _Accelerator:
        def communication_backend_name(self) -> str:
            return "nccl"

    accelerator.get_accelerator = lambda: _Accelerator()

    runtime = ModuleType("deepspeed.runtime")
    engine_mod = ModuleType("deepspeed.runtime.engine")

    class DeepSpeedEngine:
        pass

    engine_mod.DeepSpeedEngine = DeepSpeedEngine

    utils_mod = ModuleType("deepspeed.runtime.utils")
    utils_mod.clip_grad_norm_ = lambda *args, **kwargs: None

    sys.modules["deepspeed"] = ds
    sys.modules["deepspeed.accelerator"] = accelerator
    sys.modules["deepspeed.runtime"] = runtime
    sys.modules["deepspeed.runtime.engine"] = engine_mod
    sys.modules["deepspeed.runtime.utils"] = utils_mod


def install_hparams_load_fix() -> None:
    """Ignore extra keys in enhancer hparams.yaml (checkpoint vs package mismatch)."""
    from dataclasses import fields
    from pathlib import Path

    from omegaconf import OmegaConf

    import resemble_enhance.hparams as hp_mod

    if getattr(hp_mod.HParams, "_anthill_from_yaml", None) is not None:
        return

    @classmethod
    def from_yaml(cls, path: Path) -> hp_mod.HParams:
        hp_mod.logger.info(f"Reading hparams from {path}")
        loaded = OmegaConf.load(path)
        names = {f.name for f in fields(cls)}
        kwargs = {k: loaded[k] for k in names if k in loaded}
        return cls(**dict(OmegaConf.merge(cls(), OmegaConf.create(kwargs))))

    from_yaml._anthill_from_yaml = True  # type: ignore[attr-defined]
    hp_mod.HParams.from_yaml = from_yaml


def bootstrap_resemble_inference() -> None:
    """Apply patches required before importing resemble_enhance on this platform."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    install_omegaconf_path_fix()
    install_deepspeed_stub()
    import resemble_enhance.hparams  # noqa: F401 — load module before patching

    install_hparams_load_fix()
    _BOOTSTRAPPED = True

import importlib

run = importlib.import_module("externals.controlnet.run").run
__all__ = ["run"]

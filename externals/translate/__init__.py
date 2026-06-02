import importlib

run = importlib.import_module("externals.translate.run").run

__all__ = ["run"]

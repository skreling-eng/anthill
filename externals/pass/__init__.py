import importlib

run = importlib.import_module("externals.pass.run").run

__all__ = ["run"]

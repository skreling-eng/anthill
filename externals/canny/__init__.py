import importlib

run = importlib.import_module("externals.canny.run").run
__all__ = ["run"]

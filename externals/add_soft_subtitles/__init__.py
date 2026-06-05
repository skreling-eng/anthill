import importlib

run = importlib.import_module("externals.add_soft_subtitles.run").run

__all__ = ["run"]

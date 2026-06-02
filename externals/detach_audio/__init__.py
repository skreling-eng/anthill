import importlib

run = importlib.import_module("externals.detach_audio.run").run

__all__ = ["run"]

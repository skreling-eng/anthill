import importlib

run = importlib.import_module("externals.attach_audio.run").run

__all__ = ["run"]

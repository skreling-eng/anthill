import importlib

run = importlib.import_module("externals.audio_instruct.run").run

__all__ = ["run"]

import importlib

run = importlib.import_module("externals.video_thumbnailer.run").run

__all__ = ["run"]

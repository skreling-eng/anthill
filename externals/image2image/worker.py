"""Long-lived $image2image worker — keeps Qwen pipeline on GPU between jobs."""

from externals.comfy_inprocess.worker_main import worker_main
from externals.image2image.run import run
from externals.image2image.worker_cmd import log_worker_backend


def main() -> int:
    log_worker_backend()
    return worker_main(run)


if __name__ == "__main__":
    raise SystemExit(main())

"""Long-lived $image2video worker — keeps Wan MEGA pipeline on GPU between jobs."""

from externals.comfy_inprocess.worker_main import worker_main
from externals.image2video.run import run


def main() -> int:
    return worker_main(run)


if __name__ == "__main__":
    raise SystemExit(main())

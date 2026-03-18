"""
Pre-downloads all model weights into HF_HOME so the container is ready to
serve requests immediately on startup, with no download latency.

Usage:
    python scripts/preload_models.py

In Docker (e.g. to pre-populate a RunPod Network Volume):
    docker run --rm \
      -v /mnt/network-volume/hf-cache:/cache/huggingface \
      -e HF_HOME=/cache/huggingface \
      ghcr.io/<you>/image-generation:latest \
      python scripts/preload_models.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    from src.pipeline import get_pipeline, get_img2img_pipeline, get_controlnet_pipeline

    logger.info("=== Preloading FLUX base pipeline ===")
    get_pipeline()

    logger.info("=== Preloading img2img pipeline ===")
    get_img2img_pipeline()

    logger.info("=== Preloading ControlNet pipeline ===")
    get_controlnet_pipeline()

    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    logger.info("All models cached at: %s", cache_dir)
    logger.info("Preload complete.")


if __name__ == "__main__":
    main()

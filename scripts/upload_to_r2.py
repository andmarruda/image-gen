"""
One-time script to upload model weights from HuggingFace to Cloudflare R2.

Run this once from any machine (CPU is fine — no GPU needed):
    python scripts/upload_to_r2.py

Or inside a RunPod CPU pod with the env vars set:
    python scripts/upload_to_r2.py

The script downloads the model(s) specified by MODEL_ID (and optionally
CONTROLNET_MODEL_ID) from HuggingFace, then uploads all files to your R2 bucket.
Uploads are resume-safe — already-uploaded files are skipped.

Required env vars (.env or shell):
    R2_ACCOUNT_ID         your Cloudflare account ID
    R2_ACCESS_KEY_ID      R2 API token (Access Key ID)
    R2_SECRET_ACCESS_KEY  R2 API token (Secret Access Key)
    R2_BUCKET_NAME        target bucket name
    MODEL_ID              model to upload (default: black-forest-labs/FLUX.1-schnell)
    HF_TOKEN              required if MODEL_ID is gated (e.g. FLUX.1-dev)

Optional:
    UPLOAD_CONTROLNET     set to "true" to also upload the ControlNet adapter
    CONTROLNET_MODEL_ID   ControlNet model (default: InstantX/FLUX.1-dev-Controlnet-Canny)
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from src.r2_sync import upload

    model_id = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    hf_token = os.getenv("HF_TOKEN")

    logger.info("=== Uploading base model: %s ===", model_id)
    upload(model_id, hf_token=hf_token)

    if os.getenv("UPLOAD_CONTROLNET", "").lower() == "true":
        controlnet_id = os.getenv(
            "CONTROLNET_MODEL_ID", "InstantX/FLUX.1-dev-Controlnet-Canny"
        )
        logger.info("=== Uploading ControlNet adapter: %s ===", controlnet_id)
        upload(controlnet_id)

    logger.info("All uploads complete. Your R2 bucket is ready.")


if __name__ == "__main__":
    main()

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

_RUNPOD = os.getenv("RUNPOD_ENABLED", "").lower() == "true"
_PRELOAD = os.getenv("PRELOAD_MODELS", "").lower() == "true"


def _r2_download() -> None:
    """Download model weights from R2 to local disk before loading into GPU."""
    from src.r2_sync import download

    model_id = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    download(model_id)

    if os.getenv("DOWNLOAD_CONTROLNET", "true").lower() == "true":
        controlnet_id = os.getenv(
            "CONTROLNET_MODEL_ID", "InstantX/FLUX.1-dev-Controlnet-Canny"
        )
        download(controlnet_id)


def _preload() -> None:
    """Download and cache all model weights before serving any requests."""
    from src.pipeline import get_pipeline, get_img2img_pipeline, get_controlnet_pipeline

    if os.getenv("R2_ENABLED", "").lower() == "true":
        _r2_download()

    get_pipeline()
    get_img2img_pipeline()
    get_controlnet_pipeline()


if _RUNPOD:
    from src.runpod_handler import start

    if __name__ == "__main__":
        start()
else:
    from flask import Flask
    from src.routes import bp

    app = Flask(__name__)
    app.register_blueprint(bp)

    if _PRELOAD:
        _preload()

    if __name__ == "__main__":
        port = int(os.getenv("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)

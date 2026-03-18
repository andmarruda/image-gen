import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

_RUNPOD = os.getenv("RUNPOD_ENABLED", "").lower() == "true"
_PRELOAD = os.getenv("PRELOAD_MODELS", "").lower() == "true"


def _preload() -> None:
    """Download and cache all model weights before serving any requests."""
    from src.pipeline import get_pipeline, get_img2img_pipeline, get_controlnet_pipeline

    get_pipeline()
    get_img2img_pipeline()
    get_controlnet_pipeline()


if _RUNPOD:
    # ── RunPod serverless mode ─────────────────────────────────────────────────
    # Entrypoint: python app.py
    # The runpod SDK takes over the process loop; Flask is not used.
    from src.runpod_handler import start

    if __name__ == "__main__":
        start()
else:
    # ── Flask / Gunicorn mode (default) ───────────────────────────────────────
    # Entrypoint: gunicorn app:app
    from flask import Flask
    from src.routes import bp

    app = Flask(__name__)
    app.register_blueprint(bp)

    # Load models before accepting requests so the first call is not slow.
    # Only runs inside the Gunicorn worker process (not the master).
    if _PRELOAD:
        _preload()

    if __name__ == "__main__":
        port = int(os.getenv("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)

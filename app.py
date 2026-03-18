import logging
import os

from dotenv import load_dotenv
from src.startup import preload_models

load_dotenv()

logging.basicConfig(level=logging.INFO)

_RUNPOD = os.getenv("RUNPOD_ENABLED", "").lower() == "true"
_PRELOAD = os.getenv("PRELOAD_MODELS", "").lower() == "true"


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
        preload_models()

    if __name__ == "__main__":
        port = int(os.getenv("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)


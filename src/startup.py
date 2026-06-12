import logging
import os

logger = logging.getLogger(__name__)


def _truthy(name: str, default: str = "") -> bool:
    return os.getenv(name, default).lower() == "true"


def _download_r2_weights() -> None:
    """Download configured weights from R2 onto local disk before GPU loading."""
    from .r2_sync import download

    from .model_config import model_family, runtime_model_id

    download(runtime_model_id())

    if model_family() == "flux1" and _truthy("DOWNLOAD_CONTROLNET", "true"):
        controlnet_id = os.getenv(
            "CONTROLNET_MODEL_ID", "InstantX/FLUX.1-dev-Controlnet-Canny"
        )
        download(controlnet_id)


def preload_models() -> None:
    """
    Resolve model sources, download them when needed, and build pipelines during
    process startup so the first request does not pay the cold-start cost.
    """
    from .model_config import model_family
    from .pipeline import get_controlnet_pipeline, get_img2img_pipeline, get_pipeline

    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    logger.info("Cold start preload beginning. HF_HOME=%s", hf_home)

    if _truthy("R2_ENABLED"):
        _download_r2_weights()

    get_pipeline()
    if model_family() == "flux1":
        get_img2img_pipeline()
    if model_family() == "flux1" and _truthy("DOWNLOAD_CONTROLNET", "true"):
        get_controlnet_pipeline()

    logger.info("Cold start preload complete.")

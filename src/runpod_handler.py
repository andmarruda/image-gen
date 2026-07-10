import base64
import logging
import os
from pathlib import Path

from .conversations import prepare as prepare_conversation, save_revision
from .pipeline import generate
from .requests import generation_params
from .startup import preload_models
from .utils import image_to_bytes

logger = logging.getLogger(__name__)


def _configure_runpod_volume_defaults() -> None:
    volume = Path("/runpod-volume")
    if not volume.exists():
        return

    defaults = {
        "HF_HOME": ("/cache/huggingface", "/runpod-volume/huggingface"),
        "CONVERSATION_DIR": ("/data/conversations", "/runpod-volume/conversations"),
    }

    for name, (old_default, runpod_default) in defaults.items():
        current = os.getenv(name)
        if not current or current == old_default:
            os.environ[name] = runpod_default
            logger.info("Using RunPod Network Volume default: %s=%s", name, runpod_default)

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    if os.getenv("CONVERSATION_MEMORY_ENABLED", "true").lower() == "true":
        Path(os.environ["CONVERSATION_DIR"]).mkdir(parents=True, exist_ok=True)


def _encode(image, metadata: dict | None = None) -> dict:
    width, height = image.size
    payload = {
        "image": base64.b64encode(image_to_bytes(image)).decode("utf-8"),
        "format": "png",
        "width": width,
        "height": height,
    }
    payload.update(metadata or {})
    return payload


def handler(job: dict) -> dict:
    params = dict(job.get("input", {}))
    mode = params.pop("mode", "generate")
    if mode not in {"generate", "txt2img", "edit", "img2img"}:
        return {"error": "Valid modes: generate, txt2img, edit, img2img"}

    try:
        prepared, conversation_id = prepare_conversation(params)
        parsed = generation_params(prepared)
        if mode in {"edit", "img2img"} and not parsed["images"]:
            raise ValueError("'image' or 'images' is required")
        logger.info("RunPod job | mode=%s | references=%d", mode, len(parsed["images"]))
        image = generate(**parsed)
        metadata = {}
        if conversation_id:
            manifest = save_revision(conversation_id, image, parsed["prompt"])
            metadata = {
                "conversation_id": conversation_id,
                "revision_id": manifest["latest_revision_id"],
                "revision_count": len(manifest["revisions"]),
            }
        return _encode(image, metadata)
    except Exception as exc:
        logger.exception("Job failed")
        return {"error": str(exc)}


def start() -> None:
    import runpod

    _configure_runpod_volume_defaults()
    if os.getenv("PRELOAD_MODELS", "").lower() == "true":
        logger.info("PRELOAD_MODELS=true; loading models before accepting jobs.")
        preload_models()
    logger.info("Starting RunPod serverless worker.")
    runpod.serverless.start({"handler": handler})

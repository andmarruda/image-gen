import base64
import logging
import os

from .conversations import prepare as prepare_conversation, save_revision
from .pipeline import generate
from .requests import generation_params
from .startup import preload_models
from .utils import image_to_bytes

logger = logging.getLogger(__name__)


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

    if os.getenv("PRELOAD_MODELS", "").lower() == "true":
        preload_models()
    runpod.serverless.start({"handler": handler})

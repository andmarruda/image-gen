import logging
import hmac
import os

import torch
from flask import Blueprint, jsonify, request

from .conversations import delete as delete_conversation
from .conversations import get_manifest, prepare as prepare_conversation, save_revision
from .model_config import model_defaults, model_family, model_id
from .pipeline import generate as generate_image
from .pipeline import get_controlnet_pipeline, pipeline_is_loaded
from .requests import generation_params
from .utils import apply_canny, build_response, decode_input_image

logger = logging.getLogger(__name__)
bp = Blueprint("api", __name__)


@bp.before_request
def authenticate():
    expected = os.getenv("API_KEY")
    if not expected or request.endpoint == "api.health":
        return None
    supplied = (
        request.headers.get("X-API-Key")
        or request.headers.get("Authorization", "").removeprefix("Bearer ")
        or ""
    )
    if not hmac.compare_digest(supplied, expected):
        return jsonify({"error": "unauthorized"}), 401
    return None


def _body_with_uploads() -> dict:
    if not request.content_type or "multipart/form-data" not in request.content_type:
        return request.get_json(silent=True) or {}

    data = request.form.to_dict()
    files = request.files.getlist("images") or request.files.getlist("image")
    if files:
        data["images"] = [file.read() for file in files]
    return data


@bp.errorhandler(ValueError)
def invalid_request(exc):
    return jsonify({"error": str(exc)}), 400


@bp.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": model_id(),
            "family": model_family(),
            "loaded": pipeline_is_loaded(),
            "cuda": torch.cuda.is_available(),
            "defaults": model_defaults(),
        }
    )


def _generate_with_memory(data: dict):
    prepared, conversation_id = prepare_conversation(data)
    params = generation_params(prepared)
    logger.info(
        "generate | conversation=%s | references=%d | prompt: %.80s",
        conversation_id,
        len(params["images"]),
        params["prompt"],
    )
    image = generate_image(**params)
    metadata = {}
    if conversation_id:
        manifest = save_revision(conversation_id, image, params["prompt"])
        metadata = {
            "conversation_id": conversation_id,
            "revision_id": manifest["latest_revision_id"],
            "revision_count": len(manifest["revisions"]),
        }
    width, height = image.size
    return build_response(image, params["prompt"], width, height, metadata)


@bp.post("/generate")
def generate():
    return _generate_with_memory(_body_with_uploads())


@bp.post("/generate/edit")
@bp.post("/generate/img2img")
def edit():
    data = _body_with_uploads()
    prepared, conversation_id = prepare_conversation(data)
    if not prepared.get("images") and not prepared.get("image"):
        raise ValueError("'image' or 'images' is required")
    if conversation_id:
        prepared["conversation_id"] = conversation_id
    return _generate_with_memory(prepared)


@bp.get("/conversations/<conversation_id>")
def conversation(conversation_id: str):
    manifest = get_manifest(conversation_id)
    if manifest is None:
        return jsonify({"error": "conversation not found"}), 404
    return jsonify(manifest)


@bp.delete("/conversations/<conversation_id>")
def remove_conversation(conversation_id: str):
    if not delete_conversation(conversation_id):
        return jsonify({"error": "conversation not found"}), 404
    return jsonify({"deleted": True, "conversation_id": conversation_id})


@bp.post("/generate/controlnet")
def controlnet():
    if model_family().startswith("flux2"):
        return jsonify(
            {"error": "ControlNet is FLUX.1-only. Use /generate/edit with FLUX.2 reference images."}
        ), 409

    data = _body_with_uploads()
    prompt = data.get("prompt")
    images = data.get("images") or ([data["image"]] if data.get("image") else [])
    if not prompt:
        raise ValueError("'prompt' is required")
    if not images:
        raise ValueError("'image' is required")

    control = apply_canny(
        decode_input_image(images[0]),
        int(data.get("canny_low_threshold", 100)),
        int(data.get("canny_high_threshold", 200)),
    )
    seed = data.get("seed")
    generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    result = get_controlnet_pipeline()(
        prompt=prompt,
        control_image=control,
        controlnet_conditioning_scale=float(data.get("controlnet_conditioning_scale", 0.7)),
        num_inference_steps=int(data.get("num_inference_steps", 28)),
        guidance_scale=float(data.get("guidance_scale", 3.5)),
        width=int(data.get("width", 1024)),
        height=int(data.get("height", 1024)),
        generator=generator,
    )
    width, height = result.images[0].size
    return build_response(result.images[0], prompt, width, height)

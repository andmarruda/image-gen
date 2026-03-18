import base64
import logging
import os

import torch

from .pipeline import get_pipeline, get_img2img_pipeline, get_controlnet_pipeline
from .utils import decode_input_image, apply_canny, image_to_bytes

logger = logging.getLogger(__name__)


def _make_generator(seed) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator().manual_seed(int(seed))


def _encode(image) -> dict:
    w, h = image.size
    return {
        "image": base64.b64encode(image_to_bytes(image)).decode("utf-8"),
        "format": "png",
        "width": w,
        "height": h,
    }


def _txt2img(params: dict) -> dict:
    prompt = params.get("prompt")
    if not prompt:
        return {"error": "'prompt' is required"}

    result = get_pipeline()(
        prompt=prompt,
        num_inference_steps=int(params.get("num_inference_steps", 4)),
        guidance_scale=float(params.get("guidance_scale", 0.0)),
        width=int(params.get("width", 1024)),
        height=int(params.get("height", 1024)),
        generator=_make_generator(params.get("seed")),
    )
    return _encode(result.images[0])


def _img2img(params: dict) -> dict:
    prompt = params.get("prompt")
    image_b64 = params.get("image")
    if not prompt:
        return {"error": "'prompt' is required"}
    if not image_b64:
        return {"error": "'image' (base64) is required"}

    result = get_img2img_pipeline()(
        prompt=prompt,
        image=decode_input_image(image_b64),
        strength=float(params.get("strength", 0.75)),
        num_inference_steps=int(params.get("num_inference_steps", 4)),
        guidance_scale=float(params.get("guidance_scale", 0.0)),
        generator=_make_generator(params.get("seed")),
    )
    return _encode(result.images[0])


def _controlnet(params: dict) -> dict:
    prompt = params.get("prompt")
    image_b64 = params.get("image")
    if not prompt:
        return {"error": "'prompt' is required"}
    if not image_b64:
        return {"error": "'image' (base64) is required"}

    ref = decode_input_image(image_b64)
    control = apply_canny(
        ref,
        int(params.get("canny_low_threshold", 100)),
        int(params.get("canny_high_threshold", 200)),
    )

    result = get_controlnet_pipeline()(
        prompt=prompt,
        control_image=control,
        controlnet_conditioning_scale=float(params.get("controlnet_conditioning_scale", 0.7)),
        num_inference_steps=int(params.get("num_inference_steps", 28)),
        guidance_scale=float(params.get("guidance_scale", 3.5)),
        width=int(params.get("width", 1024)),
        height=int(params.get("height", 1024)),
        generator=_make_generator(params.get("seed")),
    )
    return _encode(result.images[0])


_MODES = {
    "txt2img": _txt2img,
    "img2img": _img2img,
    "controlnet": _controlnet,
}


def handler(job: dict) -> dict:
    params = job.get("input", {})
    mode = params.get("mode", "txt2img")

    fn = _MODES.get(mode)
    if fn is None:
        return {"error": f"Unknown mode '{mode}'. Valid modes: {list(_MODES.keys())}"}

    logger.info("RunPod job | mode=%s | prompt: %.80s", mode, params.get("prompt", ""))

    try:
        return fn(params)
    except Exception as exc:
        logger.exception("Job failed")
        return {"error": str(exc)}


def start() -> None:
    import runpod

    # Load models before the serverless loop starts so every job hits
    # warm weights — no download or load time inside the handler.
    if os.getenv("PRELOAD_MODELS", "").lower() == "true":
        logger.info("Preloading models before accepting jobs ...")
        get_pipeline()
        get_img2img_pipeline()
        get_controlnet_pipeline()
        logger.info("Preload complete.")

    runpod.serverless.start({"handler": handler})

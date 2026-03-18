import logging

import torch
from flask import Blueprint, jsonify, request

from .pipeline import get_pipeline, get_img2img_pipeline, get_controlnet_pipeline
from .utils import build_response, decode_input_image, apply_canny

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)


@bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@bp.post("/generate")
def generate():
    body = request.get_json(silent=True) or {}

    prompt = body.get("prompt")
    if not prompt:
        return jsonify({"error": "'prompt' is required"}), 400

    num_steps: int = int(body.get("num_inference_steps", 4))
    guidance_scale: float = float(body.get("guidance_scale", 0.0))
    width: int = int(body.get("width", 1024))
    height: int = int(body.get("height", 1024))
    seed = body.get("seed")

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))

    logger.info("Generating image | prompt: %.80s", prompt)

    pipe = get_pipeline()
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    return build_response(result.images[0], prompt, width, height)


@bp.post("/generate/img2img")
def generate_img2img():
    """
    Accepts a reference image + prompt and generates a new image based on both.

    Body (JSON):
        prompt          str   required
        image           str   required  — base64-encoded PNG/JPG

    Body (multipart/form-data):
        prompt          str   required
        image           file  required

    Optional:
        strength              float  0.0–1.0  (default 0.75)
        num_inference_steps   int    (default 4)
        guidance_scale        float  (default 0.0)
        seed                  int
    """
    # ── parse input ──────────────────────────────────────────────────────────
    if request.content_type and "multipart/form-data" in request.content_type:
        prompt = request.form.get("prompt")
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "'image' file is required"}), 400
        raw_image = decode_input_image(file.read())
        strength = float(request.form.get("strength", 0.75))
        num_steps = int(request.form.get("num_inference_steps", 4))
        guidance_scale = float(request.form.get("guidance_scale", 0.0))
        seed = request.form.get("seed")
    else:
        body = request.get_json(silent=True) or {}
        prompt = body.get("prompt")
        image_b64 = body.get("image")
        if not image_b64:
            return jsonify({"error": "'image' (base64) is required"}), 400
        raw_image = decode_input_image(image_b64)
        strength = float(body.get("strength", 0.75))
        num_steps = int(body.get("num_inference_steps", 4))
        guidance_scale = float(body.get("guidance_scale", 0.0))
        seed = body.get("seed")

    if not prompt:
        return jsonify({"error": "'prompt' is required"}), 400

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))

    logger.info("img2img | strength=%.2f | prompt: %.80s", strength, prompt)

    pipe = get_img2img_pipeline()
    result = pipe(
        prompt=prompt,
        image=raw_image,
        strength=strength,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    width, height = result.images[0].size
    return build_response(result.images[0], prompt, width, height)


@bp.post("/generate/controlnet")
def generate_controlnet():
    """
    Image-guided generation using ControlNet (Canny edges).

    Extracts the structural edges from the reference image and uses them
    as a hard conditioning signal, so the output preserves the composition
    of the original while following the text prompt for style and content.

    Body (JSON):
        prompt                      str    required
        image                       str    required — base64-encoded PNG/JPG

    Body (multipart/form-data):
        prompt                      str    required
        image                       file   required

    Optional:
        controlnet_conditioning_scale  float  0.0–1.0  (default 0.7)
        canny_low_threshold            int              (default 100)
        canny_high_threshold           int              (default 200)
        num_inference_steps            int              (default 28)
        guidance_scale                 float            (default 3.5)
        width                          int              (default 1024)
        height                         int              (default 1024)
        seed                           int
    """
    if request.content_type and "multipart/form-data" in request.content_type:
        prompt = request.form.get("prompt")
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "'image' file is required"}), 400
        raw_image = decode_input_image(file.read())
        conditioning_scale = float(request.form.get("controlnet_conditioning_scale", 0.7))
        canny_low = int(request.form.get("canny_low_threshold", 100))
        canny_high = int(request.form.get("canny_high_threshold", 200))
        num_steps = int(request.form.get("num_inference_steps", 28))
        guidance_scale = float(request.form.get("guidance_scale", 3.5))
        width = int(request.form.get("width", 1024))
        height = int(request.form.get("height", 1024))
        seed = request.form.get("seed")
    else:
        body = request.get_json(silent=True) or {}
        prompt = body.get("prompt")
        image_b64 = body.get("image")
        if not image_b64:
            return jsonify({"error": "'image' (base64) is required"}), 400
        raw_image = decode_input_image(image_b64)
        conditioning_scale = float(body.get("controlnet_conditioning_scale", 0.7))
        canny_low = int(body.get("canny_low_threshold", 100))
        canny_high = int(body.get("canny_high_threshold", 200))
        num_steps = int(body.get("num_inference_steps", 28))
        guidance_scale = float(body.get("guidance_scale", 3.5))
        width = int(body.get("width", 1024))
        height = int(body.get("height", 1024))
        seed = body.get("seed")

    if not prompt:
        return jsonify({"error": "'prompt' is required"}), 400

    control_image = apply_canny(raw_image, canny_low, canny_high)

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))

    logger.info(
        "controlnet | conditioning=%.2f | prompt: %.80s",
        conditioning_scale,
        prompt,
    )

    pipe = get_controlnet_pipeline()
    result = pipe(
        prompt=prompt,
        control_image=control_image,
        controlnet_conditioning_scale=conditioning_scale,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    return build_response(result.images[0], prompt, width, height)

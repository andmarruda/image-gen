from typing import Any

from .model_config import model_defaults
from .utils import decode_input_image


def generation_params(data: dict[str, Any]) -> dict[str, Any]:
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' is required")

    defaults = model_defaults()
    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    if width < 256 or height < 256 or width > 2048 or height > 2048:
        raise ValueError("width and height must be between 256 and 2048")
    if width % 16 or height % 16:
        raise ValueError("width and height must be divisible by 16")

    num_steps = int(data.get("num_inference_steps", defaults["num_inference_steps"]))
    guidance_scale = float(data.get("guidance_scale", defaults["guidance_scale"]))
    strength = float(data.get("strength", 0.75))
    if num_steps < 1 or num_steps > 100:
        raise ValueError("num_inference_steps must be between 1 and 100")
    if guidance_scale < 0 or guidance_scale > 20:
        raise ValueError("guidance_scale must be between 0 and 20")
    if strength < 0 or strength > 1:
        raise ValueError("strength must be between 0 and 1")

    raw_images = data.get("images")
    if raw_images is None and data.get("image") is not None:
        raw_images = [data["image"]]
    if raw_images is not None and not isinstance(raw_images, list):
        raise ValueError("'images' must be a list of base64-encoded images")
    if raw_images and len(raw_images) > 4:
        raise ValueError("at most 4 reference images are supported")

    return {
        "prompt": prompt.strip(),
        "images": [
            image.convert("RGB") if hasattr(image, "convert") else decode_input_image(image)
            for image in raw_images or []
        ],
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "seed": int(data["seed"]) if data.get("seed") is not None else None,
        "strength": strength,
    }

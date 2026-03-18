import io
import base64

import cv2
import numpy as np
from PIL import Image
from flask import request, Response, jsonify


def decode_input_image(source: str | bytes) -> Image.Image:
    """Accept a base64 string or raw bytes and return a PIL Image."""
    if isinstance(source, str):
        source = base64.b64decode(source)
    return Image.open(io.BytesIO(source)).convert("RGB")


def apply_canny(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """Convert a PIL image to a Canny edge map (RGB) for ControlNet conditioning."""
    arr = np.array(image.convert("RGB"))
    edges = cv2.Canny(arr, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def wants_raw_bytes() -> bool:
    accept = request.headers.get("Accept", "")
    x_format = request.headers.get("X-Response-Format", "")
    return "image/" in accept or x_format.lower() == "bytes"


def image_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def build_response(image: Image.Image, prompt: str, width: int, height: int):
    raw = image_to_bytes(image)

    if wants_raw_bytes():
        return Response(
            raw,
            mimetype="image/png",
            headers={"Content-Disposition": "inline; filename=generated.png"},
        )

    return jsonify(
        {
            "image": base64.b64encode(raw).decode("utf-8"),
            "format": "png",
            "prompt": prompt,
            "width": width,
            "height": height,
        }
    )

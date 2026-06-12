import io
import logging
import os
import threading
from typing import Any

import requests
import torch

from .model_config import model_defaults, model_family, model_id, runtime_model_id

logger = logging.getLogger(__name__)

_pipeline: Any | None = None
_img2img: Any | None = None
_controlnet: Any | None = None
_inference_lock = threading.Lock()


def pipeline_is_loaded() -> bool:
    return _pipeline is not None


def _resolve_model_source(value: str, hf_token: str | None) -> tuple[str, dict]:
    if os.getenv("R2_ENABLED", "").lower() == "true":
        from .r2_sync import local_path

        cached = local_path(value)
        if cached:
            logger.info("Loading %s from R2 local cache: %s", value, cached)
            return cached, {}

    kwargs: dict = {}
    if hf_token:
        kwargs["token"] = hf_token
    return value, kwargs


def _remote_text_encoder(prompts: list[str]) -> torch.Tensor:
    url = os.getenv(
        "FLUX2_REMOTE_TEXT_ENCODER_URL",
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
    )
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required by FLUX2_TEXT_ENCODER_MODE=remote")

    response = requests.post(
        url,
        json={"prompt": prompts},
        headers={"Authorization": f"Bearer {token}"},
        timeout=int(os.getenv("FLUX2_TEXT_ENCODER_TIMEOUT", "120")),
    )
    response.raise_for_status()
    return torch.load(io.BytesIO(response.content), map_location="cpu", weights_only=True).to("cuda")


def _load_flux2(device: str, dtype: torch.dtype):
    family = model_family()
    hf_token = os.getenv("HF_TOKEN")
    source_id = runtime_model_id()
    kwargs: dict[str, Any] = {"torch_dtype": dtype}

    if family == "flux2-dev" and os.getenv("FLUX2_DEV_4BIT", "true").lower() == "true":
        if os.getenv("FLUX2_TEXT_ENCODER_MODE", "remote").lower() == "remote":
            kwargs["text_encoder"] = None

    source, source_kwargs = _resolve_model_source(source_id, hf_token)
    kwargs.update(source_kwargs)

    if family == "flux2-klein":
        from diffusers import Flux2KleinPipeline

        pipe = Flux2KleinPipeline.from_pretrained(source, **kwargs)
    else:
        from diffusers import Flux2Pipeline

        pipe = Flux2Pipeline.from_pretrained(source, **kwargs)

    if device == "cuda":
        if os.getenv("MODEL_CPU_OFFLOAD", "false").lower() == "true":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

    return pipe


def _load_flux1(device: str, dtype: torch.dtype):
    from diffusers import FluxPipeline

    source, kwargs = _resolve_model_source(model_id(), os.getenv("HF_TOKEN"))
    pipe = FluxPipeline.from_pretrained(source, torch_dtype=dtype, **kwargs)
    if device == "cuda" and os.getenv("MODEL_CPU_OFFLOAD", "false").lower() == "true":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info("Loading %s (%s) on %s ...", model_id(), model_family(), device)

    _pipeline = _load_flux2(device, dtype) if model_family().startswith("flux2") else _load_flux1(device, dtype)
    logger.info("Model loaded.")
    return _pipeline


def generate(
    *,
    prompt: str,
    images: list | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    strength: float = 0.75,
):
    defaults = model_defaults()
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed) if seed is not None else None
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps or defaults["num_inference_steps"],
        "guidance_scale": guidance_scale if guidance_scale is not None else defaults["guidance_scale"],
        "width": width,
        "height": height,
        "generator": generator,
    }

    if model_family().startswith("flux2"):
        if images:
            kwargs["image"] = images
        if model_family() == "flux2-dev" and os.getenv("FLUX2_TEXT_ENCODER_MODE", "remote").lower() == "remote":
            kwargs["prompt_embeds"] = _remote_text_encoder([prompt])
            kwargs.pop("prompt")
    elif images:
        kwargs["image"] = images[0]
        kwargs["strength"] = strength
        pipe = get_img2img_pipeline()
        with _inference_lock:
            return pipe(**kwargs).images[0]

    with _inference_lock:
        return get_pipeline()(**kwargs).images[0]


def get_img2img_pipeline():
    global _img2img
    if model_family().startswith("flux2"):
        return get_pipeline()
    if _img2img is None:
        from diffusers import FluxImg2ImgPipeline

        _img2img = FluxImg2ImgPipeline.from_pipe(get_pipeline())
    return _img2img


def get_controlnet_pipeline():
    global _controlnet
    if model_family().startswith("flux2"):
        raise RuntimeError("ControlNet is only available for FLUX.1; use FLUX.2 reference editing")
    if _controlnet is None:
        from diffusers import FluxControlNetModel, FluxControlNetPipeline

        controlnet_id = os.getenv("CONTROLNET_MODEL_ID", "InstantX/FLUX.1-dev-Controlnet-Canny")
        source, kwargs = _resolve_model_source(controlnet_id, None)
        controlnet = FluxControlNetModel.from_pretrained(source, torch_dtype=torch.bfloat16, **kwargs)
        _controlnet = FluxControlNetPipeline.from_pipe(get_pipeline(), controlnet=controlnet)
    return _controlnet

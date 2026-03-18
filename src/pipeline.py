import os
import logging

import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxControlNetPipeline, FluxControlNetModel

logger = logging.getLogger(__name__)

_txt2img: FluxPipeline | None = None
_img2img: FluxImg2ImgPipeline | None = None
_controlnet: FluxControlNetPipeline | None = None


def get_pipeline() -> FluxPipeline:
    global _txt2img
    if _txt2img is not None:
        return _txt2img

    model_id = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    hf_token = os.getenv("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    logger.info("Loading model %s on %s ...", model_id, device)

    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    _txt2img = FluxPipeline.from_pretrained(model_id, **kwargs).to(device)

    if device == "cuda":
        _txt2img.enable_model_cpu_offload()

    logger.info("Model loaded.")
    return _txt2img


def get_img2img_pipeline() -> FluxImg2ImgPipeline:
    """Reuses the already-loaded txt2img weights — no extra VRAM."""
    global _img2img
    if _img2img is not None:
        return _img2img

    txt2img = get_pipeline()

    logger.info("Building img2img pipeline from loaded weights ...")
    _img2img = FluxImg2ImgPipeline.from_pipe(txt2img)
    logger.info("img2img pipeline ready.")
    return _img2img


def get_controlnet_pipeline() -> FluxControlNetPipeline:
    """
    Loads the ControlNet model and wires it into the base FLUX weights.
    The transformer, VAE and text encoders are shared — only the small
    ControlNet adapter is loaded on top.
    """
    global _controlnet
    if _controlnet is not None:
        return _controlnet

    txt2img = get_pipeline()
    controlnet_id = os.getenv("CONTROLNET_MODEL_ID", "InstantX/FLUX.1-dev-Controlnet-Canny")
    dtype = txt2img.dtype if hasattr(txt2img, "dtype") else torch.bfloat16

    logger.info("Loading ControlNet adapter %s ...", controlnet_id)
    controlnet = FluxControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)

    _controlnet = FluxControlNetPipeline.from_pipe(txt2img, controlnet=controlnet)
    logger.info("ControlNet pipeline ready.")
    return _controlnet

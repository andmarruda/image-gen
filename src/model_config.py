import os

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-dev"
FLUX2_DEV_4BIT_ID = "diffusers/FLUX.2-dev-bnb-4bit"


def model_id() -> str:
    return os.getenv("MODEL_ID", DEFAULT_MODEL_ID)


def model_family(value: str | None = None) -> str:
    value = (value or model_id()).lower()
    if "flux.2-klein" in value:
        return "flux2-klein"
    if "flux.2" in value:
        return "flux2-dev"
    return "flux1"


def model_defaults() -> dict[str, float | int]:
    family = model_family()
    if family == "flux2-dev":
        return {"num_inference_steps": 28, "guidance_scale": 4.0}
    if family == "flux2-klein":
        return {"num_inference_steps": 4, "guidance_scale": 1.0}
    if "schnell" in model_id().lower():
        return {"num_inference_steps": 4, "guidance_scale": 0.0}
    return {"num_inference_steps": 28, "guidance_scale": 3.5}


def runtime_model_id() -> str:
    if model_family() == "flux2-dev" and os.getenv("FLUX2_DEV_4BIT", "true").lower() == "true":
        return os.getenv("FLUX2_DEV_QUANTIZED_MODEL_ID", FLUX2_DEV_4BIT_ID)
    return model_id()

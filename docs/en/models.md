# Models: FLUX.2 and FLUX.1

`MODEL_ID` selects the model at startup. Each instance loads one model at a
time; change the variable and restart to switch.

## Comparison

| Model | Detected family | Default steps/guidance | Suggested use |
|---|---|---|---|
| `black-forest-labs/FLUX.2-dev` | `flux2-dev` | `28` / `4.0` | Study, quality, editing |
| `black-forest-labs/FLUX.2-klein-4B` | `flux2-klein` | `4` / `1.0` | Lower cost and latency |
| `black-forest-labs/FLUX.1-schnell` | `flux1` | `4` / `0.0` | Fast FLUX.1 |
| `black-forest-labs/FLUX.1-dev` | `flux1` | `28` / `3.5` | Higher-quality FLUX.1 |

Always verify each checkpoint license. Open FLUX.2 Dev weights are intended for
non-commercial/study use under BFL terms.

## Default FLUX.2 Dev

Recommended project configuration:

```env
MODEL_ID=black-forest-labs/FLUX.2-dev
HF_TOKEN=hf_...
FLUX2_DEV_4BIT=true
FLUX2_DEV_QUANTIZED_MODEL_ID=diffusers/FLUX.2-dev-bnb-4bit
FLUX2_TEXT_ENCODER_MODE=remote
MODEL_CPU_OFFLOAD=false
```

The runtime loads the quantized checkpoint and requests embeddings from the
service configured by `FLUX2_REMOTE_TEXT_ENCODER_URL`.

Consequences:

- Requires `HF_TOKEN`.
- Depends on network access for the remote encoder.
- Prompts are sent to the remote encoder service.
- This profile targets GPUs with approximately 24 GB VRAM.

Use `FLUX2_TEXT_ENCODER_MODE=local` only with enough memory for the local encoder.

## FLUX.2 Klein

```env
MODEL_ID=black-forest-labs/FLUX.2-klein-4B
MODEL_CPU_OFFLOAD=false
```

Use it to lower cost and latency. It keeps generation, editing, and multiple
references through the same API.

## FLUX.1

```env
MODEL_ID=black-forest-labs/FLUX.1-schnell
```

or:

```env
MODEL_ID=black-forest-labs/FLUX.1-dev
HF_TOKEN=hf_...
```

With FLUX.1:

- Text-to-image uses `FluxPipeline`.
- Editing uses `FluxImg2ImgPipeline`.
- Only the first reference is used for img2img.
- `strength` controls distance from the reference.
- Canny ControlNet becomes available.

## FLUX.1 ControlNet

```env
MODEL_ID=black-forest-labs/FLUX.1-dev
CONTROLNET_MODEL_ID=InstantX/FLUX.1-dev-Controlnet-Canny
DOWNLOAD_CONTROLNET=true
```

Use `POST /generate/controlnet` to preserve edges/composition. FLUX.2 does not
use this route; native reference editing is available at `/generate/edit`.

## FLUX.1/2 as a guardrail?

FLUX.1 or FLUX.2 may be experimental stages for generation, editing, or visual
consistency checks, but neither is a reliable safety guardrail. Do not use a
generator alone to decide whether content is safe.

See [Security and guardrails](guardrails.md) for a proper architecture.


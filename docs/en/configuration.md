# Configuration Reference

Variables may be set through `.env`, Docker, RunPod, or the process environment.

## Runtime and API

| Variable | Default | Description |
|---|---|---|
| `RUNPOD_ENABLED` | `false` | Use the RunPod handler instead of HTTP |
| `PRELOAD_MODELS` | `false` | Load weights and pipelines during startup |
| `PORT` | `5000` | HTTP port |
| `API_KEY` | unset | Protects every route except `/health` |
| `MAX_REQUEST_MB` | `32` | Maximum HTTP request size |

## Model

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `black-forest-labs/FLUX.2-dev` | Configured model |
| `HF_TOKEN` | unset | Token for gated models and remote encoder |
| `HF_HOME` | `/cache/huggingface` in Docker; use `/runpod-volume/huggingface` on RunPod Serverless | Hugging Face cache |
| `MODEL_CPU_OFFLOAD` | `false` | Move components between CPU/GPU to reduce VRAM |
| `FLUX2_DEV_4BIT` | `true` | Use the quantized Dev checkpoint |
| `FLUX2_DEV_QUANTIZED_MODEL_ID` | `diffusers/FLUX.2-dev-bnb-4bit` | Runtime 4-bit checkpoint |
| `FLUX2_TEXT_ENCODER_MODE` | `remote` in `.env.example` | `remote` removes the local encoder |
| `FLUX2_REMOTE_TEXT_ENCODER_URL` | Hugging Face endpoint configured in code | Remote embedding service |
| `FLUX2_TEXT_ENCODER_TIMEOUT` | `120` | Remote encoder timeout in seconds |

When `FLUX2_TEXT_ENCODER_MODE=remote`, prompts are sent to the remote service.

## Conversation memory

| Variable | Default | Description |
|---|---|---|
| `CONVERSATION_MEMORY_ENABLED` | `true` | Enable visual memory |
| `CONVERSATION_DIR` | `/data/conversations`; use `/runpod-volume/conversations` on RunPod Serverless | Images and manifests directory |
| `CONVERSATION_MAX_REVISIONS` | `10` | Revisions retained per conversation |

## FLUX.1 and ControlNet

| Variable | Default | Description |
|---|---|---|
| `CONTROLNET_MODEL_ID` | `InstantX/FLUX.1-dev-Controlnet-Canny` | ControlNet adapter |
| `DOWNLOAD_CONTROLNET` | `true` | Load/download ControlNet during FLUX.1 preload |
| `UPLOAD_CONTROLNET` | `false` | Upload ControlNet to R2 through the script |

These options are ignored for FLUX.2.

## Cloudflare R2

| Variable | Default | Description |
|---|---|---|
| `R2_ENABLED` | `false` | Enable model-weight download through R2 |
| `R2_ACCOUNT_ID` | required with R2 | Cloudflare account ID |
| `R2_ACCESS_KEY_ID` | required with R2 | Access key |
| `R2_SECRET_ACCESS_KEY` | required with R2 | Secret key |
| `R2_BUCKET_NAME` | required with R2 | Model-weight bucket |

## Ready-to-use profiles

FLUX.2 Dev for study:

```env
MODEL_ID=black-forest-labs/FLUX.2-dev
HF_TOKEN=hf_...
FLUX2_DEV_4BIT=true
FLUX2_TEXT_ENCODER_MODE=remote
MODEL_CPU_OFFLOAD=false
```

Lower-cost FLUX.2 Klein:

```env
MODEL_ID=black-forest-labs/FLUX.2-klein-4B
MODEL_CPU_OFFLOAD=false
```

FLUX.1 Dev with ControlNet:

```env
MODEL_ID=black-forest-labs/FLUX.1-dev
HF_TOKEN=hf_...
DOWNLOAD_CONTROLNET=true
CONTROLNET_MODEL_ID=InstantX/FLUX.1-dev-Controlnet-Canny
```

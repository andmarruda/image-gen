# Deployment and Operations

## Local Docker

```bash
cp .env.example .env
docker compose up --build
```

The service uses one worker and one thread because the pipeline occupies the GPU
and should not receive concurrent inference calls in one process. The code also
uses an inference lock.

## Published image

`.github/workflows/docker-build.yml`:

- Runs configuration tests.
- Publishes to GitHub Container Registry.
- Optionally publishes to Docker Hub.
- Uses GitHub Actions build cache.

For Docker Hub, configure:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## RunPod Serverless

### Build and endpoint setup

1. Build and publish the Docker image through GitHub Actions, or push an
   equivalent image to a registry that RunPod can pull.
2. Create a RunPod Serverless endpoint using the published image, for example
   `ghcr.io/YOUR_USER/YOUR_REPO:latest`.
3. Select a CUDA-capable GPU with enough VRAM for the configured model. The
   default FLUX.2 Dev 4-bit profile is intended for roughly 24 GB+ GPUs.
4. Attach a Network Volume to the endpoint so model weights and conversation
   memory survive cold starts.
5. Add the environment variables below in the RunPod endpoint settings.
6. Send jobs with the `input` object documented in
   [API reference: RunPod Serverless](api.md#runpod-serverless).

Typical variables:

```env
RUNPOD_ENABLED=true
MODEL_ID=black-forest-labs/FLUX.2-dev
HF_TOKEN=hf_...
FLUX2_DEV_4BIT=true
FLUX2_TEXT_ENCODER_MODE=remote
MODEL_CPU_OFFLOAD=false
PRELOAD_MODELS=false
HF_HOME=/runpod-volume/huggingface
CONVERSATION_DIR=/runpod-volume/conversations
```

Required variables:

- `RUNPOD_ENABLED=true`: starts the RunPod serverless worker instead of the HTTP
  Gunicorn server.
- `MODEL_ID`: selects the model to load.
- `HF_TOKEN`: required for gated Hugging Face models and for the default FLUX.2
  remote text encoder.
- `HF_HOME=/runpod-volume/huggingface`: persists the Hugging Face/model cache.
- `CONVERSATION_DIR=/runpod-volume/conversations`: persists visual memory and
  revision history.

Recommended variables:

- `FLUX2_DEV_4BIT=true`: uses the quantized FLUX.2 Dev checkpoint.
- `FLUX2_TEXT_ENCODER_MODE=remote`: avoids loading the text encoder locally.
- `MODEL_CPU_OFFLOAD=false`: fastest option when the GPU has enough VRAM.
- `PRELOAD_MODELS=false`: good serverless default; the first request loads the
  model.
- `CONVERSATION_MEMORY_ENABLED=true`: keeps generated revisions available for
  iterative edits.
- `CONVERSATION_MAX_REVISIONS=10`: caps stored revisions per conversation.

Attach a Network Volume to the endpoint. In RunPod Serverless, that volume is
mounted inside the worker at `/runpod-volume`. Use:

- `/runpod-volume/huggingface` for weights/Hugging Face cache.
- `/runpod-volume/conversations` for memory.

For serverless, `PRELOAD_MODELS=false` starts the handler quickly, but the first
request pays model-loading time. Persistent pods usually benefit from
`PRELOAD_MODELS=true`.

Do not bake model weights into the Docker image. Keep the image focused on code
and dependencies, then let `HF_HOME` or the optional R2 cache provide the
weights at runtime.

## Preload weights

```bash
python scripts/preload_models.py
```

Inside a container with a volume:

```bash
docker run --rm --gpus all \
  -v /your/cache:/cache/huggingface \
  --env-file .env \
  your-image \
  python scripts/preload_models.py
```

## Cloudflare R2 cache

The project can copy weights from Hugging Face to R2 and download them on cold
start.

```env
R2_ENABLED=true
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=flux-models
```

Initial upload:

```bash
python scripts/upload_to_r2.py
```

For FLUX.2 Dev 4-bit, the script uploads the actual runtime model selected by
`FLUX2_DEV_QUANTIZED_MODEL_ID`.

## Scaling

- Run one replica/worker per GPU.
- Put a queue or load balancer before workers.
- Scale horizontally by adding GPUs.
- Share weights and conversation storage between replicas.
- Do not increase Gunicorn threads to parallelize a single GPU.
- Compare serverless and dedicated GPU costs for steady traffic.

## Recommended observability

The project logs mode, reference count, and part of the prompt. For real
operations, add metrics for:

- Queue, loading, and inference latency.
- Cold starts.
- Failures and OOM events.
- GPU and VRAM usage.
- Images by model, resolution, and step count.
- Conversation-storage growth.

## Troubleshooting

### Model does not download

- Accept the Hugging Face license.
- Verify `HF_TOKEN`.
- Verify network or R2 access.

### Out of VRAM

- Verify `FLUX2_DEV_4BIT=true`.
- Lower resolution.
- Enable `MODEL_CPU_OFFLOAD=true` and accept higher latency.
- Use FLUX.2 Klein.

### Conversation disappears

`CONVERSATION_DIR` is not on persistent/shared storage.

### ControlNet returns 409

The active model is FLUX.2. Switch to FLUX.1 or use `/generate/edit`.

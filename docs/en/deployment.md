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

Typical variables:

```env
RUNPOD_ENABLED=true
MODEL_ID=black-forest-labs/FLUX.2-dev
HF_TOKEN=hf_...
FLUX2_DEV_4BIT=true
FLUX2_TEXT_ENCODER_MODE=remote
MODEL_CPU_OFFLOAD=false
PRELOAD_MODELS=false
HF_HOME=/cache/huggingface
CONVERSATION_DIR=/data/conversations
```

Mount persistent storage at:

- `/cache/huggingface` for weights.
- `/data/conversations` for memory.

For serverless, `PRELOAD_MODELS=false` starts the handler quickly, but the first
request pays model-loading time. Persistent pods usually benefit from
`PRELOAD_MODELS=true`.

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


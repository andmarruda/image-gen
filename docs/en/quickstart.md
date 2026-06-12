# Quickstart

## Requirements

- Docker with Docker Compose.
- Compatible NVIDIA GPU.
- NVIDIA Container Toolkit.
- Approximately 24 GB VRAM for the default FLUX.2 Dev 4-bit profile.
- Hugging Face token with accepted FLUX.2 Dev access.

## Configuration

```bash
cp .env.example .env
```

Set at least:

```env
HF_TOKEN=hf_your_token
MODEL_ID=black-forest-labs/FLUX.2-dev
FLUX2_DEV_4BIT=true
FLUX2_TEXT_ENCODER_MODE=remote
MODEL_CPU_OFFLOAD=false
```

Protect generation routes:

```env
API_KEY=a-long-random-key
```

## Start with Docker Compose

```bash
docker compose up --build
```

Compose creates persistent volumes for:

- Model weights and cache: `/cache/huggingface`.
- Conversations and revisions: `/data/conversations`.

## Check the service

```bash
curl http://localhost:5000/health
```

Example:

```json
{
  "status": "ok",
  "model": "black-forest-labs/FLUX.2-dev",
  "family": "flux2-dev",
  "loaded": false,
  "cuda": true,
  "defaults": {
    "num_inference_steps": 28,
    "guidance_scale": 4.0
  }
}
```

`loaded=false` only means lazy model loading has not happened yet.

## Generate the first image

Base64 JSON response:

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "prompt": "a futuristic observatory on a mountain, starry night",
    "seed": 42
  }'
```

Raw PNG response:

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: image/png" \
  -d '{"prompt":"a futuristic observatory on a mountain"}' \
  --output observatory.png
```

For PNG responses, memory IDs are returned in the `X-Conversation-Id` and
`X-Revision-Id` headers.

## Next steps

- [Edit images and use every route](api.md)
- [Continue an image with memory](conversation-memory.md)
- [Choose another model](models.md)


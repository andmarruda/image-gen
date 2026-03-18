# FLUX Image Generation API

> A production-ready, serverless-first inference server for the **FLUX.1 model family** — supporting both schnell and dev variants, with three generation modes: text-to-image, image-to-image, and ControlNet.

---

- [English Documentation](#english-documentation)
- [Documentação em Português](#documentação-em-português)

---

# English Documentation

## The Story

Every great product deserves great visuals. But integrating image generation into a real application has historically meant one of two things: paying per-request to a closed API you don't control, or wrestling with a Python notebook that was never meant to serve production traffic.

This project exists to close that gap.

**FLUX.1**, released by Black Forest Labs, represents a leap forward in open image generation. It comes in two flavors — **schnell** for raw speed, and **dev** for maximum quality — and this server supports both through a single environment variable.

What was missing was a clean, battle-tested API layer around them.

This is that layer.

A single Docker image. A configurable port. Three generation modes. No cloud vendor lock-in, no per-image fees, no terms of service surprises. Deploy it on any GPU-enabled machine or serverless platform — RunPod, Lambda Labs, Vast.ai, your own bare metal — and your application gets a dead-simple HTTP interface to one of the most capable open image models ever released.

---

## Understanding the Models: schnell vs dev

Before deploying, it's worth understanding what you're choosing between. Both models share the same transformer architecture, but they were trained differently for different purposes.

### FLUX.1-schnell

Schnell (German for "fast") is a **distilled** version of FLUX. Distillation is a training technique where a smaller or faster model is trained to mimic the output of a larger, slower "teacher" model — compressing knowledge from many inference steps into fewer. The result is a model that produces excellent images in as few as **4 denoising steps**.

This has a direct consequence for how you prompt it: because the model is distilled, it doesn't use Classifier-Free Guidance (CFG) — the mechanism that normally lets you dial up how closely the output follows the prompt. **`guidance_scale` must be set to `0.0` for schnell.** Setting it higher produces degraded results.

**Key characteristics:**
- 4 steps is the sweet spot (1–8 is the usable range)
- `guidance_scale` must be `0.0` (distilled model — CFG is disabled)
- ~23 GB on disk (weights)
- **Apache 2.0 licensed** — fully open for commercial use
- Best for: high-throughput production, cost-sensitive inference, prototyping

### FLUX.1-dev

Dev is the **non-distilled** guidance-distilled version — trained with full guidance, giving it significantly stronger prompt adherence and higher quality outputs. It requires more inference steps and can leverage CFG, which means you have a real dial to control how literally the model interprets your prompt.

This model is also **gated** on HuggingFace: you must accept Black Forest Labs' license terms on the model page and provide your HuggingFace token to download it.

**Key characteristics:**
- 20–50 steps is the recommended range (28 is a solid default)
- `guidance_scale` between `3.5` and `7.0` (higher = more literal prompt following)
- Same ~23 GB on disk (identical architecture to schnell)
- **Non-commercial license** — check [the model page](https://huggingface.co/black-forest-labs/FLUX.1-dev) before using in production
- Best for: maximum quality, artistic projects, research, fine-tuned workflows

### Side-by-side comparison

| | FLUX.1-schnell | FLUX.1-dev |
|---|---|---|
| Architecture | Distilled | Non-distilled |
| Recommended steps | 4 | 20–50 |
| `guidance_scale` | `0.0` (must be) | `3.5`–`7.0` |
| Quality ceiling | High | Higher |
| Speed | Fastest open model | Slower |
| License | Apache 2.0 | Non-commercial |
| HuggingFace gated | No | **Yes** (token required) |
| Size on disk | ~23 GB | ~23 GB |

### Switching between models

Switching is a single `.env` change — the server code reads `MODEL_ID` at startup and loads whichever model you point it at. No code changes required.

**To use schnell (default):**
```env
MODEL_ID=black-forest-labs/FLUX.1-schnell
# HF_TOKEN not required
```

**To use dev:**
```env
MODEL_ID=black-forest-labs/FLUX.1-dev
HF_TOKEN=hf_your_token_here
```

> **Important:** After switching to dev, update your API requests too.
> - Change `num_inference_steps` from `4` → `28` (or higher)
> - Change `guidance_scale` from `0.0` → `3.5`–`7.0`
>
> Sending schnell-optimized parameters to dev produces noticeably weaker results. The model is capable of much more — you just need to give it the steps and guidance to express that.

---

## Features

- **Text-to-image** — generate from a prompt with full parameter control
- **Image-to-image** — use a reference image as a starting point, guided by your prompt
- **ControlNet (Canny)** — preserve the exact structure and composition of a reference image while repainting it with a new style, content, or concept
- **Model-agnostic** — swap between FLUX.1-schnell and FLUX.1-dev via environment variable
- **Flexible response format** — receive the result as raw PNG bytes or a base64-encoded JSON payload, controlled by a single request header
- **Lazy model loading** — models are downloaded and loaded on first request; the container starts instantly
- **Persistent model cache** — mount a volume at `/cache/huggingface` and weights are downloaded exactly once, forever
- **`PRELOAD_MODELS=true`** — warm up all pipelines at startup so the first request is as fast as every subsequent one
- **Shared weights** — all three pipelines share the same transformer, VAE, and text encoders; switching between modes costs zero extra VRAM
- **Production server** — Gunicorn with configurable port via `.env`; no dev server in the hot path
- **Serverless-ready Dockerfile** — single-stage CUDA 12.1 image, no docker-compose required

---

## Quickstart

### 1. Clone and configure

```bash
git clone <your-repo>
cd image-generation
cp .env.example .env
```

Edit `.env` as needed. At minimum, review `MODEL_ID` and set `HF_TOKEN` if using FLUX.1-dev.

### 2. Build the image

```bash
docker build -t flux-api .
```

### 3. Run

```bash
# With GPU (recommended)
docker run --gpus all -p 5000:5000 --env-file .env flux-api

# With a persistent model cache (avoids re-downloading on every container restart)
docker run --gpus all -p 5000:5000 \
  -v /data/hf-cache:/cache/huggingface \
  --env-file .env \
  flux-api
```

> **First run:** FLUX weights (~23 GB) are downloaded on the first request. Subsequent requests use the local cache. Mount a volume to persist the cache across container restarts.

---

## API Reference

### `GET /health`

Liveness check. Returns `200 OK` when the server is running.

```json
{ "status": "ok" }
```

---

### `POST /generate` — Text to Image

Generate an image purely from a text prompt.

**Request body (JSON):**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | The image description |
| `num_inference_steps` | int | `4` | Denoising steps. Use `4` for schnell, `28`+ for dev |
| `guidance_scale` | float | `0.0` | CFG scale. Must be `0.0` for schnell; use `3.5`–`7.0` for dev |
| `width` | int | `1024` | Output width in pixels |
| `height` | int | `1024` | Output height in pixels |
| `seed` | int | — | Fix the random seed for reproducibility |

**Response format:**

| `Accept` header | Response |
|---|---|
| `application/json` (default) | JSON with `image` as base64 string |
| `image/png` | Raw PNG bytes |

You can also use the `X-Response-Format: bytes` header instead of `Accept`.

**Example — schnell (fast, 4 steps):**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a lone astronaut standing on a red sand dune, golden hour, cinematic",
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "seed": 42
  }'
```

**Example — dev (higher quality, 28 steps):**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a lone astronaut standing on a red sand dune, golden hour, cinematic, hyperrealistic",
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "seed": 42
  }'
```

**Example — raw PNG output:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -H "Accept: image/png" \
  -d '{"prompt": "a lone astronaut standing on a red sand dune, golden hour, cinematic"}' \
  --output result.png
```

---

### `POST /generate/img2img` — Image to Image

Provide a reference image and a prompt. The model starts from the reference and evolves it toward your description. Best for loose variations where you want to preserve the general mood or color palette, but not the exact structure.

The `strength` parameter controls how much the model deviates from the input image: `0.0` returns the original unchanged; `1.0` gives the model complete freedom to ignore the reference. Values between `0.5` and `0.8` are the useful range for creative variation.

**Request body (JSON or multipart/form-data):**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | The target description |
| `image` | string (base64) / file | **required** | Reference image |
| `strength` | float | `0.75` | How far to drift from the original. `0.0` = no change, `1.0` = ignore reference |
| `num_inference_steps` | int | `4` | Denoising steps (use `28`+ for dev) |
| `guidance_scale` | float | `0.0` | CFG scale (use `3.5`–`7.0` for dev) |
| `seed` | int | — | Random seed |

**Example — JSON:**
```bash
curl -X POST http://localhost:5000/generate/img2img \
  -H "Content-Type: application/json" \
  -H "Accept: image/png" \
  -d '{
    "prompt": "same composition, cyberpunk neon aesthetic, rain-soaked streets",
    "image": "<base64-encoded-image>",
    "strength": 0.7
  }' \
  --output result.png
```

**Example — multipart upload:**
```bash
curl -X POST http://localhost:5000/generate/img2img \
  -H "Accept: image/png" \
  -F "prompt=same composition, cyberpunk neon aesthetic" \
  -F "image=@./reference.png" \
  -F "strength=0.7" \
  --output result.png
```

---

### `POST /generate/controlnet` — Structure-Preserving Generation

The most powerful mode. A Canny edge detector extracts the structural skeleton of your reference image — every silhouette, every architectural line, every contour — and uses it as a hard structural constraint during generation. The model then fills that skeleton with exactly what your prompt describes.

The result is consistent composition with complete creative freedom over style, lighting, material, and content.

**How Canny edge detection works:** The algorithm finds regions of strong intensity gradient in the image (i.e., edges), applies two thresholds, and traces the resulting contours. `canny_low_threshold` controls how sensitive it is to faint edges; `canny_high_threshold` controls what qualifies as a strong edge. The output is a binary edge map — white lines on black — that the ControlNet uses to constrain generation.

> **Note on model compatibility:** The default ControlNet adapter (`InstantX/FLUX.1-dev-Controlnet-Canny`) was trained on FLUX.1-dev weights. It works with both schnell and dev as the base model, but produces the most consistent results when paired with dev. If you're running schnell and notice structural drift, switching to dev will improve ControlNet accuracy.

**Request body (JSON or multipart/form-data):**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | What to generate inside the reference structure |
| `image` | string (base64) / file | **required** | Reference image for edge extraction |
| `controlnet_conditioning_scale` | float | `0.7` | How strictly to follow the edges. `1.0` = rigid, `0.3` = loose |
| `canny_low_threshold` | int | `100` | Lower bound for edge detection sensitivity |
| `canny_high_threshold` | int | `200` | Upper bound for edge detection sensitivity |
| `num_inference_steps` | int | `28` | Denoising steps. ControlNet benefits from more steps |
| `guidance_scale` | float | `3.5` | CFG scale |
| `width` | int | `1024` | Output width |
| `height` | int | `1024` | Output height |
| `seed` | int | — | Random seed |

**Example — turn a photo of a building into a watercolor painting:**
```bash
curl -X POST http://localhost:5000/generate/controlnet \
  -H "Accept: image/png" \
  -F "prompt=watercolor painting of a gothic cathedral, soft brushstrokes, pastel sky" \
  -F "image=@./building.jpg" \
  -F "controlnet_conditioning_scale=0.8" \
  --output result.png
```

**Example — restyle a portrait while keeping the pose:**
```bash
curl -X POST http://localhost:5000/generate/controlnet \
  -H "Content-Type: application/json" \
  -H "Accept: image/png" \
  -d '{
    "prompt": "oil painting portrait, Renaissance style, warm candlelight",
    "image": "<base64-encoded-portrait>",
    "controlnet_conditioning_scale": 0.75,
    "num_inference_steps": 28
  }' \
  --output result.png
```

---

## When to Use Which Mode

| Goal | Recommended route |
|---|---|
| Generate from scratch | `/generate` |
| Loose variation of an existing image | `/generate/img2img` |
| Restyle while keeping composition/structure | `/generate/controlnet` |
| Portrait restyle, keeping pose | `/generate/controlnet` |
| Architecture restyle | `/generate/controlnet` |
| Product mockup with new background | `/generate/img2img` |

---

## Configuration

All configuration is done through environment variables. Copy `.env.example` to `.env` and edit as needed.

| Variable | Default | Description |
|---|---|---|
| `RUNPOD_ENABLED` | `false` | Set to `true` to run as a RunPod serverless handler |
| `PRELOAD_MODELS` | `false` | Set to `true` to download and load all models at startup |
| `PORT` | `5000` | Port the server listens on (Flask/Gunicorn mode only) |
| `MODEL_ID` | `black-forest-labs/FLUX.1-schnell` | HuggingFace model ID. Change to `black-forest-labs/FLUX.1-dev` to use the dev model |
| `CONTROLNET_MODEL_ID` | `InstantX/FLUX.1-dev-Controlnet-Canny` | ControlNet adapter model ID |
| `HF_TOKEN` | — | HuggingFace token. **Required** when using any gated model (FLUX.1-dev) |
| `HF_HOME` | `/cache/huggingface` | Model cache directory inside the container |

---

## Project Structure

```
image-generation/
├── .github/
│   └── workflows/
│       └── docker-build.yml   # CI: build & push to GHCR on every push to main
├── config/
│   └── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # Model loading and pipeline management
│   ├── routes.py              # Flask Blueprint — all HTTP endpoints
│   ├── runpod_handler.py      # RunPod serverless handler
│   └── utils.py               # Image encoding, Canny edge detection
├── app.py                     # Entrypoint — branches to Flask or RunPod
├── Dockerfile                 # CUDA 12.1 single-stage image
├── .env.example               # Environment variable template
└── .dockerignore
```

---

## Architecture Notes

**Dual runtime** — a single `app.py` entrypoint supports both deployment models. When `RUNPOD_ENABLED=false` (default), Gunicorn serves HTTP traffic. When `RUNPOD_ENABLED=true`, the RunPod serverless SDK takes over the process loop. The same generation logic powers both paths.

**Lazy loading** — no model is loaded at container startup. The first request to any endpoint triggers the download and load of the required pipeline. This keeps cold start time near zero and avoids downloading models that won't be used.

**Shared weights** — `FluxImg2ImgPipeline` and `FluxControlNetPipeline` are both constructed via `from_pipe()` from the base `FluxPipeline`. The transformer (~12 GB), VAE, and both text encoders are shared across all three pipelines. Only the tiny ControlNet adapter (~300 MB) is an additional allocation.

**Single worker** — Gunicorn is configured with `--workers 1 --threads 1`. The FLUX model is not thread-safe and holds most of the GPU memory. Multiple workers would OOM on any reasonable GPU. Concurrency is handled at the infrastructure layer (multiple replicas, request queuing).

---

## Model Caching

FLUX weights weigh ~23 GB. Without a persistent cache, every new container downloads those weights from scratch — which takes several minutes and costs egress bandwidth.

The solution is a single mounted volume.

### How it works

The container stores all downloaded weights in `HF_HOME` (`/cache/huggingface` by default). Mounting a volume at that path makes the cache survive container restarts, redeployments, and image updates. The HuggingFace library checks the cache before downloading — if the files are there, it skips the download entirely.

### First-time setup

Run the preload script once against your persistent volume to populate it:

```bash
docker run --rm --gpus all \
  -v /your/persistent/path:/cache/huggingface \
  -e HF_HOME=/cache/huggingface \
  ghcr.io/<you>/image-generation:latest \
  python scripts/preload_models.py
```

This downloads all three pipelines (base + img2img + ControlNet) and exits. Every subsequent container launch using that volume skips the download and goes straight to loading weights from disk.

### Every launch after that

```bash
docker run --gpus all \
  -v /your/persistent/path:/cache/huggingface \
  -e HF_HOME=/cache/huggingface \
  -e PRELOAD_MODELS=true \
  -p 5000:5000 \
  ghcr.io/<you>/image-generation:latest
```

### RunPod Network Volume

RunPod's Network Volumes are persistent SSDs shared across worker restarts — the right tool for this.

1. Go to **Storage → New Network Volume**, create a 60 GB volume in the same region as your endpoint
2. In your Serverless Endpoint settings, attach the volume and set its **Mount Path** to `/cache/huggingface`
3. Add env vars: `HF_HOME=/cache/huggingface` and `PRELOAD_MODELS=true`
4. Run the preload script once manually from a one-off pod to populate the volume
5. From that point on, every worker that spins up loads from the volume — no downloads, minimal cold start

---

## Deployment

### GitHub Actions — Automated Builds

Every push to `main` automatically builds the Docker image and pushes it to the **GitHub Container Registry (GHCR)**. No external registry account required — authentication uses the built-in `GITHUB_TOKEN`.

The workflow generates three tag variants:

| Tag | Example | When |
|---|---|---|
| `latest` | `ghcr.io/you/image-generation:latest` | Every push to `main` |
| Semver | `ghcr.io/you/image-generation:1.2.3` | On `v*.*.*` git tags |
| SHA | `ghcr.io/you/image-generation:sha-a1b2c3d` | Every push, for traceability |

Pull requests trigger a build but no push, so every PR is validated before it lands.

---

### RunPod Serverless

RunPod Serverless scales your worker to zero when idle and bills only for active compute time — the most cost-effective way to run GPU inference.

**1. Push your image** (or let GitHub Actions do it automatically):
```bash
docker build -t ghcr.io/<your-username>/image-generation:latest .
docker push ghcr.io/<your-username>/image-generation:latest
```

**2. Create a Serverless endpoint on RunPod:**
- Go to **Serverless → New Endpoint**
- Set **Container Image** to `ghcr.io/<your-username>/image-generation:latest`
- Add environment variable: `RUNPOD_ENABLED=true`
- Set GPU: RTX 4090 or A100 recommended for FLUX
- Set **Container Disk**: 50 GB minimum (model cache)

**3. Submit a job:**
```bash
curl -X POST https://api.runpod.io/v2/<endpoint-id>/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "txt2img",
      "prompt": "a lone astronaut on a red sand dune, golden hour, cinematic",
      "seed": 42
    }
  }'
```

**4. Retrieve the result:**
```bash
curl https://api.runpod.io/v2/<endpoint-id>/status/<job-id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

The response `output.image` contains the base64-encoded PNG.

**RunPod job input schema:**

All three generation modes are available. Pass `mode` to select:

```json
{
  "input": {
    "mode": "txt2img | img2img | controlnet",
    "prompt": "your prompt here",
    "image": "<base64>",
    "...same optional params as the HTTP API..."
  }
}
```

---

### RunPod Pods (HTTP mode)

If you prefer a persistent pod with the standard HTTP interface, deploy without setting `RUNPOD_ENABLED`:

- Set **Container Image** to your GHCR image
- Expose **port 5000**
- Leave `RUNPOD_ENABLED` unset (defaults to `false`)
- Your pod serves the full HTTP API described above

---

### Any GPU Server

```bash
docker run --gpus all -d \
  -p 5000:5000 \
  -v /data/hf-cache:/cache/huggingface \
  --env-file .env \
  --restart unless-stopped \
  ghcr.io/<your-username>/image-generation:latest
```

---

## Acknowledgments

- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) by Black Forest Labs — Apache 2.0
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) by Black Forest Labs — Non-commercial
- [InstantX ControlNet for FLUX](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny) — Apache 2.0
- [Diffusers](https://github.com/huggingface/diffusers) by HuggingFace
- [RunPod](https://runpod.io) — serverless GPU infrastructure

---

---

# Documentação em Português

## A Proposta

Todo produto merece visuais à altura. Mas integrar geração de imagens a uma aplicação real sempre significou uma de duas coisas: pagar por requisição a uma API fechada que você não controla, ou tentar colocar em produção um notebook Python que nunca foi pensado para servir tráfego real.

Este projeto existe para fechar essa lacuna.

**FLUX.1**, lançado pelo Black Forest Labs, representa um salto na geração de imagens open-source. Ele vem em dois sabores — **schnell** para velocidade máxima e **dev** para qualidade máxima — e este servidor suporta ambos através de uma única variável de ambiente.

O que faltava era uma camada de API limpa e testada em produção em torno deles.

Esta é essa camada.

---

## Entendendo os Modelos: schnell vs dev

Antes de fazer o deploy, vale entender o que você está escolhendo. Ambos os modelos compartilham a mesma arquitetura transformer, mas foram treinados de formas diferentes para propósitos diferentes.

### FLUX.1-schnell

Schnell (em alemão, "rápido") é uma versão **destilada** do FLUX. Destilação é uma técnica de treinamento onde um modelo menor ou mais rápido aprende a imitar a saída de um modelo "professor" maior e mais lento — comprimindo o conhecimento de muitos passos de inferência em menos. O resultado é um modelo que produz imagens excelentes em apenas **4 passos de denoising**.

Isso tem uma consequência direta na forma como você o usa: como o modelo é destilado, ele **não utiliza Classifier-Free Guidance (CFG)** — o mecanismo que normalmente permite controlar o quanto a saída segue o prompt. **`guidance_scale` deve ser `0.0` para o schnell.** Valores maiores degradam o resultado.

**Características principais:**
- 4 passos é o ponto ideal (1–8 é o intervalo utilizável)
- `guidance_scale` deve ser `0.0` (modelo destilado — CFG desativado)
- ~23 GB em disco (pesos)
- **Licença Apache 2.0** — totalmente aberto para uso comercial
- Ideal para: produção de alta taxa, inferência com custo controlado, prototipagem

### FLUX.1-dev

Dev é a versão **não-destilada** com guidance — treinada com guidance completo, o que resulta em aderência significativamente maior ao prompt e saídas de qualidade superior. Ele requer mais passos de inferência e pode usar CFG, o que significa que você tem controle real sobre o quanto o modelo interpreta o prompt literalmente.

Este modelo também é **gated** no HuggingFace: você precisa aceitar os termos de licença do Black Forest Labs na página do modelo e fornecer seu token do HuggingFace para baixá-lo.

**Características principais:**
- 20–50 passos é o intervalo recomendado (28 é um bom padrão)
- `guidance_scale` entre `3.5` e `7.0` (maior = mais literal na interpretação do prompt)
- Mesmo ~23 GB em disco (arquitetura idêntica ao schnell)
- **Licença não-comercial** — verifique [a página do modelo](https://huggingface.co/black-forest-labs/FLUX.1-dev) antes de usar em produção
- Ideal para: qualidade máxima, projetos artísticos, pesquisa, workflows com fine-tuning

### Comparação lado a lado

| | FLUX.1-schnell | FLUX.1-dev |
|---|---|---|
| Arquitetura | Destilado | Não-destilado |
| Passos recomendados | 4 | 20–50 |
| `guidance_scale` | `0.0` (obrigatório) | `3.5`–`7.0` |
| Teto de qualidade | Alto | Mais alto |
| Velocidade | Mais rápido modelo open | Mais lento |
| Licença | Apache 2.0 | Não-comercial |
| Gated no HuggingFace | Não | **Sim** (token obrigatório) |
| Tamanho em disco | ~23 GB | ~23 GB |

### Trocando entre os modelos

Trocar é uma mudança de uma linha no `.env` — o código do servidor lê `MODEL_ID` na inicialização e carrega o modelo que você apontar. Nenhuma mudança de código é necessária.

**Para usar o schnell (padrão):**
```env
MODEL_ID=black-forest-labs/FLUX.1-schnell
# HF_TOKEN não é necessário
```

**Para usar o dev:**
```env
MODEL_ID=black-forest-labs/FLUX.1-dev
HF_TOKEN=hf_seu_token_aqui
```

> **Importante:** Ao trocar para o dev, atualize também os parâmetros nas suas requisições à API.
> - Mude `num_inference_steps` de `4` → `28` (ou mais)
> - Mude `guidance_scale` de `0.0` → `3.5`–`7.0`
>
> Enviar parâmetros otimizados para o schnell ao dev produz resultados visivelmente piores. O modelo é capaz de muito mais — mas precisa dos passos e do guidance para expressar isso.

---

## Funcionalidades

- **Text-to-image** — gere a partir de um prompt com controle total dos parâmetros
- **Image-to-image** — use uma imagem de referência como ponto de partida, guiada pelo seu prompt
- **ControlNet (Canny)** — preserve a estrutura e composição exata de uma imagem de referência enquanto a repinta com novo estilo, conteúdo ou conceito
- **Agnóstico ao modelo** — troque entre FLUX.1-schnell e FLUX.1-dev via variável de ambiente
- **Formato de resposta flexível** — receba o resultado como bytes PNG brutos ou payload JSON com base64, controlado por um único header da requisição
- **Carregamento lazy de modelos** — os modelos são baixados e carregados na primeira requisição; o container inicia instantaneamente
- **Cache persistente de modelos** — monte um volume em `/cache/huggingface` e os pesos são baixados uma única vez, para sempre
- **`PRELOAD_MODELS=true`** — aquece todos os pipelines na inicialização para que a primeira requisição seja tão rápida quanto as seguintes
- **Pesos compartilhados** — todos os três pipelines compartilham o mesmo transformer, VAE e text encoders; trocar entre modos não custa VRAM extra
- **Servidor de produção** — Gunicorn com porta configurável via `.env`; sem servidor de dev no caminho crítico
- **Dockerfile serverless-ready** — imagem single-stage CUDA 12.1, sem docker-compose necessário

---

## Quickstart

### 1. Clone e configure

```bash
git clone <seu-repo>
cd image-generation
cp .env.example .env
```

Edite o `.env` conforme necessário. No mínimo, revise `MODEL_ID` e defina `HF_TOKEN` se for usar o FLUX.1-dev.

### 2. Construa a imagem

```bash
docker build -t flux-api .
```

### 3. Execute

```bash
# Com GPU (recomendado)
docker run --gpus all -p 5000:5000 --env-file .env flux-api

# Com cache persistente de modelos (evita re-download a cada reinício do container)
docker run --gpus all -p 5000:5000 \
  -v /data/hf-cache:/cache/huggingface \
  --env-file .env \
  flux-api
```

> **Primeira execução:** Os pesos do FLUX (~23 GB) são baixados na primeira requisição. Requisições subsequentes usam o cache local. Monte um volume para persistir o cache entre reinícios do container.

---

## Referência da API

### `GET /health`

Verificação de liveness. Retorna `200 OK` quando o servidor está em execução.

```json
{ "status": "ok" }
```

---

### `POST /generate` — Text to Image

Gera uma imagem puramente a partir de um prompt de texto.

**Corpo da requisição (JSON):**

| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `prompt` | string | **obrigatório** | A descrição da imagem |
| `num_inference_steps` | int | `4` | Passos de denoising. Use `4` para schnell, `28`+ para dev |
| `guidance_scale` | float | `0.0` | Escala CFG. Deve ser `0.0` para schnell; use `3.5`–`7.0` para dev |
| `width` | int | `1024` | Largura da saída em pixels |
| `height` | int | `1024` | Altura da saída em pixels |
| `seed` | int | — | Semente aleatória para reprodutibilidade |

**Formato de resposta:**

| Header `Accept` | Resposta |
|---|---|
| `application/json` (padrão) | JSON com `image` como string base64 |
| `image/png` | Bytes PNG brutos |

Você também pode usar o header `X-Response-Format: bytes` ao invés de `Accept`.

**Exemplo — schnell (rápido, 4 passos):**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "um astronauta solitário em uma duna de areia vermelha, hora dourada, cinematográfico",
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "seed": 42
  }'
```

**Exemplo — dev (maior qualidade, 28 passos):**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "um astronauta solitário em uma duna de areia vermelha, hora dourada, cinematográfico, hiperrealista",
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "seed": 42
  }'
```

**Exemplo — saída PNG bruta:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -H "Accept: image/png" \
  -d '{"prompt": "um astronauta solitário em uma duna de areia vermelha, hora dourada, cinematográfico"}' \
  --output resultado.png
```

---

### `POST /generate/img2img` — Image to Image

Forneça uma imagem de referência e um prompt. O modelo parte da referência e a evolui em direção à sua descrição. Ideal para variações livres onde você quer preservar o humor geral ou a paleta de cores, mas não a estrutura exata.

O parâmetro `strength` controla o quanto o modelo se desvia da imagem de entrada: `0.0` retorna o original sem alteração; `1.0` dá ao modelo liberdade total para ignorar a referência. Valores entre `0.5` e `0.8` são o intervalo útil para variação criativa.

**Corpo da requisição (JSON ou multipart/form-data):**

| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `prompt` | string | **obrigatório** | A descrição do alvo |
| `image` | string (base64) / arquivo | **obrigatório** | Imagem de referência |
| `strength` | float | `0.75` | O quanto se afastar do original. `0.0` = sem mudança, `1.0` = ignora a referência |
| `num_inference_steps` | int | `4` | Passos de denoising (use `28`+ para dev) |
| `guidance_scale` | float | `0.0` | Escala CFG (use `3.5`–`7.0` para dev) |
| `seed` | int | — | Semente aleatória |

**Exemplo — JSON:**
```bash
curl -X POST http://localhost:5000/generate/img2img \
  -H "Content-Type: application/json" \
  -H "Accept: image/png" \
  -d '{
    "prompt": "mesma composição, estética cyberpunk neon, ruas molhadas de chuva",
    "image": "<imagem-em-base64>",
    "strength": 0.7
  }' \
  --output resultado.png
```

**Exemplo — upload multipart:**
```bash
curl -X POST http://localhost:5000/generate/img2img \
  -H "Accept: image/png" \
  -F "prompt=mesma composição, estética cyberpunk neon" \
  -F "image=@./referencia.png" \
  -F "strength=0.7" \
  --output resultado.png
```

---

### `POST /generate/controlnet` — Geração com Preservação de Estrutura

O modo mais poderoso. Um detector de bordas Canny extrai o esqueleto estrutural da sua imagem de referência — cada silhueta, cada linha arquitetônica, cada contorno — e o usa como uma restrição estrutural rígida durante a geração. O modelo então preenche esse esqueleto com exatamente o que o seu prompt descreve.

O resultado é composição consistente com liberdade criativa total sobre estilo, iluminação, material e conteúdo.

**Como funciona a detecção de bordas Canny:** O algoritmo encontra regiões de forte gradiente de intensidade na imagem (ou seja, bordas), aplica dois limiares e traça os contornos resultantes. `canny_low_threshold` controla a sensibilidade a bordas fracas; `canny_high_threshold` controla o que se qualifica como borda forte. A saída é um mapa de bordas binário — linhas brancas em preto — que o ControlNet usa para restringir a geração.

> **Nota sobre compatibilidade de modelos:** O adapter ControlNet padrão (`InstantX/FLUX.1-dev-Controlnet-Canny`) foi treinado com pesos do FLUX.1-dev. Ele funciona com schnell e dev como modelo base, mas produz resultados mais consistentes quando combinado com dev. Se você está rodando schnell e nota desvio estrutural, trocar para dev melhorará a precisão do ControlNet.

**Corpo da requisição (JSON ou multipart/form-data):**

| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `prompt` | string | **obrigatório** | O que gerar dentro da estrutura de referência |
| `image` | string (base64) / arquivo | **obrigatório** | Imagem de referência para extração de bordas |
| `controlnet_conditioning_scale` | float | `0.7` | O quanto seguir as bordas rigidamente. `1.0` = rígido, `0.3` = solto |
| `canny_low_threshold` | int | `100` | Limiar inferior da sensibilidade de detecção de bordas |
| `canny_high_threshold` | int | `200` | Limiar superior da sensibilidade de detecção de bordas |
| `num_inference_steps` | int | `28` | Passos de denoising. ControlNet se beneficia de mais passos |
| `guidance_scale` | float | `3.5` | Escala CFG |
| `width` | int | `1024` | Largura da saída |
| `height` | int | `1024` | Altura da saída |
| `seed` | int | — | Semente aleatória |

**Exemplo — transformar foto de um prédio em aquarela:**
```bash
curl -X POST http://localhost:5000/generate/controlnet \
  -H "Accept: image/png" \
  -F "prompt=pintura em aquarela de uma catedral gótica, pinceladas suaves, céu pastel" \
  -F "image=@./predio.jpg" \
  -F "controlnet_conditioning_scale=0.8" \
  --output resultado.png
```

**Exemplo — reestilizar um retrato mantendo a pose:**
```bash
curl -X POST http://localhost:5000/generate/controlnet \
  -H "Content-Type: application/json" \
  -H "Accept: image/png" \
  -d '{
    "prompt": "retrato em pintura a óleo, estilo Renascentista, luz de vela quente",
    "image": "<retrato-em-base64>",
    "controlnet_conditioning_scale": 0.75,
    "num_inference_steps": 28
  }' \
  --output resultado.png
```

---

## Quando Usar Cada Modo

| Objetivo | Rota recomendada |
|---|---|
| Gerar do zero | `/generate` |
| Variação livre de uma imagem existente | `/generate/img2img` |
| Reestilizar preservando composição/estrutura | `/generate/controlnet` |
| Reestilizar retrato mantendo a pose | `/generate/controlnet` |
| Reestilizar arquitetura | `/generate/controlnet` |
| Mockup de produto com novo fundo | `/generate/img2img` |

---

## Configuração

Toda a configuração é feita através de variáveis de ambiente. Copie `.env.example` para `.env` e edite conforme necessário.

| Variável | Padrão | Descrição |
|---|---|---|
| `RUNPOD_ENABLED` | `false` | Defina como `true` para rodar como handler serverless do RunPod |
| `PRELOAD_MODELS` | `false` | Defina como `true` para baixar e carregar todos os modelos na inicialização |
| `PORT` | `5000` | Porta que o servidor escuta (apenas no modo Flask/Gunicorn) |
| `MODEL_ID` | `black-forest-labs/FLUX.1-schnell` | ID do modelo no HuggingFace. Mude para `black-forest-labs/FLUX.1-dev` para usar o modelo dev |
| `CONTROLNET_MODEL_ID` | `InstantX/FLUX.1-dev-Controlnet-Canny` | ID do adapter ControlNet |
| `HF_TOKEN` | — | Token do HuggingFace. **Obrigatório** ao usar qualquer modelo gated (FLUX.1-dev) |
| `HF_HOME` | `/cache/huggingface` | Diretório de cache de modelos dentro do container |

---

## Estrutura do Projeto

```
image-generation/
├── .github/
│   └── workflows/
│       └── docker-build.yml   # CI: build & push para GHCR a cada push na main
├── config/
│   └── requirements.txt       # Dependências Python
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # Carregamento de modelos e gerenciamento de pipelines
│   ├── routes.py              # Flask Blueprint — todos os endpoints HTTP
│   ├── runpod_handler.py      # Handler serverless do RunPod
│   └── utils.py               # Encoding de imagens, detecção de bordas Canny
├── app.py                     # Entrypoint — ramifica para Flask ou RunPod
├── Dockerfile                 # Imagem single-stage CUDA 12.1
├── .env.example               # Template de variáveis de ambiente
└── .dockerignore
```

---

## Notas de Arquitetura

**Runtime duplo** — um único entrypoint `app.py` suporta ambos os modelos de deploy. Quando `RUNPOD_ENABLED=false` (padrão), o Gunicorn serve tráfego HTTP. Quando `RUNPOD_ENABLED=true`, o SDK serverless do RunPod assume o loop de processo. A mesma lógica de geração alimenta ambos os caminhos.

**Carregamento lazy** — nenhum modelo é carregado na inicialização do container. A primeira requisição a qualquer endpoint dispara o download e carregamento do pipeline necessário. Isso mantém o cold start perto de zero e evita baixar modelos que não serão usados.

**Pesos compartilhados** — `FluxImg2ImgPipeline` e `FluxControlNetPipeline` são construídos via `from_pipe()` a partir do `FluxPipeline` base. O transformer (~12 GB), VAE e ambos os text encoders são compartilhados entre os três pipelines. Apenas o pequeno adapter ControlNet (~300 MB) é uma alocação adicional.

**Worker único** — o Gunicorn é configurado com `--workers 1 --threads 1`. O modelo FLUX não é thread-safe e ocupa a maior parte da memória GPU. Múltiplos workers causariam OOM em qualquer GPU razoável. A concorrência é tratada na camada de infraestrutura (múltiplas réplicas, fila de requisições).

---

## Cache de Modelos

Os pesos do FLUX pesam ~23 GB. Sem cache persistente, cada novo container baixa esses pesos do zero — o que leva vários minutos e consome banda de saída.

A solução é um único volume montado.

### Como funciona

O container armazena todos os pesos baixados em `HF_HOME` (`/cache/huggingface` por padrão). Montar um volume nesse caminho faz o cache sobreviver a reinícios do container, redeploys e atualizações de imagem. A biblioteca HuggingFace verifica o cache antes de baixar — se os arquivos estiverem lá, pula o download.

### Configuração inicial

Execute o script de preload uma vez contra seu volume persistente para populá-lo:

```bash
docker run --rm --gpus all \
  -v /seu/caminho/persistente:/cache/huggingface \
  -e HF_HOME=/cache/huggingface \
  ghcr.io/<voce>/image-generation:latest \
  python scripts/preload_models.py
```

Isso baixa todos os três pipelines (base + img2img + ControlNet) e encerra. Cada lançamento subsequente do container usando esse volume pula o download e vai direto para carregar os pesos do disco.

### Cada lançamento após isso

```bash
docker run --gpus all \
  -v /seu/caminho/persistente:/cache/huggingface \
  -e HF_HOME=/cache/huggingface \
  -e PRELOAD_MODELS=true \
  -p 5000:5000 \
  ghcr.io/<voce>/image-generation:latest
```

### RunPod Network Volume

Os Network Volumes do RunPod são SSDs persistentes compartilhados entre reinícios de workers — a ferramenta certa para isso.

1. Vá em **Storage → New Network Volume**, crie um volume de 60 GB na mesma região do seu endpoint
2. Nas configurações do Serverless Endpoint, anexe o volume e defina seu **Mount Path** como `/cache/huggingface`
3. Adicione as env vars: `HF_HOME=/cache/huggingface` e `PRELOAD_MODELS=true`
4. Execute o script de preload uma vez manualmente a partir de um pod avulso para popular o volume
5. A partir daí, cada worker que iniciar carregará do volume — sem downloads, cold start mínimo

---

## Deploy

### GitHub Actions — Builds Automatizados

Cada push na `main` constrói automaticamente a imagem Docker e a envia para o **GitHub Container Registry (GHCR)**. Nenhuma conta de registry externo é necessária — a autenticação usa o `GITHUB_TOKEN` embutido.

O workflow gera três variantes de tag:

| Tag | Exemplo | Quando |
|---|---|---|
| `latest` | `ghcr.io/voce/image-generation:latest` | A cada push na `main` |
| Semver | `ghcr.io/voce/image-generation:1.2.3` | Em git tags `v*.*.*` |
| SHA | `ghcr.io/voce/image-generation:sha-a1b2c3d` | A cada push, para rastreabilidade |

Pull requests disparam um build mas sem push, então cada PR é validado antes de chegar à main.

---

### RunPod Serverless

O RunPod Serverless escala seu worker para zero quando ocioso e cobra apenas pelo tempo de computação ativo — a forma mais econômica de rodar inferência em GPU.

**1. Faça o push da sua imagem** (ou deixe o GitHub Actions fazer automaticamente):
```bash
docker build -t ghcr.io/<seu-usuario>/image-generation:latest .
docker push ghcr.io/<seu-usuario>/image-generation:latest
```

**2. Crie um endpoint Serverless no RunPod:**
- Vá em **Serverless → New Endpoint**
- Defina **Container Image** como `ghcr.io/<seu-usuario>/image-generation:latest`
- Adicione a variável de ambiente: `RUNPOD_ENABLED=true`
- GPU: RTX 4090 ou A100 recomendados para FLUX
- **Container Disk**: mínimo 50 GB (cache de modelos)

**3. Envie um job:**
```bash
curl -X POST https://api.runpod.io/v2/<endpoint-id>/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "txt2img",
      "prompt": "um astronauta solitário em uma duna de areia vermelha, hora dourada, cinematográfico",
      "seed": 42
    }
  }'
```

**4. Recupere o resultado:**
```bash
curl https://api.runpod.io/v2/<endpoint-id>/status/<job-id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

O `output.image` da resposta contém o PNG codificado em base64.

**Schema de input do job RunPod:**

Os três modos de geração estão disponíveis. Passe `mode` para selecionar:

```json
{
  "input": {
    "mode": "txt2img | img2img | controlnet",
    "prompt": "seu prompt aqui",
    "image": "<base64>",
    "...mesmos parâmetros opcionais da API HTTP..."
  }
}
```

---

### RunPod Pods (modo HTTP)

Se preferir um pod persistente com a interface HTTP padrão, faça o deploy sem definir `RUNPOD_ENABLED`:

- Defina **Container Image** para sua imagem no GHCR
- Exponha a **porta 5000**
- Deixe `RUNPOD_ENABLED` sem definição (padrão `false`)
- Seu pod serve a API HTTP completa descrita acima

---

### Qualquer Servidor GPU

```bash
docker run --gpus all -d \
  -p 5000:5000 \
  -v /data/hf-cache:/cache/huggingface \
  --env-file .env \
  --restart unless-stopped \
  ghcr.io/<seu-usuario>/image-generation:latest
```

---

## Créditos

- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) pelo Black Forest Labs — Apache 2.0
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) pelo Black Forest Labs — Não-comercial
- [InstantX ControlNet para FLUX](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny) — Apache 2.0
- [Diffusers](https://github.com/huggingface/diffusers) pelo HuggingFace
- [RunPod](https://runpod.io) — infraestrutura GPU serverless

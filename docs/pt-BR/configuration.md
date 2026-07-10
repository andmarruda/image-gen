# Referência de Configuração

As variáveis podem ser definidas em `.env`, no Docker, no RunPod ou diretamente
no ambiente do processo.

## Runtime e API

| Variável | Padrão | Descrição |
|---|---|---|
| `RUNPOD_ENABLED` | `false` | Usa handler RunPod em vez do servidor HTTP |
| `PRELOAD_MODELS` | `false` | Carrega pesos e pipelines no startup |
| `PORT` | `5000` | Porta HTTP |
| `API_KEY` | não definida | Protege rotas, exceto `/health` |
| `MAX_REQUEST_MB` | `32` | Tamanho máximo do request HTTP |

## Modelo

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_ID` | `black-forest-labs/FLUX.2-dev` | Modelo configurado |
| `HF_TOKEN` | não definido | Token para modelos gated e encoder remoto |
| `HF_HOME` | `/cache/huggingface` no Docker; use `/runpod-volume/huggingface` no RunPod Serverless | Cache Hugging Face |
| `MODEL_CPU_OFFLOAD` | `false` | Move componentes entre CPU/GPU para reduzir VRAM |
| `FLUX2_DEV_4BIT` | `true` | Usa checkpoint Dev quantizado |
| `FLUX2_DEV_QUANTIZED_MODEL_ID` | `diffusers/FLUX.2-dev-bnb-4bit` | Checkpoint runtime 4-bit |
| `FLUX2_TEXT_ENCODER_MODE` | `remote` no `.env.example` | `remote` remove encoder local |
| `FLUX2_REMOTE_TEXT_ENCODER_URL` | endpoint Hugging Face configurado no código | Serviço remoto de embeddings |
| `FLUX2_TEXT_ENCODER_TIMEOUT` | `120` | Timeout do encoder remoto em segundos |

Observação: quando `FLUX2_TEXT_ENCODER_MODE=remote`, o prompt é enviado ao
serviço remoto.

## Memória de conversa

| Variável | Padrão | Descrição |
|---|---|---|
| `CONVERSATION_MEMORY_ENABLED` | `true` | Ativa memória visual |
| `CONVERSATION_DIR` | `/data/conversations`; use `/runpod-volume/conversations` no RunPod Serverless | Diretório de imagens e manifestos |
| `CONVERSATION_MAX_REVISIONS` | `10` | Revisões mantidas por conversa |

## FLUX.1 e ControlNet

| Variável | Padrão | Descrição |
|---|---|---|
| `CONTROLNET_MODEL_ID` | `InstantX/FLUX.1-dev-Controlnet-Canny` | Adapter ControlNet |
| `DOWNLOAD_CONTROLNET` | `true` | Carrega/baixa ControlNet no preload FLUX.1 |
| `UPLOAD_CONTROLNET` | `false` | Envia ControlNet ao R2 pelo script |

Essas opções são ignoradas para FLUX.2.

## Cloudflare R2

| Variável | Padrão | Descrição |
|---|---|---|
| `R2_ENABLED` | `false` | Ativa download dos pesos via R2 |
| `R2_ACCOUNT_ID` | obrigatório com R2 | Account ID Cloudflare |
| `R2_ACCESS_KEY_ID` | obrigatório com R2 | Access key |
| `R2_SECRET_ACCESS_KEY` | obrigatório com R2 | Secret key |
| `R2_BUCKET_NAME` | obrigatório com R2 | Bucket dos pesos |

## Perfis prontos

FLUX.2 Dev para estudo:

```env
MODEL_ID=black-forest-labs/FLUX.2-dev
HF_TOKEN=hf_...
FLUX2_DEV_4BIT=true
FLUX2_TEXT_ENCODER_MODE=remote
MODEL_CPU_OFFLOAD=false
```

FLUX.2 Klein econômico:

```env
MODEL_ID=black-forest-labs/FLUX.2-klein-4B
MODEL_CPU_OFFLOAD=false
```

FLUX.1 Dev com ControlNet:

```env
MODEL_ID=black-forest-labs/FLUX.1-dev
HF_TOKEN=hf_...
DOWNLOAD_CONTROLNET=true
CONTROLNET_MODEL_ID=InstantX/FLUX.1-dev-Controlnet-Canny
```

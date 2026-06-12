# Deploy e Operação

## Docker local

```bash
cp .env.example .env
docker compose up --build
```

O serviço usa um worker e uma thread porque a pipeline ocupa a GPU e não deve
receber inferências concorrentes no mesmo processo. O código também usa um lock
de inferência.

## Imagem publicada

O workflow `.github/workflows/docker-build.yml`:

- Executa testes de configuração.
- Publica no GitHub Container Registry.
- Publica opcionalmente no Docker Hub.
- Usa cache de build do GitHub Actions.

Para Docker Hub, configure os secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## RunPod Serverless

Variáveis típicas:

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

Monte armazenamento persistente em:

- `/cache/huggingface` para pesos.
- `/data/conversations` para memória.

Em serverless, `PRELOAD_MODELS=false` inicia o handler rapidamente, mas a
primeira requisição paga o carregamento. Em pods persistentes,
`PRELOAD_MODELS=true` costuma ser melhor.

## Pré-carregar pesos

```bash
python scripts/preload_models.py
```

Dentro de um container com volume:

```bash
docker run --rm --gpus all \
  -v /seu/cache:/cache/huggingface \
  --env-file .env \
  sua-imagem \
  python scripts/preload_models.py
```

## Cache Cloudflare R2

O projeto pode copiar pesos do Hugging Face para R2 e baixá-los no cold start.

Configuração:

```env
R2_ENABLED=true
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=flux-models
```

Upload inicial:

```bash
python scripts/upload_to_r2.py
```

Para FLUX.2 Dev 4-bit, o script envia o modelo efetivamente carregado, definido
por `FLUX2_DEV_QUANTIZED_MODEL_ID`.

## Escalabilidade

- Uma réplica/worker por GPU.
- Coloque fila ou balanceador antes dos workers.
- Escale horizontalmente adicionando GPUs.
- Compartilhe pesos e conversas entre réplicas.
- Não aumente threads do Gunicorn para tentar paralelizar uma única GPU.
- Para tráfego constante, compare serverless com GPU dedicada.

## Observabilidade recomendada

O projeto registra modo, referências e parte do prompt. Para operação real,
adicione métricas de:

- Tempo de fila, carga e inferência.
- Cold starts.
- Falhas e OOM.
- GPU/VRAM.
- Imagens por modelo, resolução e passos.
- Crescimento do armazenamento de conversas.

## Troubleshooting

### O modelo não baixa

- Aceite a licença no Hugging Face.
- Confira `HF_TOKEN`.
- Confira acesso de rede ou cache R2.

### Falta de VRAM

- Confirme `FLUX2_DEV_4BIT=true`.
- Reduza resolução.
- Ative `MODEL_CPU_OFFLOAD=true` aceitando maior latência.
- Use FLUX.2 Klein.

### A conversa desaparece

`CONVERSATION_DIR` não está em armazenamento persistente/compartilhado.

### ControlNet retorna 409

O modelo ativo é FLUX.2. Troque para FLUX.1 ou use `/generate/edit`.


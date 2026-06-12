# Início Rápido

## Requisitos

- Docker com Docker Compose.
- GPU NVIDIA compatível.
- NVIDIA Container Toolkit.
- Aproximadamente 24 GB de VRAM para o perfil padrão FLUX.2 Dev 4-bit.
- Token Hugging Face com acesso aceito ao FLUX.2 Dev.

## Configuração

```bash
cp .env.example .env
```

Edite `.env` e configure no mínimo:

```env
HF_TOKEN=hf_seu_token
MODEL_ID=black-forest-labs/FLUX.2-dev
FLUX2_DEV_4BIT=true
FLUX2_TEXT_ENCODER_MODE=remote
MODEL_CPU_OFFLOAD=false
```

Para proteger as rotas de geração:

```env
API_KEY=uma-chave-longa-e-aleatoria
```

## Iniciar com Docker Compose

```bash
docker compose up --build
```

O Compose cria volumes persistentes para:

- Pesos e cache: `/cache/huggingface`.
- Conversas e revisões: `/data/conversations`.

## Verificar o serviço

```bash
curl http://localhost:5000/health
```

Exemplo de resposta:

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

`loaded=false` significa apenas que o carregamento lazy ainda não ocorreu.

## Gerar a primeira imagem

Resposta JSON com imagem base64:

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "prompt": "um observatório futurista no topo de uma montanha, noite estrelada",
    "seed": 42
  }'
```

Receber PNG diretamente:

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: image/png" \
  -d '{"prompt":"um observatório futurista no topo de uma montanha"}' \
  --output observatorio.png
```

Quando a resposta for PNG, os IDs da memória ficam nos headers
`X-Conversation-Id` e `X-Revision-Id`.

## Próximos passos

- [Editar imagens e usar todas as rotas](api.md)
- [Continuar uma imagem usando memória](conversation-memory.md)
- [Escolher outro modelo](models.md)


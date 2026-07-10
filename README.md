# FLUX.2 Image API

API de geração e edição de imagens pronta para Docker, GPU dedicada e RunPod
Serverless. O perfil padrão usa FLUX.2 Dev quantizado em 4-bit para estudo e
mantém memória visual entre pedidos de uma conversa.

## Documentação completa

- [Português](docs/pt-BR/README.md)
- [English](docs/en/README.md)
- [Índice bilíngue](docs/README.md)

RunPod setup, environment variables, and an example metadata-rich request
payload are documented in English:

- [RunPod deployment](docs/en/deployment.md#runpod-serverless)
- [RunPod tracking payload example](docs/en/api.md#runpod-tracking-payload-example)

## Escolha recomendada

| Perfil | Modelo | GPU | Passos | Uso |
|---|---|---:|---:|---|
| Padrão de estudo | `black-forest-labs/FLUX.2-dev` em 4-bit | 24 GB+ | 28 | Pesquisa e experimentos |
| Econômico | `black-forest-labs/FLUX.2-klein-4B` | 16 GB+ | 4 | Alta velocidade |
| Legado rápido | `black-forest-labs/FLUX.1-schnell` | 24 GB+ | 4 | Compatibilidade |

O FLUX.2-dev é o padrão deste projeto. Ele tem 32 bilhões de parâmetros e usa
os pesos abertos sob licença não comercial. O perfil 4-bit com encoder remoto
permite estudo em GPUs de aproximadamente 24 GB.

## Rodar localmente

Requisitos: Docker, NVIDIA Container Toolkit e uma GPU NVIDIA.

```bash
cp .env.example .env
docker compose up --build
```

O cache dos pesos fica no volume `huggingface-cache`. Na primeira execução o
modelo é baixado; as próximas inicializações reutilizam o volume.

```bash
curl http://localhost:5000/health

curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: image/png" \
  -d '{"prompt":"uma campanha de perfume futurista, fotografia de estúdio","seed":42}' \
  --output result.png
```

## Memória de conversa

O modelo não mantém estado interno entre chamadas. A API cria uma memória
visual: salva cada imagem gerada e envia a última imagem novamente ao FLUX.2-dev
quando você continua usando o mesmo `conversation_id`.

Primeiro pedido:

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"um carro conceito vermelho em um estúdio escuro"}'
```

A resposta contém:

```json
{
  "conversation_id": "72e5...",
  "revision_id": "f31a...",
  "revision_count": 1
}
```

Para pedir uma alteração, envie somente o ID e a nova instrução:

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id":"72e5...",
    "prompt":"mantenha o carro, mas troque a cor para azul e aumente a iluminação"
  }'
```

Use `use_previous=false` para iniciar uma composição nova mantendo o mesmo ID,
ou `remember=false` para não armazenar aquela geração. Consulte o histórico com
`GET /conversations/<conversation_id>` e apague com
`DELETE /conversations/<conversation_id>`.

Para voltar a uma imagem antiga da conversa, envie também o `revision_id`
retornado pelo histórico. A nova imagem será criada a partir daquela revisão.

O histórico fica em `CONVERSATION_DIR`. Em RunPod Serverless, esse diretório
precisa estar em um Network Volume compartilhado para sobreviver a cold starts.

## Editar com referências

FLUX.2 usa a mesma pipeline para geração, edição e múltiplas referências.

```bash
curl -X POST http://localhost:5000/generate/edit \
  -H "Accept: image/png" \
  -F "prompt=coloque o produto da primeira imagem no cenário da segunda" \
  -F "images=@produto.png" \
  -F "images=@cenario.png" \
  --output edit.png
```

Também é possível enviar JSON com `image` ou `images` em base64. A rota antiga
`/generate/img2img` continua funcionando.

Parâmetros disponíveis:

| Campo | Padrão Dev | Descrição |
|---|---:|---|
| `prompt` | obrigatório | Instrução de geração ou edição |
| `images` | `[]` | Até quatro referências em base64 |
| `width`, `height` | `1024` | Entre 256 e 2048, divisível por 16 |
| `num_inference_steps` | `28` | Mais passos aumentam custo |
| `guidance_scale` | `4.0` | Aderência ao prompt |
| `seed` | aleatório | Reprodutibilidade |
| `conversation_id` | gerado | Continua uma edição anterior |
| `revision_id` | última revisão | Escolhe uma imagem antiga da conversa |

## FLUX.2-dev em uma GPU de 24 GB

Aceite a licença do modelo no Hugging Face e configure:

```env
MODEL_ID=black-forest-labs/FLUX.2-dev
HF_TOKEN=hf_...
FLUX2_DEV_4BIT=true
FLUX2_TEXT_ENCODER_MODE=remote
MODEL_CPU_OFFLOAD=false
```

Esse perfil carrega `diffusers/FLUX.2-dev-bnb-4bit` e usa o encoder de texto
remoto indicado pela documentação oficial. Para eliminar a chamada externa,
use `FLUX2_TEXT_ENCODER_MODE=local`, mas prepare uma máquina com muito mais RAM
e VRAM.

## Deploy escalável

### RunPod Serverless

1. Publique a imagem pelo workflow GitHub Actions.
2. Crie um endpoint RunPod usando `ghcr.io/SEU_USUARIO/SEU_REPO:latest`.
3. Anexe um Network Volume ao endpoint. Em RunPod Serverless ele monta em
   `/runpod-volume`.
4. Configure `HF_HOME=/runpod-volume/huggingface` para persistir o cache dos
   pesos e `CONVERSATION_DIR=/runpod-volume/conversations` para persistir a
   memória.
5. Configure `RUNPOD_ENABLED=true`, `MODEL_ID` e `HF_TOKEN`.
6. Use um worker por GPU e escale o número de workers pela fila.

Não coloque os pesos dentro da imagem Docker. Isso deixa o build enorme e torna
cada atualização lenta. O container deve ser pequeno e os pesos devem ficar no
Network Volume ou no cache R2 já suportado pelo projeto.

### GPU dedicada

Para tráfego constante, uma GPU dedicada com `PRELOAD_MODELS=true` normalmente
custa menos que serverless e elimina cold start. Rode várias réplicas, cada uma
com uma GPU, atrás de um balanceador ou fila. O processo já serializa inferências
por GPU para evitar estouro de VRAM.

### Publicação Docker

O workflow publica sempre no GHCR. Para publicar também no Docker Hub, crie os
secrets `DOCKERHUB_USERNAME` e `DOCKERHUB_TOKEN` no GitHub.

## Variáveis importantes

Consulte [.env.example](.env.example). As principais são:

- `MODEL_ID`: modelo carregado.
- `MODEL_CPU_OFFLOAD`: reduz VRAM e aumenta latência.
- `PRELOAD_MODELS`: carrega o modelo ao iniciar.
- `RUNPOD_ENABLED`: ativa o handler serverless.
- `HF_HOME`: diretório persistente dos pesos.
- `R2_ENABLED`: usa o cache Cloudflare R2 existente.
- `API_KEY`: protege as rotas de geração quando configurada.
- `CONVERSATION_DIR`: diretório persistente das imagens e revisões.

## Segurança

Antes de expor a API publicamente, adicione autenticação, limite por cliente,
moderação de prompt/imagem e armazenamento assíncrono do resultado. As licenças
e políticas da BFL exigem atenção especial para modelos não comerciais.

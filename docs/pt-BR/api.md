# Referência da API

Base URL local padrão: `http://localhost:5000`

Quando `API_KEY` estiver configurada, envie uma destas opções:

```http
X-API-Key: sua-chave
Authorization: Bearer sua-chave
```

`GET /health` não exige autenticação.

## Parâmetros comuns

| Campo | Tipo | Padrão FLUX.2 Dev | Regras |
|---|---|---:|---|
| `prompt` | string | obrigatório | Não pode ser vazio |
| `image` | base64 ou arquivo | nenhum | Atalho para uma referência |
| `images` | lista base64 ou arquivos | `[]` | Máximo de 4 |
| `width` | inteiro | `1024` | 256–2048, divisível por 16 |
| `height` | inteiro | `1024` | 256–2048, divisível por 16 |
| `num_inference_steps` | inteiro | `28` | 1–100 |
| `guidance_scale` | número | `4.0` | 0–20 |
| `seed` | inteiro | aleatório | Reprodutibilidade |
| `strength` | número | `0.75` | 0–1, usado em img2img FLUX.1 |
| `conversation_id` | string | gerado | Letras, números, `_` e `-`, máximo 80 |
| `revision_id` | string | revisão mais recente | Seleciona revisão anterior |
| `use_previous` | boolean/string | `true` | Reutiliza imagem da conversa |
| `remember` | boolean/string | `true` | Salva resultado na memória |

## Formatos de resposta

O padrão é JSON:

```json
{
  "image": "iVBORw0KGgo...",
  "format": "png",
  "prompt": "descrição",
  "width": 1024,
  "height": 1024,
  "conversation_id": "abc123",
  "revision_id": "def456",
  "revision_count": 1
}
```

Para receber PNG binário, envie `Accept: image/png` ou
`X-Response-Format: bytes`.

## `GET /health`

Retorna estado, modelo configurado, família, disponibilidade CUDA e padrões.
Esse endpoint não força o carregamento do modelo.

## `POST /generate`

Gera uma imagem a partir de texto. Também edita automaticamente a última imagem
quando `conversation_id` aponta para uma conversa existente.

```bash
curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "prompt": "uma biblioteca brutalista com jardins internos",
    "width": 1024,
    "height": 1024,
    "seed": 7
  }'
```

## `POST /generate/edit`

Exige uma referência explícita ou uma imagem recuperável pela memória.
`POST /generate/img2img` é um alias compatível.

Multipart com múltiplas referências:

```bash
curl -X POST "$BASE_URL/generate/edit" \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: image/png" \
  -F "prompt=use o produto da primeira imagem no cenário da segunda" \
  -F "images=@produto.png" \
  -F "images=@cenario.png" \
  --output resultado.png
```

JSON base64:

```json
{
  "prompt": "transforme em uma cena noturna",
  "images": ["BASE64_DA_IMAGEM_1", "BASE64_DA_IMAGEM_2"]
}
```

Referências explícitas têm prioridade sobre a imagem anterior da conversa.

## `POST /generate/controlnet`

Disponível somente quando um modelo FLUX.1 está configurado. A aplicação extrai
bordas Canny da imagem e usa o adapter `CONTROLNET_MODEL_ID`.

Campos adicionais:

| Campo | Padrão |
|---|---:|
| `controlnet_conditioning_scale` | `0.7` |
| `canny_low_threshold` | `100` |
| `canny_high_threshold` | `200` |

Quando FLUX.2 está ativo, a rota retorna HTTP `409`. Use `/generate/edit`.

## Conversas

```http
GET /conversations/<conversation_id>
DELETE /conversations/<conversation_id>
```

O `GET` retorna o manifesto com prompts, datas e IDs de revisão. O `DELETE`
remove manifesto e imagens armazenadas.

## Erros comuns

| Status | Significado |
|---:|---|
| `400` | Prompt ausente, dimensão inválida, revisão inexistente ou parâmetros inválidos |
| `401` | API key ausente/incorreta |
| `404` | Conversa não encontrada |
| `409` | Tentativa de usar ControlNet com FLUX.2 |
| `413` | Payload maior que `MAX_REQUEST_MB` |
| `500` | Falha durante carga ou inferência |

## RunPod Serverless

O handler recebe o conteúdo em `input`:

```json
{
  "input": {
    "mode": "generate",
    "prompt": "uma cidade solar flutuante",
    "conversation_id": "estudo-01",
    "seed": 42
  }
}
```

Modos válidos: `generate`, `txt2img`, `edit` e `img2img`. ControlNet não está
exposto pelo handler RunPod atual.


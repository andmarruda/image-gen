# API Reference

Default local base URL: `http://localhost:5000`

When `API_KEY` is configured, send either:

```http
X-API-Key: your-key
Authorization: Bearer your-key
```

`GET /health` does not require authentication.

## Common parameters

| Field | Type | FLUX.2 Dev default | Rules |
|---|---|---:|---|
| `prompt` | string | required | Must not be empty |
| `image` | base64 or file | none | Shortcut for one reference |
| `images` | base64 list or files | `[]` | Maximum 4 |
| `width` | integer | `1024` | 256–2048, divisible by 16 |
| `height` | integer | `1024` | 256–2048, divisible by 16 |
| `num_inference_steps` | integer | `28` | 1–100 |
| `guidance_scale` | number | `4.0` | 0–20 |
| `seed` | integer | random | Reproducibility |
| `strength` | number | `0.75` | 0–1, used by FLUX.1 img2img |
| `conversation_id` | string | generated | Letters, numbers, `_`, `-`; max 80 |
| `revision_id` | string | latest revision | Selects an older revision |
| `use_previous` | boolean/string | `true` | Reuses conversation image |
| `remember` | boolean/string | `true` | Stores the result |

## Response formats

JSON is the default:

```json
{
  "image": "iVBORw0KGgo...",
  "format": "png",
  "prompt": "description",
  "width": 1024,
  "height": 1024,
  "conversation_id": "abc123",
  "revision_id": "def456",
  "revision_count": 1
}
```

Send `Accept: image/png` or `X-Response-Format: bytes` for raw PNG output.

## `GET /health`

Returns service status, configured model, model family, CUDA availability, and
defaults. It does not force the model to load.

## `POST /generate`

Generates from text. It also edits the latest image when `conversation_id`
points to an existing conversation.

```bash
curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "prompt": "a brutalist library with indoor gardens",
    "width": 1024,
    "height": 1024,
    "seed": 7
  }'
```

## `POST /generate/edit`

Requires an explicit reference or an image available through conversation
memory. `POST /generate/img2img` is a compatibility alias.

Multipart with multiple references:

```bash
curl -X POST "$BASE_URL/generate/edit" \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: image/png" \
  -F "prompt=place the product from the first image in the second image scene" \
  -F "images=@product.png" \
  -F "images=@scene.png" \
  --output result.png
```

Base64 JSON:

```json
{
  "prompt": "turn this into a night scene",
  "images": ["BASE64_IMAGE_1", "BASE64_IMAGE_2"]
}
```

Explicit references take precedence over the previous conversation image.

## `POST /generate/controlnet`

Available only when a FLUX.1 model is configured. The application extracts
Canny edges and uses `CONTROLNET_MODEL_ID`.

Additional fields:

| Field | Default |
|---|---:|
| `controlnet_conditioning_scale` | `0.7` |
| `canny_low_threshold` | `100` |
| `canny_high_threshold` | `200` |

With FLUX.2 active, this route returns HTTP `409`. Use `/generate/edit`.

## Conversations

```http
GET /conversations/<conversation_id>
DELETE /conversations/<conversation_id>
```

`GET` returns prompts, dates, and revision IDs. `DELETE` removes the manifest
and stored images.

## Common errors

| Status | Meaning |
|---:|---|
| `400` | Missing prompt, invalid dimensions/parameters, or missing revision |
| `401` | Missing or invalid API key |
| `404` | Conversation not found |
| `409` | ControlNet attempted with FLUX.2 |
| `413` | Payload exceeds `MAX_REQUEST_MB` |
| `500` | Model loading or inference failure |

## RunPod Serverless

The handler receives fields inside `input`:

```json
{
  "input": {
    "mode": "generate",
    "prompt": "a floating solar city",
    "conversation_id": "study-01",
    "seed": 42
  }
}
```

Valid modes: `generate`, `txt2img`, `edit`, and `img2img`. ControlNet is not
exposed through the current RunPod handler.

### RunPod tracking payload example

Use this shape when the caller needs to keep tenant, user, brand, preset, and
session context next to each generated image for later evaluation or training
work:

```json
{
  "input": {
    "mode": "generate",
    "prompt": "premium skincare product on a clean reflective surface, soft studio lighting, realistic advertising photography, pastel botanical background, no visible text",
    "negative_prompt": "text, letters, words, captions, logos, watermarks, brand marks, distorted hands, extra fingers, low quality, blurry, noisy",
    "width": 1080,
    "height": 1080,
    "format": "png",
    "response_format": "base64",
    "seed": null,
    "metadata": {
      "tenant_id": 123,
      "user_id": 456,
      "brand_id": "789",
      "intent": "post",
      "preset_id": "instagram_post",
      "agent_session_id": "uuid-da-sessao"
    }
  }
}
```

Current handler behavior:

- `prompt`, `width`, `height`, `seed`, `mode`, `image`, `images`, and generation
  parameters are used by inference.
- The RunPod handler always returns JSON with a base64 PNG in `image`.
- `metadata`, `negative_prompt`, `format`, and `response_format` are useful
  caller-side tracking fields today. Store them with the job request/response if
  you plan to build evaluation datasets later.

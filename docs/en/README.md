# English Documentation

This API provides image generation and editing with FLUX.2 and FLUX.1, visual
conversation memory, a traditional HTTP server, and RunPod Serverless support.

## Start here

1. [Quickstart](quickstart.md): configure the environment and generate an image.
2. [API reference](api.md): routes, fields, responses, and errors.
3. [Conversation memory](conversation-memory.md): iterative edits and old revisions.
4. [Models](models.md): choose FLUX.2 Dev, FLUX.2 Klein, or FLUX.1.
5. [Configuration](configuration.md): environment-variable reference.
6. [Deployment and operations](deployment.md): Docker, RunPod, R2, and scaling.
7. [Security and guardrails](guardrails.md): existing protections and recommendations.

## Capabilities

- Text-to-image generation.
- Editing with one to four reference images.
- Persistent visual memory through `conversation_id`.
- Revision history and old-revision selection.
- Base64 JSON or raw PNG responses.
- FLUX.2 Dev 4-bit as the default study profile.
- FLUX.2 Klein, FLUX.1 Schnell, and FLUX.1 Dev compatibility.
- Canny ControlNet when a FLUX.1 model is active.
- Persistent Hugging Face or Cloudflare R2 model cache.
- HTTP API or RunPod Serverless handler.

## Important memory limitation

The model has no internal state between requests. The application saves each
generated image and submits it again as a reference on the next conversation
request. This provides visual continuity, but every edit is still a new
inference.

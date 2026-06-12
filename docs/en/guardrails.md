# Security and Guardrails

## Implemented protections

| Protection | Behavior |
|---|---|
| Optional API key | Protects every route except `/health` |
| Safe comparison | Uses constant-time key comparison |
| Payload limit | `MAX_REQUEST_MB`, default 32 MB |
| Resolution limit | 256–2048, divisible by 16 |
| Step limit | 1–100 |
| Guidance limit | 0–20 |
| Reference limit | Maximum 4 |
| Validated IDs | Prevents traversal through conversation/revision IDs |
| Bounded history | `CONVERSATION_MAX_REVISIONS` |
| Serialized inference | Prevents unsafe concurrent calls on one GPU |
| Conversation deletion | Removes stored prompts and images |

Configure authentication:

```env
API_KEY=generate-a-long-random-key
MAX_REQUEST_MB=32
```

## Not implemented

- Prompt moderation.
- Input or generated-image moderation.
- Per-user/IP rate limiting.
- GPU or budget quotas.
- Conversation ownership authorization.
- Storage encryption for images.
- Automatic conversation expiration.
- Structured audit logging.
- Prompt-injection filtering for external systems.

Do not expose the API directly to the internet assuming `API_KEY` solves all
these concerns.

## Recommended guardrail architecture

```text
Client
  -> Gateway with authentication, rate limits, and cost controls
  -> Prompt and input-image moderation
  -> Generation queue
  -> FLUX worker
  -> Generated-image moderation
  -> Storage/delivery
```

## Using FLUX.1 or FLUX.2 as a guardrail

A generator may experimentally help to:

- Normalize or reconstruct an image.
- Create controlled-style versions.
- Compare visual consistency.

However, FLUX.1/2 are not safety classifiers and may fail silently. Use a
dedicated moderation model/service before and after generation.

## Protecting memory

Currently, anyone with API access and a known `conversation_id` can read or
delete that conversation. In a multi-user environment:

- Associate conversations with authenticated users in a database.
- Authorize `GET` and `DELETE` by owner.
- Use unpredictable IDs.
- Define retention and expiration.
- Avoid sensitive data in prompts and images.

## Cost and abuse controls

Technical limits reduce risk but do not control budget. Recommendations:

- Gateway rate limiting.
- Maximum concurrent jobs per user.
- Daily image or GPU-second quotas.
- Allowed resolutions and step counts by plan.
- Job timeouts and cancellation.
- Cost and utilization alerts.


# Conversation Memory

## How it works

FLUX has no internal state between calls. The API implements visual memory:

1. Generate an image.
2. Save `latest.png`, a PNG revision, and `manifest.json`.
3. Return `conversation_id` and `revision_id`.
4. Load the previous image on the next call with the same conversation ID.
5. Submit that image as a model reference with the new instruction.

## Create and continue a conversation

First generation:

```bash
curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a yellow gardening robot inside a greenhouse"}'
```

Continue:

```bash
curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id":"RETURNED_ID",
    "prompt":"keep the robot and composition, but change the scene to sunrise"
  }'
```

For better results, explicitly state what must remain and what must change.
Memory provides the image; it does not interpret vague intent.

## Return to an older revision

Read history:

```bash
curl "$BASE_URL/conversations/RETURNED_ID"
```

Then submit a `revision_id`:

```json
{
  "conversation_id": "RETURNED_ID",
  "revision_id": "OLD_REVISION",
  "prompt": "create a different visual direction from this version"
}
```

## Controls

- `use_previous=false`: do not use the latest image, but save the result in the same conversation.
- `remember=false`: do not retrieve or save memory for this call.
- Explicit `image`/`images`: overrides the automatic conversation reference.
- `CONVERSATION_MAX_REVISIONS`: limits stored revisions and deletes old ones.
- `CONVERSATION_MEMORY_ENABLED=false`: disables memory globally.

## Persistence and scaling

The default path is `/data/conversations`. Docker Compose provides a persistent
volume.

For RunPod Serverless or multiple replicas, use shared storage. Without a
Network Volume, one worker may not find another worker's conversation, and
conversations disappear during cold starts.

The current implementation uses local files and a per-process lock. It is
appropriate for study. For distributed high concurrency, consider:

- Object storage for PNG files.
- A database for manifests.
- Distributed locks or per-conversation queues.
- User identity bound to each conversation.

## Privacy

Images and prompts remain stored until deletion or revision pruning. Delete a
conversation when finished:

```bash
curl -X DELETE "$BASE_URL/conversations/RETURNED_ID"
```


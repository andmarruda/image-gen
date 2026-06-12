import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")
_lock = threading.Lock()


def enabled() -> bool:
    return os.getenv("CONVERSATION_MEMORY_ENABLED", "true").lower() == "true"


def _root() -> Path:
    return Path(os.getenv("CONVERSATION_DIR", "/data/conversations"))


def normalize_id(value: str | None) -> str:
    if value is None:
        return uuid.uuid4().hex
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError("conversation_id must contain only letters, numbers, '_' or '-'")
    return value


def _conversation_dir(conversation_id: str) -> Path:
    return _root() / normalize_id(conversation_id)


def _manifest_path(conversation_id: str) -> Path:
    return _conversation_dir(conversation_id) / "manifest.json"


def get_manifest(conversation_id: str) -> dict | None:
    path = _manifest_path(conversation_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest(conversation_id: str):
    from PIL import Image

    path = _conversation_dir(conversation_id) / "latest.png"
    if not path.exists():
        return None
    with Image.open(path) as image:
        return image.convert("RGB")


def load_revision(conversation_id: str, revision_id: str):
    from PIL import Image

    revision_id = normalize_id(revision_id)
    path = _conversation_dir(conversation_id) / "revisions" / f"{revision_id}.png"
    if not path.exists():
        return None
    with Image.open(path) as image:
        return image.convert("RGB")


def save_revision(conversation_id: str, image: Any, prompt: str) -> dict:
    conversation_id = normalize_id(conversation_id)
    revision_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    directory = _conversation_dir(conversation_id)
    revisions_dir = directory / "revisions"
    max_revisions = max(1, int(os.getenv("CONVERSATION_MAX_REVISIONS", "10")))

    with _lock:
        revisions_dir.mkdir(parents=True, exist_ok=True)
        revision_path = revisions_dir / f"{revision_id}.png"
        image.save(revision_path, format="PNG")
        image.save(directory / "latest.png", format="PNG")

        manifest = get_manifest(conversation_id) or {
            "conversation_id": conversation_id,
            "created_at": created_at,
            "revisions": [],
        }
        manifest["updated_at"] = created_at
        manifest["latest_revision_id"] = revision_id
        manifest["revisions"].append(
            {"revision_id": revision_id, "created_at": created_at, "prompt": prompt}
        )

        removed = manifest["revisions"][:-max_revisions]
        manifest["revisions"] = manifest["revisions"][-max_revisions:]
        for revision in removed:
            old_path = revisions_dir / f"{revision['revision_id']}.png"
            old_path.unlink(missing_ok=True)

        temporary = directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
        temporary.replace(_manifest_path(conversation_id))
        return manifest


def delete(conversation_id: str) -> bool:
    import shutil

    directory = _conversation_dir(conversation_id)
    if not directory.exists():
        return False
    with _lock:
        shutil.rmtree(directory)
    return True


def prepare(data: dict) -> tuple[dict, str | None]:
    if not enabled() or str(data.get("remember", "true")).lower() == "false":
        return data, None

    prepared = dict(data)
    conversation_id = normalize_id(prepared.pop("conversation_id", None))
    use_previous = str(prepared.pop("use_previous", "true")).lower() != "false"
    revision_id = prepared.pop("revision_id", None)
    has_explicit_images = prepared.get("images") is not None or prepared.get("image") is not None

    if use_previous and not has_explicit_images:
        previous = (
            load_revision(conversation_id, revision_id)
            if revision_id is not None
            else load_latest(conversation_id)
        )
        if revision_id is not None and previous is None:
            raise ValueError("revision_id was not found in this conversation")
        if previous is not None:
            prepared["images"] = [previous]

    return prepared, conversation_id

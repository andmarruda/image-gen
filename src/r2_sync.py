"""
Cloudflare R2 model cache — upload once, cold-start anywhere.

Upload flow (one-time, run from any machine with HF access):
    python scripts/upload_to_r2.py

Download flow (automatic on cold start when R2_ENABLED=true):
    Called from app.py before model loading.
    Downloads model files from R2 to local disk, then pipeline loads from disk.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Sentinel file written after a successful download — prevents re-downloading
# on warm workers where the container disk persists between jobs.
_COMPLETE_MARKER = ".r2_complete"


def _client():
    import boto3

    account_id = os.getenv("R2_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("R2_ACCOUNT_ID is not set")

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


def _local_dir(model_id: str) -> Path:
    """
    Returns the local directory where the model will be stored.

    Example:
        model_id = "black-forest-labs/FLUX.1-dev"
        → {HF_HOME}/r2/black-forest-labs--FLUX.1-dev/
    """
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    safe_name = model_id.replace("/", "--")
    return Path(hf_home) / "r2" / safe_name


def is_cached(model_id: str) -> bool:
    """Returns True if this model was fully downloaded from R2."""
    return (_local_dir(model_id) / _COMPLETE_MARKER).exists()


def local_path(model_id: str) -> str | None:
    """
    Returns the local path for a model downloaded from R2, or None if not cached.
    pipeline.py uses this to decide between local load and HuggingFace download.
    """
    d = _local_dir(model_id)
    if (d / _COMPLETE_MARKER).exists():
        return str(d)
    return None


def download(model_id: str) -> str:
    """
    Downloads all model files from R2 to local disk.
    Skips files that are already present (resume-safe).
    Returns the local directory path.
    """
    if is_cached(model_id):
        logger.info("R2 cache hit for %s — skipping download.", model_id)
        return str(_local_dir(model_id))

    bucket = os.getenv("R2_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("R2_BUCKET_NAME is not set")

    client = _client()
    local = _local_dir(model_id)
    local.mkdir(parents=True, exist_ok=True)

    prefix = f"{model_id}/"
    paginator = client.get_paginator("list_objects_v2")

    files_downloaded = 0
    files_skipped = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(prefix):]
            if not rel_path or rel_path == _COMPLETE_MARKER:
                continue

            dest = local / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() and dest.stat().st_size == obj["Size"]:
                files_skipped += 1
                continue

            logger.info("[R2 ↓] %s", rel_path)
            client.download_file(bucket, key, str(dest))
            files_downloaded += 1

    # Write sentinel so warm workers skip the download entirely
    (local / _COMPLETE_MARKER).touch()

    logger.info(
        "R2 download complete for %s — %d downloaded, %d skipped.",
        model_id,
        files_downloaded,
        files_skipped,
    )
    return str(local)


def upload(model_id: str, hf_token: str | None = None) -> None:
    """
    Downloads a model from HuggingFace and uploads all files to R2.
    Skips files already present in R2 (resume-safe).
    Intended to run once from a CPU machine before deploying to RunPod.
    """
    import tempfile

    from huggingface_hub import snapshot_download

    bucket = os.getenv("R2_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("R2_BUCKET_NAME is not set")

    client = _client()

    # Build index of files already in R2 to skip re-uploading
    logger.info("Indexing existing R2 objects for %s ...", model_id)
    existing: dict[str, int] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{model_id}/"):
        for obj in page.get("Contents", []):
            existing[obj["Key"]] = obj["Size"]
    logger.info("Found %d existing objects in R2.", len(existing))

    with tempfile.TemporaryDirectory() as tmp:
        logger.info("Downloading %s from HuggingFace ...", model_id)
        kwargs: dict = {"local_dir": tmp, "local_dir_use_symlinks": False}
        if hf_token:
            kwargs["token"] = hf_token
        snapshot_download(model_id, **kwargs)

        files = [p for p in Path(tmp).rglob("*") if p.is_file()]
        logger.info("Uploading %d files to R2 bucket %s ...", len(files), bucket)

        uploaded = 0
        skipped = 0
        for file_path in files:
            rel = file_path.relative_to(tmp)
            key = f"{model_id}/{rel}"

            if key in existing and existing[key] == file_path.stat().st_size:
                logger.info("[R2 ↑ skip] %s", rel)
                skipped += 1
                continue

            logger.info("[R2 ↑] %s", rel)
            client.upload_file(str(file_path), bucket, key)
            uploaded += 1

    logger.info(
        "R2 upload complete for %s — %d uploaded, %d skipped.",
        model_id,
        uploaded,
        skipped,
    )

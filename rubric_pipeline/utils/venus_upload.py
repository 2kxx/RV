# -*- coding: utf-8 -*-
"""Upload videos to Tencent Venus temporary storage and return Gemini
multimodal_url descriptors usable by venus llmproxy.

Two-step flow (mirrors the GeminiApi reference impl provided by the user):
    1. POST {model, filename} to ``/chat/temp/storage/info`` with
       Venus secret_id/secret_key signing → receive ``{putUrl, bucketName, filepath}``.
    2. PUT the raw video bytes to ``putUrl``.
    3. Return ``{bucketName, filepath, mimeType}`` for embedding into a
       ``gemini_multimodal_url`` content part of an OpenAI-compatible request.

A JSON disk cache (keyed by ``sha256(local_path) + size + mtime`` or by remote
URL) avoids re-uploading the same clip across pipeline stages and across
variants (rubric-gen / rubric-judge / zero-shot / positional-bias). The cache
file path can be overridden with ``ARR_VENUS_UPLOAD_CACHE``; default is
``runs/v_arr_mvp/_venus_upload_cache.json``.

Auth env vars (read lazily at first upload):
    VENUS_SECRET_ID
    VENUS_SECRET_KEY

Optional env vars:
    ARR_VENUS_MODEL          (default: env MODEL or "gemini-3.1-pro")
    ARR_VENUS_UPLOAD_URL     (default: http://v2.open.venus.oa.com/chat/temp/storage/info)
    ARR_VENUS_UPLOAD_CACHE   (default: runs/v_arr_mvp/_venus_upload_cache.json)
    ARR_VENUS_TMP_DIR        (default: tempfile.gettempdir() + /arr_venus_dl)
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class VenusUploadError(RuntimeError):
    """Raised when uploading a video to Venus temp storage fails."""


_DEFAULT_UPLOAD_URL = "http://v2.open.venus.oa.com/chat/temp/storage/info"
_DEFAULT_CACHE_PATH = "runs/v_arr_mvp/_venus_upload_cache.json"

_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, str]] | None = None
_cache_path: Path | None = None

_client_lock = threading.Lock()
_venus_client = None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _resolve_cache_path() -> Path:
    global _cache_path
    if _cache_path is not None:
        return _cache_path
    raw = os.environ.get("ARR_VENUS_UPLOAD_CACHE", _DEFAULT_CACHE_PATH)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    _cache_path = p
    return p


def _load_cache() -> Dict[str, Dict[str, str]]:
    global _cache
    if _cache is not None:
        return _cache
    p = _resolve_cache_path()
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                _cache = json.load(f)
                if not isinstance(_cache, dict):
                    _cache = {}
        except Exception as e:  # pragma: no cover - corrupt cache, start fresh
            logger.warning("[venus_upload] cache load failed (%s); resetting", e)
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save_cache() -> None:
    cache = _load_cache()
    p = _resolve_cache_path()
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def _make_local_cache_key(local_path: Path) -> str:
    st = local_path.stat()
    h = hashlib.sha256()
    h.update(str(local_path.resolve()).encode("utf-8"))
    h.update(f"|{st.st_size}|{int(st.st_mtime)}".encode("utf-8"))
    return f"file://{h.hexdigest()}"


def _make_remote_cache_key(url: str) -> str:
    return f"url://{url}"


# ---------------------------------------------------------------------------
# Remote helpers
# ---------------------------------------------------------------------------


def _is_remote_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _download_to_tmp(url: str) -> Path:
    """Download a remote video to a deterministic tmp file and return the path."""
    import requests

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or ".mp4"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]

    tmp_root = Path(
        os.environ.get("ARR_VENUS_TMP_DIR")
        or (Path(tempfile.gettempdir()) / "arr_venus_dl"),
    )
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_root / f"{digest}{suffix}"

    if tmp_path.exists() and tmp_path.stat().st_size > 0:
        return tmp_path

    logger.info("[venus_upload] downloading %s -> %s", url, tmp_path)
    resp = requests.get(url, stream=True, timeout=180)
    if resp.status_code != 200:
        raise VenusUploadError(
            f"failed to download {url}: HTTP {resp.status_code}",
        )
    with tmp_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return tmp_path


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("video/"):
        return mime
    return "video/mp4"


# ---------------------------------------------------------------------------
# Venus client
# ---------------------------------------------------------------------------


def _get_venus_client():
    """Lazily build a Venus signing HTTP client. Raises VenusUploadError if
    venus_api_base is unavailable or credentials are missing."""
    global _venus_client
    if _venus_client is not None:
        return _venus_client

    with _client_lock:
        if _venus_client is not None:
            return _venus_client

        try:
            from venus_api_base.http_client import HttpClient as VenusHttpClient
            from venus_api_base.config import Config as VenusConfig
        except ImportError as exc:  # pragma: no cover
            raise VenusUploadError(
                "ARR_VIDEO_BLOCK_TYPE=gemini_multimodal_url requires the "
                "`venus-api-base` package. Install it with "
                "`pip install venus-api-base`.",
            ) from exc

        secret_id = os.environ.get("VENUS_SECRET_ID", "").strip()
        secret_key = os.environ.get("VENUS_SECRET_KEY", "").strip()
        if not secret_id or not secret_key:
            raise VenusUploadError(
                "VENUS_SECRET_ID / VENUS_SECRET_KEY environment variables "
                "are required to upload videos to venus temp storage.",
            )

        _venus_client = VenusHttpClient(
            secret_id=secret_id,
            secret_key=secret_key,
            config=VenusConfig(read_timeout=500),
        )
        return _venus_client


def _request_upload_slot(local_path: Path) -> Dict[str, Any]:
    client = _get_venus_client()
    upload_url = os.environ.get("ARR_VENUS_UPLOAD_URL", _DEFAULT_UPLOAD_URL)
    model = (
        os.environ.get("ARR_VENUS_MODEL")
        or os.environ.get("MODEL")
        or "gemini-3.1-pro"
    )
    body = {"model": model, "filename": str(local_path)}
    ret = client.post(
        url=upload_url,
        header={"Content-Type": "application/json"},
        body=json.dumps(body),
    )
    if not isinstance(ret, dict) or ret.get("code") != 0:
        msg = ret.get("message") if isinstance(ret, dict) else str(ret)
        raise VenusUploadError(f"venus preSign failed: {msg}")
    data = ret.get("data") or {}
    if not data.get("putUrl") or not data.get("bucketName") or not data.get("filepath"):
        raise VenusUploadError(f"venus preSign missing fields: {data}")
    return data


def _put_bytes(put_url: str, local_path: Path) -> None:
    import requests

    with local_path.open("rb") as f:
        resp = requests.put(put_url, data=f, timeout=600)
    if resp.status_code not in (200, 201, 204):
        raise VenusUploadError(
            f"venus PUT upload failed (HTTP {resp.status_code}): {resp.text[:500]}",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upload_video_to_venus(source: str) -> Dict[str, str]:
    """Upload a video to Venus temp storage. Cached by source identity.

    Args:
        source: A local file path OR a remote http(s) URL pointing to a video.

    Returns:
        ``{"bucketName": str, "filepath": str, "mimeType": str}``

    Raises:
        VenusUploadError on any failure (auth, preSign, HTTP PUT, ...).
    """
    if not source or not isinstance(source, str):
        raise VenusUploadError(f"invalid video source: {source!r}")

    # ---- Resolve cache key + concrete local path ---------------------------
    if _is_remote_url(source):
        cache_key = _make_remote_cache_key(source)
        cache = _load_cache()
        with _cache_lock:
            cached = cache.get(cache_key)
        if cached:
            return dict(cached)
        local_path = _download_to_tmp(source)
    else:
        local_path = Path(source).expanduser()
        if not local_path.is_file():
            raise VenusUploadError(f"local video not found: {local_path}")
        cache_key = _make_local_cache_key(local_path)
        cache = _load_cache()
        with _cache_lock:
            cached = cache.get(cache_key)
        if cached:
            return dict(cached)

    # ---- Upload -----------------------------------------------------------
    logger.info("[venus_upload] uploading %s", local_path)
    slot = _request_upload_slot(local_path)
    _put_bytes(slot["putUrl"], local_path)

    info = {
        "bucketName": slot["bucketName"],
        "filepath": slot["filepath"],
        "mimeType": _guess_mime(local_path),
    }

    # ---- Persist cache ----------------------------------------------------
    with _cache_lock:
        cache = _load_cache()
        cache[cache_key] = info
        try:
            _save_cache()
        except Exception as e:  # pragma: no cover - non-fatal
            logger.warning("[venus_upload] cache save failed: %s", e)

    return info

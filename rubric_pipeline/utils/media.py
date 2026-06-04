# -*- coding: utf-8 -*-
"""Utilities for image and video Auto-Rubric inputs.

Unifies task-type normalization, path extraction, and VLM message attachment
for text-to-image, image editing, text-to-video, and video editing tasks.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from rubric_pipeline.utils.vision import (
    TASK_IMAGE_EDIT,
    TASK_T2I,
    _as_path_list,
    _first_path_from_keys,
    add_images_to_messages,
    count_outputs as count_image_outputs,
    extract_image_paths,
    has_base_image,
    image_to_base64_url,
    is_image_edit_task,
    normalize_task_type as normalize_image_task_type,
    split_image_paths,
)
from rubric_pipeline.utils.venus_upload import (
    upload_video_to_venus,
    VenusUploadError,
)

TASK_T2V = "t2v"
TASK_VIDEO_EDIT = "video_edit"
# EvalVerse extended video tasks
TASK_R2V = "r2v"
TASK_T2V_MULTISHOT = "t2v_multishot"
TASK_T2V_SOUND = "t2v_sound"

VIDEO_INPUT_NATIVE = "native_video"
VIDEO_INPUT_UNIFORM_FRAMES = "uniform_frames"
VideoInputStrategy = Literal["native_video", "uniform_frames", "keyframe_plus_clip"]

_IMAGE_TASKS = {TASK_T2I, TASK_IMAGE_EDIT}
_VIDEO_TASKS = {TASK_T2V, TASK_VIDEO_EDIT, TASK_R2V, TASK_T2V_MULTISHOT, TASK_T2V_SOUND}


def normalize_task_type(task_type: str | None) -> str:
    """Normalize task aliases to one of the supported task types."""
    value = (task_type or TASK_T2I).strip().lower().replace("-", "_")
    if value in {
        "t2v",
        "text_to_video",
        "text2video",
        "video_generation",
        "t2v_gen",
    }:
        return TASK_T2V
    if value in {
        "video_edit",
        "video_editing",
        "v2v",
        "video_to_video",
    }:
        return TASK_VIDEO_EDIT
    if value in {
        "r2v",
        "reference_to_video",
        "subject_to_video",
        "object_to_video",
        "scene_to_video",
        "camera_to_video",
    }:
        return TASK_R2V
    if value in {
        "t2v_multishot",
        "text_to_multi_shot_video",
        "multi_shot",
        "multishot",
    }:
        return TASK_T2V_MULTISHOT
    if value in {
        "t2v_sound",
        "text_to_video_with_sound",
        "video_with_sound",
        "audio_visual",
    }:
        return TASK_T2V_SOUND
    
    # Handle image task types directly to avoid circular imports
    if value in {
        "t2i",
        "text_to_image",
        "text2image",
        "image_generation",
        "t2i_gen",
    }:
        return TASK_T2I
    if value in {
        "image_edit",
        "image_editing",
        "i2i",
        "image_to_image",
    }:
        return TASK_IMAGE_EDIT
    
    # Default to text-to-image task
    return TASK_T2I


def is_video_task(task_type: str | None) -> bool:
    return normalize_task_type(task_type) in _VIDEO_TASKS


def is_video_edit_task(task_type: str | None) -> bool:
    return normalize_task_type(task_type) == TASK_VIDEO_EDIT


def get_video_input_strategy() -> VideoInputStrategy:
    strategy = os.environ.get("ARR_VIDEO_INPUT_STRATEGY", VIDEO_INPUT_NATIVE).strip().lower()
    if strategy in {VIDEO_INPUT_NATIVE, VIDEO_INPUT_UNIFORM_FRAMES, "keyframe_plus_clip"}:
        return strategy  # type: ignore[return-value]
    return VIDEO_INPUT_NATIVE


def get_uniform_frame_count() -> int:
    return max(2, int(os.environ.get("ARR_VIDEO_FRAME_COUNT", "8")))


def split_video_paths(data: Mapping[str, Any], task_type: str | None) -> tuple[str | None, list[str]]:
    """Split a sample into optional base video and candidate output videos."""
    task = normalize_task_type(task_type)

    if task in {TASK_T2V, TASK_T2V_MULTISHOT, TASK_T2V_SOUND}:
        outputs = _first_path_from_keys(
            data,
            (
                "video_paths",
                "response",
                "responses",
                "videos",
                "video",
                "generated_video",
                "generated_videos",
            ),
        )
        return None, outputs

    base_paths = _first_path_from_keys(
        data,
        (
            "source_video",
            "base_video",
            "original_video",
            "input_video",
            "reference_video",
            "ref_video",
        ),
    )

    explicit_edited = _first_path_from_keys(
        data,
        (
            "edited_video",
            "edited_videos",
            "target_video",
            "target_videos",
            "output_video",
            "output_videos",
            "result_video",
            "result_videos",
        ),
    )

    if base_paths:
        if explicit_edited:
            return base_paths[0], explicit_edited
        response_paths = _first_path_from_keys(
            data,
            ("response", "responses", "videos", "video_paths", "video"),
        )
        return base_paths[0], response_paths

    if explicit_edited:
        return None, explicit_edited

    response_paths = _first_path_from_keys(
        data,
        ("response", "responses", "videos", "video_paths", "video"),
    )
    if not response_paths:
        return None, []

    label_rank = data.get("label_rank")
    if isinstance(label_rank, Iterable) and not isinstance(label_rank, (str, bytes, Mapping)):
        expected_outputs = len(list(label_rank))
        if len(response_paths) == expected_outputs + 1:
            return response_paths[0], response_paths[1:]
        return None, response_paths

    if "label_score" in data and len(response_paths) == 2:
        return response_paths[0], response_paths[1:]

    if bool(data.get("response_includes_base")) and len(response_paths) >= 2:
        return response_paths[0], response_paths[1:]

    if len(response_paths) >= 3:
        return response_paths[0], response_paths[1:]

    return None, response_paths


def has_base_video(data: Mapping[str, Any], task_type: str | None) -> bool:
    base_path, _ = split_video_paths(data, task_type)
    return base_path is not None


def extract_video_paths(data: Mapping[str, Any], task_type: str | None) -> list[str]:
    """Extract video paths: optional base video, then candidate outputs."""
    base_path, output_paths = split_video_paths(data, task_type)
    if base_path:
        return [base_path, *output_paths]
    return output_paths


def extract_media_paths(data: Mapping[str, Any], task_type: str | None) -> list[str]:
    """Return image or video paths depending on task type."""
    if is_video_task(task_type):
        return extract_video_paths(data, task_type)
    return extract_image_paths(data, task_type)


def count_outputs(
    paths_or_data: list[str] | Mapping[str, Any],
    task_type: str | None,
    has_base: bool | None = None,
) -> int:
    """Count candidate outputs for image or video tasks."""
    if is_video_task(task_type):
        if isinstance(paths_or_data, Mapping):
            _, output_paths = split_video_paths(paths_or_data, task_type)
            return len(output_paths)
        task = normalize_task_type(task_type)
        if (task in {TASK_VIDEO_EDIT, TASK_R2V}) and has_base:
            return max(0, len(paths_or_data) - 1)
        return len(paths_or_data)
    return count_image_outputs(paths_or_data, task_type, has_base=has_base)


def has_base_media(data: Mapping[str, Any], task_type: str | None) -> bool:
    if is_video_task(task_type):
        return has_base_video(data, task_type)
    return has_base_image(data, task_type)


def video_to_base64_url(path: str) -> str:
    """Convert a local video path to a data URL."""
    if path.startswith(("http://", "https://", "data:")):
        return path

    video_path = Path(path).expanduser()
    mime_type, _ = mimetypes.guess_type(str(video_path))
    if not mime_type or not mime_type.startswith("video/"):
        mime_type = "video/mp4"

    with video_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _sample_uniform_frame_paths(video_path: str, num_frames: int) -> list[str]:
    """Extract uniformly sampled frames to temporary PNG files."""
    if isinstance(video_path, str) and video_path.startswith(("http://", "https://", "data:")):
        # decord cannot open remote URLs ("Protocol not found"); the caller is
        # expected to handle URLs via prepare_vlm_video_blocks' native_video path.
        raise ValueError(
            f"Cannot sample frames from a remote URL: {video_path}. "
            "Use ARR_VIDEO_INPUT_STRATEGY=native_video for hosted videos.",
        )
    try:
        import decord  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "uniform_frames strategy requires decord. "
            "Install decord or set ARR_VIDEO_INPUT_STRATEGY=native_video.",
        ) from exc

    reader = decord.VideoReader(str(Path(video_path).expanduser()))
    total = len(reader)
    if total == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    if total <= num_frames:
        indices = list(range(total))
    else:
        step = (total - 1) / max(num_frames - 1, 1)
        indices = [min(total - 1, int(round(i * step))) for i in range(num_frames)]

    frame_paths: list[str] = []
    tmp_dir = tempfile.mkdtemp(prefix="arr_video_frames_")
    stem = Path(video_path).stem
    for idx, frame_idx in enumerate(indices):
        frame = reader[frame_idx].asnumpy()
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("uniform_frames strategy requires Pillow.") from exc

        out_path = Path(tmp_dir) / f"{stem}_frame_{idx:03d}.png"
        Image.fromarray(frame).save(out_path)
        frame_paths.append(str(out_path))
    return frame_paths


def _make_video_block(
    url_or_path: str,
    original_path: str | None = None,
) -> dict[str, Any]:
    """Wrap a video reference into a content part the LLM endpoint accepts.

    Override via ``ARR_VIDEO_BLOCK_TYPE``:

    - ``image_url`` (default, legacy): wraps the url/data-uri as an image_url
      block. Works only if the endpoint dispatches on mime type (some Gemini
      OpenAI-compat layers used to). Tencent venus llmproxy rejects mp4
      bytes here.
    - ``video_url``: legacy behaviour, useful for gateways that really
      implement the ``video_url`` extension.
    - ``gemini_multimodal_url``: A path. Uploads the video to venus via
      preSign+PUT, then emits a Gemini-native multimodal_url block referring
      to the uploaded file. This is what tencent venus llmproxy actually
      accepts for native video input. ``original_path`` should be the local
      file path or remote URL (``upload_video_to_venus`` will download remote
      URLs to a tmp file before uploading).
    """
    block_type = (os.getenv("ARR_VIDEO_BLOCK_TYPE") or "image_url").strip().lower()

    if block_type == "gemini_multimodal_url":
        src = original_path if original_path is not None else url_or_path
        info = upload_video_to_venus(src)  # raises VenusUploadError on failure
        return {
            "type": "gemini_multimodal_url",
            "gemini_multimodal_url": {
                "bucketName": info["bucketName"],
                "filepath": info["filepath"],
                "mimeType": info["mimeType"],
            },
        }

    if block_type == "video_url":
        return {"type": "video_url", "video_url": {"url": url_or_path}}
    return {"type": "image_url", "image_url": {"url": url_or_path}}


def prepare_vlm_video_blocks(
    video_paths: list[str],
    task_type: str | None,
    has_base: bool | None = None,
    strategy: VideoInputStrategy | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible content blocks for video inputs."""
    if not video_paths:
        return []

    strategy = strategy or get_video_input_strategy()
    task = normalize_task_type(task_type)
    blocks: list[dict[str, Any]] = []

    if has_base is None:
        has_base = task in {TASK_VIDEO_EDIT, TASK_R2V} and len(video_paths) >= 2

    if strategy == VIDEO_INPUT_UNIFORM_FRAMES:
        blocks.append(
            {
                "type": "text",
                "text": (
                    "The following are uniformly sampled frames from each video clip. "
                    "Judge temporal quality using motion continuity across frames."
                ),
            },
        )

        def _append_frames_labeled(label: str, path: str) -> None:
            blocks.append({"type": "text", "text": label})
            for frame_path in _sample_uniform_frame_paths(path, get_uniform_frame_count()):
                blocks.append(
                    {"type": "image_url", "image_url": {"url": image_to_base64_url(frame_path)}},
                )

        if task in {TASK_VIDEO_EDIT, TASK_R2V} and has_base and len(video_paths) > 1:
            base_label = (
                "Ground-Truth Video (reference) — sampled frames:"
                if task == TASK_R2V
                else "Video BASE (original clip) — sampled frames:"
            )
            _append_frames_labeled(base_label, video_paths[0])
            candidate_paths = video_paths[1:]
            generated_label = (
                "Generated Video" if task == TASK_R2V else "Edited Video"
            )
            if len(candidate_paths) == 1:
                _append_frames_labeled(f"{generated_label} — sampled frames:", candidate_paths[0])
            else:
                for idx, path in enumerate(candidate_paths, start=1):
                    _append_frames_labeled(f"{generated_label} {idx} — sampled frames:", path)
        else:
            for idx, path in enumerate(video_paths, start=1):
                label = "Generated Video:" if len(video_paths) == 1 else f"Video {idx}:"
                _append_frames_labeled(f"{label} sampled frames", path)
        return blocks

    block_type = (os.getenv("ARR_VIDEO_BLOCK_TYPE") or "image_url").strip().lower()
    use_venus_upload = block_type == "gemini_multimodal_url"

    # When using venus upload we pass the original path/URL straight through;
    # _make_video_block will handle the upload. Otherwise we pre-encode each
    # clip as a data URL or pass a remote URL through unchanged.
    if use_venus_upload:
        video_refs = list(video_paths)
    else:
        video_refs = [video_to_base64_url(path) for path in video_paths]

    def _vblock(idx: int) -> dict[str, Any]:
        # original_path is the raw entry from `video_paths`; ref is what's
        # actually inlined into the message body for non-venus paths.
        return _make_video_block(video_refs[idx], original_path=video_paths[idx])

    if task in {TASK_VIDEO_EDIT, TASK_R2V} and has_base and len(video_refs) > 1:
        base_label = (
            "Ground-Truth Video (reference, EvalVerse R2V):"
            if task == TASK_R2V
            else "Video BASE (original clip):"
        )
        candidate_label_prefix = "Generated Video" if task == TASK_R2V else "Edited Video"
        blocks.extend(
            [
                {"type": "text", "text": base_label},
                _vblock(0),
            ],
        )
        candidate_count = len(video_refs) - 1
        if candidate_count == 1:
            blocks.append({"type": "text", "text": f"{candidate_label_prefix}:"})
            blocks.append(_vblock(1))
        else:
            for offset in range(candidate_count):
                blocks.append(
                    {"type": "text", "text": f"{candidate_label_prefix} {offset + 1}:"},
                )
                blocks.append(_vblock(offset + 1))
    elif task == TASK_VIDEO_EDIT:
        for idx in range(len(video_refs)):
            label = "Edited Video:" if len(video_refs) == 1 else f"Edited Video {idx + 1}:"
            blocks.append({"type": "text", "text": label})
            blocks.append(_vblock(idx))
    else:
        for idx in range(len(video_refs)):
            label = "Generated Video:" if len(video_refs) == 1 else f"Video {idx + 1}:"
            blocks.append({"type": "text", "text": label})
            blocks.append(_vblock(idx))
    return blocks


def add_videos_to_messages(
    messages: list[dict[str, Any]],
    video_paths: list[str],
    task_type: str | None,
    has_base: bool | None = None,
    strategy: VideoInputStrategy | None = None,
    reference_images: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Attach videos and optional reference images to the final user message.

    `reference_images` keys recognized:
        - subject_image / subject_reference
        - object_image
        - scene_image
        - depth_sequence (handled as image if single frame, else ignored here)
    """
    if not video_paths and not reference_images:
        return messages

    content_text = messages[-1]["content"]
    blocks: list[dict[str, Any]] = []

    if reference_images:
        for ref_key, ref_label in (
            ("subject_image", "Reference Subject Image:"),
            ("subject_reference", "Reference Subject Image:"),
            ("object_image", "Reference Object Image:"),
            ("scene_image", "Reference Scene Image:"),
        ):
            ref_value = reference_images.get(ref_key)
            if not ref_value:
                continue
            ref_paths = [ref_value] if isinstance(ref_value, str) else list(ref_value)
            for path in ref_paths:
                blocks.append({"type": "text", "text": ref_label})
                blocks.append({"type": "image_url", "image_url": {"url": image_to_base64_url(path)}})

    blocks.extend(prepare_vlm_video_blocks(video_paths, task_type, has_base=has_base, strategy=strategy))
    blocks.append({"type": "text", "text": content_text})
    messages[-1]["content"] = blocks
    return messages


def attach_media_to_messages(
    messages: list[dict[str, Any]],
    media_paths: list[str],
    task_type: str | None,
    has_base: bool | None = None,
    video_strategy: VideoInputStrategy | None = None,
    reference_images: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Attach images or videos to the final user message based on task type."""
    if not media_paths and not reference_images:
        return messages
    if is_video_task(task_type):
        return add_videos_to_messages(
            messages,
            media_paths,
            task_type,
            has_base=has_base,
            strategy=video_strategy,
            reference_images=reference_images,
        )
    return add_images_to_messages(messages, media_paths, task_type, has_base=has_base)


def extract_reference_images(data: Mapping[str, Any]) -> dict[str, str]:
    """Extract reference image fields recognized by EvalVerse R2V family."""
    refs: dict[str, str] = {}
    for key in (
        "subject_image",
        "subject_reference",
        "object_image",
        "scene_image",
    ):
        value = data.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = value[0] if value else None
            if not value:
                continue
        refs[key] = str(value)
    nested = data.get("metadata", {}).get("references") if isinstance(data.get("metadata"), Mapping) else None
    if nested:
        for key in ("subject_image", "object_image", "scene_image"):
            if key in nested and key not in refs:
                value = nested[key]
                if isinstance(value, list):
                    value = value[0] if value else None
                if value:
                    refs[key] = str(value)
    return refs

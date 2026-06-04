"""Placeholders for EvalVerse Professional Operators (E_prof).

EvalVerse (SIGGRAPH Asia 2026) uses specialized operators to extract
deterministic perception evidence E_prof before VLM reasoning. This module
defines a thin interface and stub implementations so the reward pipeline can
attach operator outputs to samples without committing to specific external
dependencies.

Recommended operators (per EvalVerse Sec. 6.1.1):
- DINO / DINOv2: cross-frame identity tracking.
- InsightFace / ArcFace: face identity preservation.
- YOLO11: semantic anchoring of objects.
- SyncNet: audio-visual sync.
- Whisper: speech / emotion recognition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class OperatorEvidence:
    """Container for operator-extracted perception priors."""
    face_identity: dict[str, Any] = field(default_factory=dict)
    object_detection: dict[str, Any] = field(default_factory=dict)
    camera_trajectory: dict[str, Any] = field(default_factory=dict)
    lighting: dict[str, Any] = field(default_factory=dict)
    aesthetic: dict[str, Any] = field(default_factory=dict)
    sentiment: dict[str, Any] = field(default_factory=dict)
    lip_sync: dict[str, Any] = field(default_factory=dict)
    speech_emotion: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_identity": self.face_identity,
            "object_detection": self.object_detection,
            "camera_trajectory": self.camera_trajectory,
            "lighting": self.lighting,
            "aesthetic": self.aesthetic,
            "sentiment": self.sentiment,
            "lip_sync": self.lip_sync,
            "speech_emotion": self.speech_emotion,
            "extra": self.extra,
        }


class BaseOperator:
    """Interface for operator extractors. Override `extract` in subclasses."""

    name: str = "base_operator"

    def extract(self, video_path: str, audio_path: str | None = None) -> Mapping[str, Any]:
        raise NotImplementedError


def attach_operator_evidence(
    sample: dict[str, Any],
    evidence: OperatorEvidence,
) -> dict[str, Any]:
    """Attach E_prof evidence to a sample's metadata for downstream rubric consumption."""
    metadata = dict(sample.get("metadata") or {})
    metadata["operator_evidence"] = evidence.to_dict()
    sample["metadata"] = metadata
    return sample


__all__ = ["OperatorEvidence", "BaseOperator", "attach_operator_evidence"]

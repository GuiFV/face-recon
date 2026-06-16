"""Domain models shared across the pipeline.

These are deliberately framework free (no torch, supervision or roboflow imports) so the
decision logic stays unit testable without the heavy CV stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# COCO-17 keypoint ordering, as produced by standard pose models.
COCO_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class BBox:
    """Axis aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class Pose:
    """A set of named COCO keypoints for a single person."""

    keypoints: dict[str, Keypoint] = field(default_factory=dict)

    def get(self, name: str) -> Keypoint | None:
        return self.keypoints.get(name)

    def visible(self, name: str, min_confidence: float) -> bool:
        kp = self.keypoints.get(name)
        return kp is not None and kp.confidence >= min_confidence

    @classmethod
    def from_coco_list(cls, points: list[Keypoint]) -> Pose:
        """Build a Pose from a COCO ordered list of keypoints."""
        mapping = {
            name: kp for name, kp in zip(COCO_KEYPOINTS, points, strict=False)
        }
        return cls(keypoints=mapping)

    @classmethod
    def from_triples(cls, triples: dict[str, tuple[float, float, float]]) -> Pose:
        """Build a Pose from a name -> (x, y, confidence) mapping. Handy in tests."""
        return cls(keypoints={n: Keypoint(*t) for n, t in triples.items()})


@dataclass
class Detection:
    """A detected person in a single frame."""

    bbox: BBox
    confidence: float
    track_id: int | None = None


@dataclass
class TrackObservation:
    """Everything known about one track in one frame."""

    track_id: int
    frame_index: int
    person: Detection
    pose: Pose | None = None
    face_bbox: BBox | None = None


class ChallengeKind(StrEnum):
    """The active liveness challenges a person can be asked to perform."""

    SMILE = "smile"
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


@dataclass(frozen=True)
class ChallengeSignals:
    """Per-frame signals a face/pose model provides for the liveness challenges.

    All optional: a given frame may not yield every signal. Values are model-normalised
    (roughly 0..1 for smile/eye_open; head_yaw is signed, negative left and positive right,
    ~0 centred). The source of these signals (a face-keypoint model or a landmark library)
    is decided later and plugs in behind this; the challenge logic does not depend on it.
    """

    smile: float | None = None
    eye_open: float | None = None
    head_yaw: float | None = None

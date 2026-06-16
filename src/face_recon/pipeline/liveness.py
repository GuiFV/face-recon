"""Liveness / anti spoofing checks.

All checks are intentionally additive: a real, live person should pass them, and the
cheapest defeats (a printed photo, a phone screen held up to the camera) should fail at
least one. The functions here are pure geometry over pose keypoints and bounding boxes,
so they are fully unit tested without the CV stack.

The geometry gate (geometry_gate_ok) must pass before identity is even checked: a wrong
shaped or wrong placed face is rejected here and never reaches recognition (see
pipeline.orchestrator and pipeline.state_machine). The active liveness challenge (smile,
blink, turn) supersedes the older passive head-motion check kept below.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from face_recon.core.models import BBox, Pose

# Keypoints that define a present, plausible upper body.
TORSO_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)


@dataclass(frozen=True)
class LivenessResult:
    """Aggregate liveness outcome plus the individual signals, for metrics and tuning."""

    passed: bool
    skeleton_completeness: float
    head_shoulder_ok: bool
    face_scale_ok: bool
    head_motion: float
    failed_checks: tuple[str, ...]


def _distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def skeleton_completeness(pose: Pose, min_confidence: float) -> float:
    """Fraction of the torso defining keypoints that are confidently visible."""
    present = sum(1 for name in TORSO_KEYPOINTS if pose.visible(name, min_confidence))
    return present / len(TORSO_KEYPOINTS)


def shoulder_width(pose: Pose, min_confidence: float) -> float | None:
    """Pixel distance between the shoulders, or None if either is not visible."""
    if not pose.visible("left_shoulder", min_confidence):
        return None
    if not pose.visible("right_shoulder", min_confidence):
        return None
    return _distance(pose.get("left_shoulder"), pose.get("right_shoulder"))


def head_size(pose: Pose, min_confidence: float) -> float | None:
    """Estimate head width from the ears, falling back to the eyes. None if neither."""
    if pose.visible("left_ear", min_confidence) and pose.visible("right_ear", min_confidence):
        return _distance(pose.get("left_ear"), pose.get("right_ear"))
    if pose.visible("left_eye", min_confidence) and pose.visible("right_eye", min_confidence):
        # Eye spacing is roughly half the head width, so scale up.
        return 2.0 * _distance(pose.get("left_eye"), pose.get("right_eye"))
    return None


def head_shoulder_proportion_ok(
    pose: Pose,
    min_confidence: float,
    ratio_min: float,
    ratio_max: float,
) -> bool:
    """True if head size sits in a plausible ratio to shoulder width."""
    sw = shoulder_width(pose, min_confidence)
    hs = head_size(pose, min_confidence)
    if not sw or not hs:
        return False
    ratio = hs / sw
    return ratio_min <= ratio <= ratio_max


def face_body_scale_consistent(
    face_bbox: BBox | None,
    pose: Pose,
    min_confidence: float,
    scale_min: float,
    scale_max: float,
) -> bool:
    """True if the face crop width is consistent with the body scale (shoulder width)."""
    if face_bbox is None:
        return False
    sw = shoulder_width(pose, min_confidence)
    if not sw:
        return False
    ratio = face_bbox.width / sw
    return scale_min <= ratio <= scale_max


def face_position_ok(face_bbox: BBox | None, pose: Pose, min_confidence: float) -> bool:
    """True if the face sits where a head should: above the shoulders and within their span.

    Catches the cheap attacks geometry alone should reject: a phone held at chest height (the
    face centre is below the shoulder line) or a face floating off to the side of the body.
    """
    if face_bbox is None:
        return False
    if not pose.visible("left_shoulder", min_confidence):
        return False
    if not pose.visible("right_shoulder", min_confidence):
        return False
    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    fx, fy = face_bbox.center
    left, right = min(ls.x, rs.x), max(ls.x, rs.x)
    span = right - left
    margin = 0.5 * span  # allow the head to sit a little outside the shoulder line
    if not (left - margin <= fx <= right + margin):
        return False
    shoulder_y = (ls.y + rs.y) / 2.0
    # Image y grows downward, so the head (smaller y) must be above the shoulders.
    return fy <= shoulder_y


def geometry_gate_ok(
    pose: Pose,
    face_bbox: BBox | None,
    *,
    min_confidence: float,
    head_shoulder_ratio_min: float,
    head_shoulder_ratio_max: float,
    face_body_scale_min: float,
    face_body_scale_max: float,
    completeness_required: float = 1.0,
) -> bool:
    """The full geometry gate: a plausible body, with the face the right size and in the right
    place. Must pass before identity is even checked, so a wrong-shaped or wrong-placed face
    never reaches the recognition stage.
    """
    return (
        skeleton_completeness(pose, min_confidence) >= completeness_required
        and head_shoulder_proportion_ok(
            pose, min_confidence, head_shoulder_ratio_min, head_shoulder_ratio_max
        )
        and face_body_scale_consistent(
            face_bbox, pose, min_confidence, face_body_scale_min, face_body_scale_max
        )
        and face_position_ok(face_bbox, pose, min_confidence)
    )


def head_motion_independent_of_torso(
    history: Sequence[Pose],
    min_confidence: float,
) -> float:
    """Measure head motion that is independent of the torso, scale invariant.

    For each frame the nose position is taken relative to the shoulder midpoint and
    normalised by the shoulder width. The spread of that relative vector across frames is
    the signal: a live person nods and turns so it varies, while a rigid photo moves head
    and torso together so it stays near zero. Returns the mean distance of the relative
    nose positions from their centroid.
    """
    rel: list[tuple[float, float]] = []
    for pose in history:
        if not pose.visible("nose", min_confidence):
            continue
        if not pose.visible("left_shoulder", min_confidence):
            continue
        if not pose.visible("right_shoulder", min_confidence):
            continue
        ls = pose.get("left_shoulder")
        rs = pose.get("right_shoulder")
        nose = pose.get("nose")
        cx = (ls.x + rs.x) / 2.0
        cy = (ls.y + rs.y) / 2.0
        sw = math.hypot(ls.x - rs.x, ls.y - rs.y)
        if sw == 0.0:
            continue
        rel.append(((nose.x - cx) / sw, (nose.y - cy) / sw))

    if len(rel) < 2:
        return 0.0

    mx = sum(p[0] for p in rel) / len(rel)
    my = sum(p[1] for p in rel) / len(rel)
    spread = sum(math.hypot(p[0] - mx, p[1] - my) for p in rel) / len(rel)
    return spread


def evaluate_liveness(
    pose: Pose,
    face_bbox: BBox | None,
    history: Sequence[Pose],
    *,
    min_confidence: float,
    head_shoulder_ratio_min: float,
    head_shoulder_ratio_max: float,
    face_body_scale_min: float,
    face_body_scale_max: float,
    head_motion_min: float,
    completeness_required: float = 1.0,
) -> LivenessResult:
    """Run every liveness check and aggregate. Passes only if all checks pass."""
    completeness = skeleton_completeness(pose, min_confidence)
    hs_ok = head_shoulder_proportion_ok(
        pose, min_confidence, head_shoulder_ratio_min, head_shoulder_ratio_max
    )
    scale_ok = face_body_scale_consistent(
        face_bbox, pose, min_confidence, face_body_scale_min, face_body_scale_max
    )
    motion = head_motion_independent_of_torso(history, min_confidence)

    failed: list[str] = []
    if completeness < completeness_required:
        failed.append("skeleton_completeness")
    if not hs_ok:
        failed.append("head_shoulder_proportion")
    if not scale_ok:
        failed.append("face_body_scale")
    if motion < head_motion_min:
        failed.append("head_motion")

    return LivenessResult(
        passed=not failed,
        skeleton_completeness=completeness,
        head_shoulder_ok=hs_ok,
        face_scale_ok=scale_ok,
        head_motion=motion,
        failed_checks=tuple(failed),
    )

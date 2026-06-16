"""Pose estimation stage.

A single person-pose model (e.g. yolov8n-pose) run on the full frame returns each detected
person together with their COCO keypoints, so one call covers both "is a person here" and the
skeleton the geometry gate needs. The raw-response -> Pose mapping (`parse_poses`) and picking
the primary subject (`primary_pose`) are pure and unit tested; `poses_from_frame` is the thin
live call into the inference server.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from face_recon.core.models import Keypoint, Pose
from face_recon.services.inference import InferenceClient


def parse_poses(response: dict[str, Any]) -> list[Pose]:
    """Map a Roboflow pose response into a Pose per detected person.

    Each prediction carries a `keypoints` list of {x, y, confidence, class/class_name}. We key
    keypoints by their COCO name; predictions without keypoints are skipped.
    """
    poses: list[Pose] = []
    for pred in response.get("predictions") or []:
        mapping: dict[str, Keypoint] = {}
        for kp in pred.get("keypoints") or []:
            name = kp.get("class_name") or kp.get("class")
            if not name:
                continue
            mapping[name] = Keypoint(
                x=float(kp.get("x", 0.0)),
                y=float(kp.get("y", 0.0)),
                confidence=float(kp.get("confidence", 0.0)),
            )
        if mapping:
            poses.append(Pose(keypoints=mapping))
    return poses


def _keypoint_span_area(pose: Pose) -> float:
    """Area of the bounding box of a pose's keypoints (a proxy for how close the person is)."""
    if not pose.keypoints:
        return 0.0
    xs = [k.x for k in pose.keypoints.values()]
    ys = [k.y for k in pose.keypoints.values()]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def primary_pose(poses: list[Pose]) -> Pose | None:
    """The largest (nearest) person's pose, or None when nobody is detected."""
    return max(poses, key=_keypoint_span_area) if poses else None


def poses_from_frame(frame: np.ndarray, client: InferenceClient, model_id: str) -> list[Pose]:
    """Run the pose model on a full frame and return a Pose per detected person."""
    return parse_poses(client.infer(frame, model_id))

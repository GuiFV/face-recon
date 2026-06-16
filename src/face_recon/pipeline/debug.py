"""Debug overlay: draw the skeleton, face box, and identity label onto a frame.

Used by the decision-service `/debug` endpoints so we can watch live what the pipeline sees
(handy during bring-up and as a demo). cv2 is imported lazily; the orchestrator stashes a
DebugSnapshot each frame and the routes render it on demand, so nothing is drawn unless
someone is watching.
"""

from __future__ import annotations

from dataclasses import dataclass

from face_recon.core.models import BBox, Pose

# COCO-17 skeleton connections, by keypoint name.
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


@dataclass(frozen=True)
class DebugSnapshot:
    """The most recent frame plus what the pipeline made of it, for the debug overlay."""

    frame: object  # BGR numpy array
    pose: Pose | None = None
    face_bbox: BBox | None = None
    label: str = ""


def annotate(frame_bgr, *, pose=None, face_bbox=None, label="", min_conf=0.3):
    """Return a copy of the frame with the skeleton, face box, and label drawn on it."""
    import cv2

    img = frame_bgr.copy()
    if pose is not None:
        for a, b in SKELETON_EDGES:
            ka, kb = pose.get(a), pose.get(b)
            if ka and kb and ka.confidence >= min_conf and kb.confidence >= min_conf:
                cv2.line(img, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)), (0, 255, 0), 2)
        for kp in pose.keypoints.values():
            if kp.confidence >= min_conf:
                cv2.circle(img, (int(kp.x), int(kp.y)), 3, (0, 200, 0), -1)
    if face_bbox is not None:
        p1 = (int(face_bbox.x1), int(face_bbox.y1))
        p2 = (int(face_bbox.x2), int(face_bbox.y2))
        cv2.rectangle(img, p1, p2, (0, 200, 255), 2)
    if label:
        org, font = (8, 26), cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, label, org, font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)  # outline
        cv2.putText(img, label, org, font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def encode_jpeg(frame_bgr) -> bytes:
    """Encode a BGR frame to JPEG bytes (empty bytes on failure)."""
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr)
    return buf.tobytes() if ok else b""


def render_snapshot(snap: DebugSnapshot) -> bytes:
    """Annotate and JPEG-encode a snapshot in one step."""
    return encode_jpeg(
        annotate(snap.frame, pose=snap.pose, face_bbox=snap.face_bbox, label=snap.label)
    )

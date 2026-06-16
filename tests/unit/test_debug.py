from __future__ import annotations

import numpy as np

from face_recon.core.models import BBox, Pose
from face_recon.pipeline.debug import DebugSnapshot, annotate, render_snapshot


def _frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def _pose():
    return Pose.from_triples(
        {
            "nose": (80, 30, 0.9),
            "left_shoulder": (40, 60, 0.9),
            "right_shoulder": (120, 60, 0.9),
        }
    )


def test_annotate_draws_on_a_copy():
    f = _frame()
    out = annotate(f, pose=_pose(), face_bbox=BBox(70, 20, 90, 45), label="armed | known 0.70")
    assert out.shape == f.shape
    assert (out != 0).any()  # something was drawn
    assert (f == 0).all()  # original untouched


def test_annotate_with_no_inputs_is_safe():
    out = annotate(_frame())
    assert out.shape == (120, 160, 3)


def test_low_confidence_keypoints_are_skipped():
    faint = Pose.from_triples({"nose": (80, 30, 0.1), "left_shoulder": (40, 60, 0.1)})
    assert (annotate(_frame(), pose=faint) == 0).all()  # nothing drawn below min_conf


def test_render_snapshot_returns_jpeg_bytes():
    snap = DebugSnapshot(frame=_frame(), pose=_pose(), face_bbox=BBox(70, 20, 90, 45), label="x")
    data = render_snapshot(snap)
    assert data[:2] == b"\xff\xd8"  # JPEG start-of-image marker

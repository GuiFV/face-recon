from __future__ import annotations

from face_recon.core.models import BBox
from face_recon.pipeline.liveness import face_position_ok, geometry_gate_ok

GATE = dict(
    min_confidence=0.3,
    head_shoulder_ratio_min=0.25,
    head_shoulder_ratio_max=0.75,
    face_body_scale_min=0.15,
    face_body_scale_max=0.80,
)


def test_face_above_shoulders_within_span_ok(upper_body_pose):
    assert face_position_ok(BBox(46, 74, 54, 86), upper_body_pose, 0.3) is True


def test_face_below_shoulders_fails(upper_body_pose):
    # A phone held at chest height: the face centre is below the shoulder line.
    assert face_position_ok(BBox(46, 110, 54, 122), upper_body_pose, 0.3) is False


def test_face_off_to_the_side_fails(upper_body_pose):
    assert face_position_ok(BBox(150, 74, 160, 86), upper_body_pose, 0.3) is False


def test_no_face_fails(upper_body_pose):
    assert face_position_ok(None, upper_body_pose, 0.3) is False


def test_geometry_gate_passes_for_plausible_body(upper_body_pose):
    assert geometry_gate_ok(upper_body_pose, BBox(46, 74, 54, 86), **GATE) is True


def test_geometry_gate_fails_on_bad_face_position(upper_body_pose):
    assert geometry_gate_ok(upper_body_pose, BBox(46, 110, 54, 122), **GATE) is False

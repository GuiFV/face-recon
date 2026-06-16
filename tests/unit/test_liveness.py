from __future__ import annotations

from face_recon.core.models import BBox, Pose
from face_recon.pipeline.liveness import (
    evaluate_liveness,
    face_body_scale_consistent,
    head_motion_independent_of_torso,
    head_shoulder_proportion_ok,
    skeleton_completeness,
)

MIN_CONF = 0.3


def _live_history(make_pose) -> list[Pose]:
    """Shoulders fixed, nose nods and turns: head moves independently of the torso."""
    noses = [(50, 80), (53, 77), (47, 83), (51, 79), (49, 81)]
    return [
        make_pose(
            nose=(nx, ny, 0.9),
            left_shoulder=(40, 100, 0.9),
            right_shoulder=(60, 100, 0.9),
        )
        for nx, ny in noses
    ]


def _rigid_history(make_pose) -> list[Pose]:
    """Whole pose translates together, as a printed photo waved in front of the camera."""
    history = []
    for i in range(5):
        dx, dy = 5 * i, 3 * i
        history.append(
            make_pose(
                nose=(50 + dx, 80 + dy, 0.9),
                left_shoulder=(40 + dx, 100 + dy, 0.9),
                right_shoulder=(60 + dx, 100 + dy, 0.9),
            )
        )
    return history


def test_skeleton_completeness(upper_body_pose, make_pose):
    assert skeleton_completeness(upper_body_pose, MIN_CONF) == 1.0
    partial = make_pose(nose=(50, 80, 0.9), left_shoulder=(40, 100, 0.9))
    assert skeleton_completeness(partial, MIN_CONF) == 0.4


def test_head_shoulder_proportion(upper_body_pose):
    # head width 10, shoulder width 20, ratio 0.5: inside [0.25, 0.75].
    assert head_shoulder_proportion_ok(upper_body_pose, MIN_CONF, 0.25, 0.75) is True
    # A tighter band that excludes 0.5 should reject.
    assert head_shoulder_proportion_ok(upper_body_pose, MIN_CONF, 0.6, 0.75) is False


def test_face_body_scale(upper_body_pose):
    # shoulder width 20; face width 8 gives ratio 0.4, inside [0.15, 0.80].
    good_face = BBox(46, 74, 54, 86)
    assert face_body_scale_consistent(good_face, upper_body_pose, MIN_CONF, 0.15, 0.80) is True
    # A face far too wide for the body (ratio 2.0) should reject.
    huge_face = BBox(20, 74, 60, 86)
    assert face_body_scale_consistent(huge_face, upper_body_pose, MIN_CONF, 0.15, 0.80) is False
    # No face crop: reject.
    assert face_body_scale_consistent(None, upper_body_pose, MIN_CONF, 0.15, 0.80) is False


def test_head_motion_live_vs_rigid(make_pose):
    live = head_motion_independent_of_torso(_live_history(make_pose), MIN_CONF)
    rigid = head_motion_independent_of_torso(_rigid_history(make_pose), MIN_CONF)
    assert rigid < 0.001
    assert live > 0.02
    assert live > rigid


def test_evaluate_liveness_passes_for_live_person(upper_body_pose, make_pose):
    face = BBox(46, 74, 54, 86)
    result = evaluate_liveness(
        upper_body_pose,
        face,
        _live_history(make_pose),
        min_confidence=MIN_CONF,
        head_shoulder_ratio_min=0.25,
        head_shoulder_ratio_max=0.75,
        face_body_scale_min=0.15,
        face_body_scale_max=0.80,
        head_motion_min=0.02,
    )
    assert result.passed is True
    assert result.failed_checks == ()


def test_evaluate_liveness_fails_rigid_photo_on_head_motion(upper_body_pose, make_pose):
    face = BBox(46, 74, 54, 86)
    result = evaluate_liveness(
        upper_body_pose,
        face,
        _rigid_history(make_pose),
        min_confidence=MIN_CONF,
        head_shoulder_ratio_min=0.25,
        head_shoulder_ratio_max=0.75,
        face_body_scale_min=0.15,
        face_body_scale_max=0.80,
        head_motion_min=0.02,
    )
    assert result.passed is False
    assert "head_motion" in result.failed_checks

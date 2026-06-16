from __future__ import annotations

from face_recon.pipeline.pose import parse_poses, primary_pose


def _person(cx, cy, spread, conf=0.9):
    """A pose prediction with a nose and two shoulders spread around (cx, cy)."""
    return {
        "keypoints": [
            {"x": cx, "y": cy - spread, "confidence": conf, "class_name": "nose"},
            {"x": cx - spread, "y": cy, "confidence": conf, "class_name": "left_shoulder"},
            {"x": cx + spread, "y": cy, "confidence": conf, "class_name": "right_shoulder"},
        ]
    }


def test_parse_poses_maps_named_keypoints():
    poses = parse_poses({"predictions": [_person(100, 100, 20)]})
    assert len(poses) == 1
    nose = poses[0].get("nose")
    assert nose is not None and (nose.x, nose.y, nose.confidence) == (100, 80, 0.9)
    assert poses[0].get("left_shoulder").x == 80


def test_parse_poses_falls_back_to_class_field():
    resp = {"predictions": [{"keypoints": [{"x": 1, "y": 2, "confidence": 0.5, "class": "nose"}]}]}
    assert parse_poses(resp)[0].get("nose").confidence == 0.5


def test_parse_poses_skips_predictions_without_keypoints():
    assert parse_poses({"predictions": [{"keypoints": []}, {}]}) == []


def test_parse_poses_handles_empty_response():
    assert parse_poses({}) == []


def test_primary_pose_picks_the_largest_person():
    far = _person(50, 50, 5)
    near = _person(200, 200, 40)
    poses = parse_poses({"predictions": [far, near]})
    primary = primary_pose(poses)
    # the near person's shoulders are 80px apart, the far person's 10px
    assert primary.get("left_shoulder").x == 160


def test_primary_pose_none_when_empty():
    assert primary_pose([]) is None

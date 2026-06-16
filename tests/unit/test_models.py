from __future__ import annotations

from face_recon.core.models import COCO_KEYPOINTS, BBox, Keypoint, Pose


def test_bbox_geometry():
    box = BBox(10, 20, 30, 60)
    assert box.width == 20
    assert box.height == 40
    assert box.area == 800
    assert box.center == (20.0, 40.0)


def test_pose_from_triples_and_visibility():
    pose = Pose.from_triples({"nose": (1.0, 2.0, 0.8), "left_eye": (0.0, 0.0, 0.1)})
    assert pose.get("nose") == Keypoint(1.0, 2.0, 0.8)
    assert pose.visible("nose", 0.5) is True
    assert pose.visible("left_eye", 0.5) is False
    assert pose.visible("right_eye", 0.5) is False


def test_pose_from_coco_list_maps_in_order():
    points = [Keypoint(float(i), float(i), 0.9) for i in range(len(COCO_KEYPOINTS))]
    pose = Pose.from_coco_list(points)
    assert pose.get("nose") == Keypoint(0.0, 0.0, 0.9)
    assert pose.get("left_shoulder") == Keypoint(5.0, 5.0, 0.9)
    assert pose.get("right_ankle") == Keypoint(16.0, 16.0, 0.9)

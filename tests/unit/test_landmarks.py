from __future__ import annotations

from face_recon.services.landmarks import signals_from_blendshapes


def test_smile_maps_high_with_eyes_open():
    s = signals_from_blendshapes(0.8, 0.7, 0.0, 0.0, 0.0)
    assert s.smile > 0.5
    assert s.eye_open > 0.9


def test_blink_maps_to_low_eye_open():
    s = signals_from_blendshapes(0.0, 0.0, 0.9, 0.85, 0.0)
    assert s.eye_open < 0.2


def test_head_yaw_passes_through_signed():
    assert signals_from_blendshapes(0, 0, 0, 0, -0.5).head_yaw == -0.5
    assert signals_from_blendshapes(0, 0, 0, 0, 0.4).head_yaw == 0.4

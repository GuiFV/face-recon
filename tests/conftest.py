"""Shared test fixtures."""

from __future__ import annotations

import pytest

from face_recon.core.models import Pose


@pytest.fixture
def make_pose():
    """Factory: make_pose(nose=(x, y, conf), left_shoulder=(...), ...) -> Pose."""

    def _make(**triples: tuple[float, float, float]) -> Pose:
        return Pose.from_triples(triples)

    return _make


@pytest.fixture
def upper_body_pose(make_pose) -> Pose:
    """A complete, plausibly proportioned upper body pose.

    Shoulder width 20, head width (ears) 10 so the head to shoulder ratio is 0.5.
    """
    return make_pose(
        nose=(50, 80, 0.9),
        left_eye=(47, 78, 0.9),
        right_eye=(53, 78, 0.9),
        left_ear=(45, 80, 0.9),
        right_ear=(55, 80, 0.9),
        left_shoulder=(40, 100, 0.9),
        right_shoulder=(60, 100, 0.9),
        left_hip=(43, 140, 0.9),
        right_hip=(57, 140, 0.9),
    )

"""MediaPipe face-landmark provider: frames -> ChallengeSignals (smile, eye_open, head_yaw).

MediaPipe Face Landmarker outputs blendshape scores (mouthSmile_left/right, eyeBlink_left/
right) and a facial transformation matrix from which head yaw is derived. The PURE mapping
from those raw values to ChallengeSignals is unit tested here; the MediaPipe call itself is
lazy-imported and runs only on the GPU host with real frames.

Convention: eyeBlink blendshapes are high when the eye is CLOSED, so eye_open = 1 - blink.
head_yaw is signed (negative = head turned left, positive = right, ~0 centred), matching the
challenge detectors.
"""

from __future__ import annotations

from face_recon.core.models import ChallengeSignals


def signals_from_blendshapes(
    smile_left: float,
    smile_right: float,
    blink_left: float,
    blink_right: float,
    head_yaw: float,
) -> ChallengeSignals:
    """Map raw MediaPipe blendshape/pose values to the per-frame challenge signals."""
    smile = (smile_left + smile_right) / 2.0
    eye_open = 1.0 - max(blink_left, blink_right)
    return ChallengeSignals(smile=smile, eye_open=eye_open, head_yaw=head_yaw)


class LandmarkProvider:
    """Wraps a MediaPipe Face Landmarker. Lazy so importing this module is cheap."""

    def __init__(self, model_asset_path: str | None = None) -> None:
        self.model_asset_path = model_asset_path
        self._landmarker = None

    def _ensure(self):
        if self._landmarker is None:
            # Lazy import: mediapipe is only needed on the GPU host at run time.
            import mediapipe as mp

            base = mp.tasks.BaseOptions(model_asset_path=self.model_asset_path)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        return self._landmarker

    def signals(self, frame) -> ChallengeSignals | None:
        """Run MediaPipe on a BGR frame and return per-frame ChallengeSignals (None if no face).

        Reads the named blendshapes (mouthSmileLeft/Right, eyeBlinkLeft/Right) and derives head
        yaw from the facial transformation matrix, then maps them with signals_from_blendshapes.
        The yaw scale/sign is a live calibration point: the turn thresholds in settings are tuned
        against these values at runtime.
        """
        import mediapipe as mp
        import numpy as np

        rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._ensure().detect(image)
        if not result.face_blendshapes:
            return None

        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
        head_yaw = 0.0
        if result.facial_transformation_matrixes:
            m = np.asarray(result.facial_transformation_matrixes[0]).reshape(4, 4)
            head_yaw = float(np.arctan2(m[0, 2], m[2, 2]))  # rotation about the vertical axis

        return signals_from_blendshapes(
            smile_left=scores.get("mouthSmileLeft", 0.0),
            smile_right=scores.get("mouthSmileRight", 0.0),
            blink_left=scores.get("eyeBlinkLeft", 0.0),
            blink_right=scores.get("eyeBlinkRight", 0.0),
            head_yaw=head_yaw,
        )

"""Face detection + ArcFace identity embedding via insightface.

Used by both enrolment (build the reference set) and the live pipeline (the "is this the
enrolled person" step). insightface's buffalo_l pack bundles face detection (SCRFD), 5-point
alignment, and an ArcFace recognition model, giving a 512-dim L2-normalised embedding with
strong identity separability, which CLIP lacked (CLIP is semantic, not face-specific). It runs
on onnxruntime: CPU for enrolment, GPU at runtime. The import is lazy so this module stays cheap
to import and the rest of the package does not depend on insightface being installed.

ArcFace embeddings are normalised, so cosine similarity is just their dot product. Identity
quality depends on face size: a face smaller than ~80 px gives a noisy embedding, so enrolment
keeps only large, confident faces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    embedding: np.ndarray  # 512-dim, L2-normalised (ArcFace)
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    det_score: float

    @property
    def min_side(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return min(x2 - x1, y2 - y1)


class FaceEmbedder:
    """Wraps insightface FaceAnalysis. Lazy: the model loads on first use."""

    def __init__(self, det_size: tuple[int, int] = (640, 640), use_gpu: bool = False) -> None:
        self.det_size = det_size
        self.use_gpu = use_gpu
        self._app = None

    def _ensure(self):
        if self._app is None:
            from insightface.app import FaceAnalysis

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.use_gpu
                else ["CPUExecutionProvider"]
            )
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0 if self.use_gpu else -1, det_size=self.det_size)
            self._app = app
        return self._app

    def faces(self, frame_bgr) -> list[DetectedFace]:
        """All faces in a BGR frame, each with its ArcFace embedding."""
        out: list[DetectedFace] = []
        for f in self._ensure().get(frame_bgr):
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            out.append(
                DetectedFace(
                    embedding=np.asarray(f.normed_embedding, dtype=float),
                    bbox=(x1, y1, x2, y2),
                    det_score=float(f.det_score),
                )
            )
        return out

    def best_face(self, frame_bgr) -> DetectedFace | None:
        """The largest face in the frame, or None if no face is detected."""
        faces = self.faces(frame_bgr)
        if not faces:
            return None
        return max(faces, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))

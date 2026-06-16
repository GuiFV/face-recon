"""Face matching against the enrolled reference set.

Pure numeric logic: given an embedding and a set of reference embeddings, decide whether
it matches the enrolled person. No CV stack imports, so this is fully unit tested.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceMatch:
    """Outcome of matching one embedding against the reference set."""

    similarity: float
    is_match: bool
    reference_index: int | None


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors. Returns 0.0 if either has zero norm."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def match_embedding(
    embedding: Sequence[float],
    references: Sequence[Sequence[float]],
    threshold: float,
    top_k: int = 1,
) -> FaceMatch:
    """Match `embedding` against the enrolled `references` by cosine similarity.

    The decision score is the mean of the `top_k` highest similarities, which is steadier
    than a single best match (one stray reference cannot wave a stranger through). With the
    default `top_k=1` this is just the best similarity. The best reference index is always
    reported for logging; `is_match` carries the threshold decision on the score.
    """
    if not references:
        return FaceMatch(similarity=0.0, is_match=False, reference_index=None)
    sims = [cosine_similarity(embedding, ref) for ref in references]
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    best_index = order[0]
    k = max(1, min(top_k, len(sims)))
    score = sum(sims[i] for i in order[:k]) / k
    return FaceMatch(
        similarity=score,
        is_match=score >= threshold,
        reference_index=best_index,
    )

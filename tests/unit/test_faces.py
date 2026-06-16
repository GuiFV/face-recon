from __future__ import annotations

import math

from face_recon.pipeline.faces import cosine_similarity, match_embedding


def test_cosine_similarity_bounds():
    assert math.isclose(cosine_similarity([1, 0, 0], [1, 0, 0]), 1.0)
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)
    assert math.isclose(cosine_similarity([1, 2, 3], [2, 4, 6]), 1.0)


def test_cosine_similarity_zero_vector_is_zero():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_match_picks_best_reference_above_threshold():
    embedding = [1.0, 0.0]
    references = [[0.0, 1.0], [0.9, 0.1], [1.0, 0.0]]
    result = match_embedding(embedding, references, threshold=0.8)
    assert result.is_match is True
    assert result.reference_index == 2
    assert math.isclose(result.similarity, 1.0)


def test_match_below_threshold_is_not_a_match():
    embedding = [1.0, 0.0]
    references = [[0.0, 1.0], [0.1, 0.9]]
    result = match_embedding(embedding, references, threshold=0.8)
    assert result.is_match is False


def test_match_with_no_references():
    result = match_embedding([1.0, 0.0], [], threshold=0.5)
    assert result.is_match is False
    assert result.reference_index is None
    assert result.similarity == 0.0


def test_top_k_mean_resists_a_single_lucky_reference():
    # One reference matches perfectly; the rest are orthogonal. top_k=1 lets it through,
    # but averaging the top 3 pulls the score below threshold (a stranger should not pass on
    # the strength of one coincidental match).
    embedding = [1.0, 0.0]
    references = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    assert match_embedding(embedding, references, threshold=0.5, top_k=1).is_match is True
    averaged = match_embedding(embedding, references, threshold=0.5, top_k=3)
    assert averaged.is_match is False
    assert averaged.reference_index == 0  # best match still reported
    assert math.isclose(averaged.similarity, 1.0 / 3.0)

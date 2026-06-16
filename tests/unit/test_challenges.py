from __future__ import annotations

from random import Random

from face_recon.core.models import ChallengeKind, ChallengeSignals
from face_recon.pipeline.challenges import (
    CHALLENGE_PROMPT,
    ChallengeConfig,
    choose_challenge,
    detect_blink,
    detect_smile,
    detect_turn,
    evaluate_challenge,
)

CFG = ChallengeConfig(
    smile_neutral_below=0.2,
    smile_above=0.5,
    blink_open_above=0.2,
    blink_closed_below=0.1,
    turn_centre_band=0.15,
    turn_threshold=0.35,
)


def _smiles(*vals):
    return [ChallengeSignals(smile=v) for v in vals]


def _eyes(*vals):
    return [ChallengeSignals(eye_open=v) for v in vals]


def _yaws(*vals):
    return [ChallengeSignals(head_yaw=v) for v in vals]


# --- selection ---


def test_choose_challenge_is_deterministic_with_seed():
    enabled = list(ChallengeKind)
    a = choose_challenge(Random(42), enabled)
    b = choose_challenge(Random(42), enabled)
    assert a == b
    assert a in enabled


def test_every_challenge_has_a_prompt():
    for kind in ChallengeKind:
        assert kind in CHALLENGE_PROMPT


# --- smile (the anti-spoof property) ---


def test_smile_neutral_then_smile_passes():
    assert detect_smile(_smiles(0.1, 0.15, 0.6, 0.7), 0.2, 0.5) is True


def test_static_printed_smile_fails_no_neutral_first():
    # A photo shows a smile from the first frame: never a neutral->smile transition.
    assert detect_smile(_smiles(0.8, 0.85, 0.9), 0.2, 0.5) is False


def test_smile_only_neutral_fails():
    assert detect_smile(_smiles(0.05, 0.1, 0.15), 0.2, 0.5) is False


# --- blink ---


def test_blink_open_then_closed_passes():
    assert detect_blink(_eyes(0.3, 0.28, 0.05), 0.2, 0.1) is True


def test_blink_always_open_fails():
    assert detect_blink(_eyes(0.3, 0.31, 0.29), 0.2, 0.1) is False


def test_blink_closed_from_start_fails():
    assert detect_blink(_eyes(0.05, 0.04, 0.03), 0.2, 0.1) is False


# --- turn ---


def test_turn_left_centre_then_left_passes():
    assert detect_turn(_yaws(0.0, -0.1, -0.5), "left", 0.15, 0.35) is True


def test_turn_left_wrong_direction_fails():
    assert detect_turn(_yaws(0.0, 0.1, 0.5), "left", 0.15, 0.35) is False


def test_turn_without_centre_first_fails():
    assert detect_turn(_yaws(-0.5, -0.6), "left", 0.15, 0.35) is False


# --- dispatch ---


def test_evaluate_challenge_dispatches():
    assert evaluate_challenge(ChallengeKind.SMILE, _smiles(0.1, 0.6), CFG) is True
    assert evaluate_challenge(ChallengeKind.BLINK, _eyes(0.3, 0.05), CFG) is True
    assert evaluate_challenge(ChallengeKind.TURN_RIGHT, _yaws(0.0, 0.5), CFG) is True
    assert evaluate_challenge(ChallengeKind.TURN_LEFT, _yaws(0.0, 0.5), CFG) is False

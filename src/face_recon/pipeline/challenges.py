"""Active liveness challenges.

After a track passes the geometry gate and the identity match, the system asks the person
to perform one live action, chosen at random per arming so it cannot be pre-recorded:

- smile: a neutral mouth that then becomes a smile,
- blink: open eyes that then close,
- turn the head left or right: a centred head that then turns past a threshold.

Each detector inspects a time-ordered series of per-frame ChallengeSignals and decides
whether the required DYNAMIC transition happened. The key anti-spoof property is that a
static image fails: a printed smile shows a smile from the very first frame with no neutral
before it, so detect_smile never fires. The source of the signals (a face-keypoint model or
a landmark library) is decided later and plugs in behind this; the logic here is
source-agnostic and unit tested.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

from face_recon.core.models import ChallengeKind, ChallengeSignals

# Spoken-prompt key per challenge; the audio handler maps this to a WAV file.
CHALLENGE_PROMPT: dict[ChallengeKind, str] = {
    ChallengeKind.SMILE: "challenge_smile",
    ChallengeKind.BLINK: "challenge_blink",
    ChallengeKind.TURN_LEFT: "challenge_turn_left",
    ChallengeKind.TURN_RIGHT: "challenge_turn_right",
}


@dataclass(frozen=True)
class ChallengeConfig:
    """Thresholds for the challenge detectors (built from Settings at runtime)."""

    smile_neutral_below: float
    smile_above: float
    blink_open_above: float
    blink_closed_below: float
    turn_centre_band: float
    turn_threshold: float


def choose_challenge(rng: Random, enabled: Sequence[ChallengeKind]) -> ChallengeKind:
    """Pick one challenge at random from the enabled set.

    The RNG is injected so tests are deterministic; at runtime it is seeded unpredictably so
    an attacker cannot know which challenge will be asked.
    """
    if not enabled:
        raise ValueError("no challenges enabled")
    return enabled[rng.randrange(len(enabled))]


def detect_smile(
    signals: Sequence[ChallengeSignals], neutral_below: float, smile_above: float
) -> bool:
    """True if a neutral mouth was seen, then a smile (a genuine transition)."""
    seen_neutral = False
    for s in signals:
        if s.smile is None:
            continue
        if s.smile <= neutral_below:
            seen_neutral = True
        elif seen_neutral and s.smile >= smile_above:
            return True
    return False


def detect_blink(
    signals: Sequence[ChallengeSignals], open_above: float, closed_below: float
) -> bool:
    """True if the eyes were open, then closed (a blink dip)."""
    seen_open = False
    for s in signals:
        if s.eye_open is None:
            continue
        if s.eye_open >= open_above:
            seen_open = True
        elif seen_open and s.eye_open <= closed_below:
            return True
    return False


def detect_turn(
    signals: Sequence[ChallengeSignals], direction: str, centre_band: float, threshold: float
) -> bool:
    """True if the head was centred, then turned past `threshold` in `direction`.

    direction is "left" (yaw goes negative) or "right" (yaw goes positive).
    """
    seen_centre = False
    for s in signals:
        if s.head_yaw is None:
            continue
        if abs(s.head_yaw) <= centre_band:
            seen_centre = True
        elif seen_centre:
            if direction == "left" and s.head_yaw <= -threshold:
                return True
            if direction == "right" and s.head_yaw >= threshold:
                return True
    return False


def evaluate_challenge(
    kind: ChallengeKind, signals: Sequence[ChallengeSignals], cfg: ChallengeConfig
) -> bool:
    """Did the person complete the given challenge over this window of signals?"""
    if kind is ChallengeKind.SMILE:
        return detect_smile(signals, cfg.smile_neutral_below, cfg.smile_above)
    if kind is ChallengeKind.BLINK:
        return detect_blink(signals, cfg.blink_open_above, cfg.blink_closed_below)
    if kind is ChallengeKind.TURN_LEFT:
        return detect_turn(signals, "left", cfg.turn_centre_band, cfg.turn_threshold)
    if kind is ChallengeKind.TURN_RIGHT:
        return detect_turn(signals, "right", cfg.turn_centre_band, cfg.turn_threshold)
    return False

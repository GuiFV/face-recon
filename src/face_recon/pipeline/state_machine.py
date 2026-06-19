"""The face-recon alarm state machine.

Pure logic with an injected clock (every input carries `now`) and an injected RNG for the
challenge choice, so it is fully unit testable with no real time or randomness. The
orchestrator computes a Perception each tick (from the camera CV, an optional external PIR,
and the challenge signals) and feeds it here; the returned Events are emitted as webhooks for
an integration to act on.

Flow:
  IDLE        quiet (no PIR) for idle_quiet_s            -> ARMING     (emit ARMING)
  ARMING      motion during the countdown               -> IDLE       (emit ARMING_CANCELLED)
              countdown elapses                          -> ARMED
  ARMED       PIR and camera person (correlated motion)  -> EVALUATING (emit MOTION)
  EVALUATING  K consecutive frames geometry_ok+identity  -> CHALLENGE  (emit CHALLENGE_ISSUED)
              window elapses without confirming identity -> ALARM      (emit ALARM)
  CHALLENGE   challenge passed                           -> IDLE       (emit GREETED, DISARMED)
              window elapses, retries remain             -> re-issue   (emit CHALLENGE_ISSUED)
              window elapses, no retries left            -> ALARM      (emit ALARM)
  ALARM       scene quiet (no PIR, no person) for grace  -> IDLE       (emit DISARMED)

Geometry gate before identity: in EVALUATING a frame only counts toward the streak when both
geometry_ok and identity_match are true, so a failed geometry check (wrong face size or
position) never reaches the identity stage, and an unrecognised person simply never
accumulates the streak and times out into ALARM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from random import Random

from face_recon.core.models import ChallengeKind
from face_recon.pipeline.challenges import choose_challenge


class AlarmState(StrEnum):
    IDLE = "idle"
    ARMING = "arming"
    ARMED = "armed"
    EVALUATING = "evaluating"
    CHALLENGE = "challenge"
    ALARM = "alarm"


class EventKind(StrEnum):
    ARMING = "arming"
    ARMING_CANCELLED = "arming_cancelled"
    MOTION = "motion"
    CHALLENGE_ISSUED = "challenge_issued"
    GREETED = "greeted"
    ALARM = "alarm"
    DISARMED = "disarmed"


@dataclass(frozen=True)
class Event:
    kind: EventKind
    challenge: ChallengeKind | None = None


@dataclass(frozen=True)
class Perception:
    """One tick of evidence fed to the state machine by the orchestrator."""

    now: float
    pir: bool = False  # PIR considered active (orchestrator applies the correlation window)
    person: bool = False  # the camera sees a person
    geometry_ok: bool | None = None  # set during EVALUATING
    identity_match: bool | None = None  # set during EVALUATING
    challenge_passed: bool | None = None  # set during CHALLENGE
    armed_switch: bool | None = None  # physical switch: False disarms, True arms, None=no signal


@dataclass(frozen=True)
class StateMachineConfig:
    idle_quiet_s: float
    countdown_s: float
    eval_window_s: float
    challenge_window_s: float
    challenge_retries: int
    alarm_grace_s: float
    k_consecutive: int
    enabled_challenges: tuple[ChallengeKind, ...]

    @classmethod
    def from_settings(cls, settings) -> StateMachineConfig:
        kinds = tuple(
            ChallengeKind(c) for c in settings.enabled_challenge_list if c in set(ChallengeKind)
        )
        return cls(
            idle_quiet_s=settings.idle_quiet_seconds,
            countdown_s=settings.arming_countdown_seconds,
            eval_window_s=settings.eval_window_seconds,
            challenge_window_s=settings.challenge_window_seconds,
            challenge_retries=settings.challenge_retries,
            alarm_grace_s=settings.alarm_grace_seconds,
            k_consecutive=settings.k_consecutive,
            enabled_challenges=kinds or tuple(ChallengeKind),
        )


@dataclass
class StateMachine:
    cfg: StateMachineConfig
    rng: Random
    state: AlarmState = AlarmState.IDLE
    # internal timers / counters
    _last_pir: float | None = field(default=None, repr=False)
    _deadline: float = field(default=0.0, repr=False)  # countdown / eval / challenge end
    _good_streak: int = field(default=0, repr=False)
    _challenge: ChallengeKind | None = field(default=None, repr=False)
    _retries_left: int = field(default=0, repr=False)
    _alarm_quiet_since: float | None = field(default=None, repr=False)

    @property
    def current_challenge(self) -> ChallengeKind | None:
        return self._challenge

    def reset(self, now: float) -> None:
        """Return to IDLE and hold the quiet timer, e.g. after a camera-stream reconnect: any
        in-flight evaluation, challenge, or alarm is dropped rather than carried as stale state
        across the gap, and the machine re-arms cleanly once the scene is quiet again."""
        self.state = AlarmState.IDLE
        self._last_pir = now
        self._deadline = 0.0
        self._good_streak = 0
        self._challenge = None
        self._retries_left = 0
        self._alarm_quiet_since = None

    def update(self, p: Perception) -> list[Event]:
        if self._last_pir is None:
            self._last_pir = p.now
        if p.pir:
            self._last_pir = p.now

        # An external arm/disarm override (e.g. a physical switch, pushed via the control API)
        # is the master control. While it says disarmed, the machine stands down and will not
        # auto-arm; a None reading means "no signal", so the autonomous flow runs unchanged.
        if p.armed_switch is False:
            return self._force_disarm(p)

        handler = {
            AlarmState.IDLE: self._idle,
            AlarmState.ARMING: self._arming,
            AlarmState.ARMED: self._armed,
            AlarmState.EVALUATING: self._evaluating,
            AlarmState.CHALLENGE: self._challenge_state,
            AlarmState.ALARM: self._alarm,
        }[self.state]
        return handler(p)

    # --- states ---

    def _idle(self, p: Perception) -> list[Event]:
        # _last_pir is guaranteed set by update() before any handler runs.
        if p.now - self._last_pir >= self.cfg.idle_quiet_s:
            self.state = AlarmState.ARMING
            self._deadline = p.now + self.cfg.countdown_s
            return [Event(EventKind.ARMING)]
        return []

    def _arming(self, p: Perception) -> list[Event]:
        if p.pir or p.person:
            self.state = AlarmState.IDLE
            self._last_pir = p.now
            return [Event(EventKind.ARMING_CANCELLED)]
        if p.now >= self._deadline:
            self.state = AlarmState.ARMED
        return []

    def _armed(self, p: Perception) -> list[Event]:
        if p.pir and p.person:
            self.state = AlarmState.EVALUATING
            self._deadline = p.now + self.cfg.eval_window_s
            self._good_streak = 0
            return [Event(EventKind.MOTION)]
        return []

    def _evaluating(self, p: Perception) -> list[Event]:
        if p.geometry_ok and p.identity_match:
            self._good_streak += 1
        else:
            self._good_streak = 0
        if self._good_streak >= self.cfg.k_consecutive:
            return self._issue_challenge(p, self.cfg.challenge_retries)
        if p.now >= self._deadline:
            return self._enter_alarm(p)
        return []

    def _challenge_state(self, p: Perception) -> list[Event]:
        if p.challenge_passed:
            self.state = AlarmState.IDLE
            self._last_pir = p.now
            self._challenge = None
            return [Event(EventKind.GREETED), Event(EventKind.DISARMED)]
        if p.now >= self._deadline:
            if self._retries_left > 0:
                return self._issue_challenge(p, self._retries_left - 1)
            return self._enter_alarm(p)
        return []

    def _alarm(self, p: Perception) -> list[Event]:
        if p.pir or p.person:
            self._alarm_quiet_since = None
            return []
        if self._alarm_quiet_since is None:
            self._alarm_quiet_since = p.now
        if p.now - self._alarm_quiet_since >= self.cfg.alarm_grace_s:
            self.state = AlarmState.IDLE
            self._last_pir = p.now
            return [Event(EventKind.DISARMED)]
        return []

    # --- helpers ---

    def _force_disarm(self, p: Perception) -> list[Event]:
        """Master disarm from the physical switch: reset to IDLE and hold (no auto-arm)."""
        self._last_pir = p.now  # hold the quiet timer so re-enabling does not arm instantly
        self._good_streak = 0
        self._challenge = None
        self._alarm_quiet_since = None
        if self.state is AlarmState.IDLE:
            return []
        self.state = AlarmState.IDLE
        return [Event(EventKind.DISARMED)]

    def _issue_challenge(self, p: Perception, retries_left: int) -> list[Event]:
        self.state = AlarmState.CHALLENGE
        self._challenge = choose_challenge(self.rng, self.cfg.enabled_challenges)
        self._deadline = p.now + self.cfg.challenge_window_s
        self._retries_left = retries_left
        return [Event(EventKind.CHALLENGE_ISSUED, challenge=self._challenge)]

    def _enter_alarm(self, p: Perception) -> list[Event]:
        self.state = AlarmState.ALARM
        self._challenge = None
        self._alarm_quiet_since = None
        return [Event(EventKind.ALARM)]

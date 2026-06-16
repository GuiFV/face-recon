from __future__ import annotations

from random import Random

from face_recon.core.models import ChallengeKind
from face_recon.pipeline.state_machine import (
    AlarmState,
    EventKind,
    Perception,
    StateMachine,
    StateMachineConfig,
)


def _cfg(**over) -> StateMachineConfig:
    base = dict(
        idle_quiet_s=10,
        countdown_s=5,
        eval_window_s=6,
        challenge_window_s=4,
        challenge_retries=1,
        alarm_grace_s=7,
        k_consecutive=3,
        enabled_challenges=tuple(ChallengeKind),
    )
    base.update(over)
    return StateMachineConfig(**base)


def _sm(**over) -> StateMachine:
    return StateMachine(cfg=_cfg(**over), rng=Random(0))


def _kinds(events):
    return [e.kind for e in events]


def _to_armed(m: StateMachine) -> None:
    m.update(Perception(now=0))  # sets last_pir = 0
    m.update(Perception(now=10))  # quiet 10s -> ARMING (deadline 15)
    m.update(Perception(now=15))  # countdown elapsed -> ARMED
    assert m.state == AlarmState.ARMED


def _to_challenge(m: StateMachine) -> None:
    _to_armed(m)
    m.update(Perception(now=16, pir=True, person=True))  # MOTION -> EVALUATING
    for t in (16.2, 16.4, 16.6):  # K=3 good frames
        m.update(Perception(now=t, pir=True, person=True, geometry_ok=True, identity_match=True))
    assert m.state == AlarmState.CHALLENGE


def test_idle_arms_after_quiet():
    m = _sm()
    assert m.update(Perception(now=0)) == []
    assert m.update(Perception(now=5)) == []
    ev = m.update(Perception(now=10))
    assert _kinds(ev) == [EventKind.ARMING]
    assert m.state == AlarmState.ARMING


def test_arming_cancelled_by_motion():
    m = _sm()
    m.update(Perception(now=0))
    m.update(Perception(now=10))
    ev = m.update(Perception(now=11, person=True))
    assert _kinds(ev) == [EventKind.ARMING_CANCELLED]
    assert m.state == AlarmState.IDLE


def test_arming_completes_to_armed():
    m = _sm()
    m.update(Perception(now=0))
    m.update(Perception(now=10))
    assert m.update(Perception(now=12)) == []
    assert m.state == AlarmState.ARMING
    m.update(Perception(now=15))
    assert m.state == AlarmState.ARMED


def test_motion_starts_evaluating():
    m = _sm()
    _to_armed(m)
    ev = m.update(Perception(now=16, pir=True, person=True))
    assert _kinds(ev) == [EventKind.MOTION]
    assert m.state == AlarmState.EVALUATING


def test_recognised_and_live_reaches_challenge():
    m = _sm()
    _to_challenge(m)
    assert m.current_challenge in set(ChallengeKind)


def test_geometry_fail_never_confirms_and_alarms():
    m = _sm()
    _to_armed(m)
    m.update(Perception(now=16, pir=True, person=True))  # EVALUATING, deadline 22
    # geometry keeps failing, so identity is never even reached; streak stays 0
    m.update(Perception(now=18, pir=True, person=True, geometry_ok=False, identity_match=True))
    ev = m.update(Perception(now=22, pir=True, person=True, geometry_ok=False))
    assert EventKind.ALARM in _kinds(ev)
    assert m.state == AlarmState.ALARM


def test_unrecognised_times_out_to_alarm():
    m = _sm()
    _to_armed(m)
    m.update(Perception(now=16, pir=True, person=True))
    m.update(Perception(now=18, pir=True, person=True, geometry_ok=True, identity_match=False))
    ev = m.update(Perception(now=22, pir=True, person=True, geometry_ok=True, identity_match=False))
    assert EventKind.ALARM in _kinds(ev)


def test_challenge_passed_greets_and_disarms():
    m = _sm()
    _to_challenge(m)
    ev = m.update(Perception(now=17, challenge_passed=True))
    assert _kinds(ev) == [EventKind.GREETED, EventKind.DISARMED]
    assert m.state == AlarmState.IDLE


def test_challenge_retries_once_then_alarms():
    m = _sm()
    _to_challenge(m)  # issued ~16.6, window 4 -> deadline ~20.6, retries_left=1
    ev = m.update(Perception(now=21))  # elapsed, retry remains -> re-issue
    assert EventKind.CHALLENGE_ISSUED in _kinds(ev)
    assert m.state == AlarmState.CHALLENGE
    ev = m.update(Perception(now=26))  # elapsed again, no retries -> alarm
    assert EventKind.ALARM in _kinds(ev)
    assert m.state == AlarmState.ALARM


def test_alarm_clears_after_quiet_grace():
    m = _sm()
    _to_armed(m)
    m.update(Perception(now=16, pir=True, person=True))
    m.update(Perception(now=22, pir=True, person=True, geometry_ok=False))  # -> ALARM
    assert m.state == AlarmState.ALARM
    m.update(Perception(now=23))  # quiet begins
    ev = m.update(Perception(now=30))  # quiet for 7s grace -> disarm
    assert EventKind.DISARMED in _kinds(ev)
    assert m.state == AlarmState.IDLE

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


def _cfg() -> StateMachineConfig:
    return StateMachineConfig(
        idle_quiet_s=10,
        countdown_s=5,
        eval_window_s=6,
        challenge_window_s=4,
        challenge_retries=1,
        alarm_grace_s=7,
        k_consecutive=3,
        enabled_challenges=(ChallengeKind.SMILE,),
    )


def _sm(state=AlarmState.IDLE) -> StateMachine:
    sm = StateMachine(cfg=_cfg(), rng=Random(0))
    sm.state = state
    return sm


def test_arm_switch_off_disarms_an_active_alarm():
    sm = _sm(AlarmState.ALARM)
    events = sm.update(Perception(now=100.0, armed_switch=False))
    assert sm.state is AlarmState.IDLE
    assert EventKind.DISARMED in [e.kind for e in events]


def test_arm_switch_off_blocks_auto_arming():
    sm = _sm(AlarmState.IDLE)
    # Even after well past the idle-quiet window, a disarmed switch keeps it idle and silent.
    assert sm.update(Perception(now=0.0, armed_switch=False)) == []
    assert sm.update(Perception(now=10_000.0, armed_switch=False)) == []
    assert sm.state is AlarmState.IDLE


def test_arm_switch_off_while_idle_emits_nothing():
    sm = _sm(AlarmState.IDLE)
    assert sm.update(Perception(now=5.0, armed_switch=False)) == []


def test_arm_switch_on_resumes_autonomous_arming():
    sm = _sm(AlarmState.IDLE)
    sm.update(Perception(now=0.0, armed_switch=True))  # sets the quiet baseline
    events = sm.update(Perception(now=20.0, armed_switch=True))  # quiet > idle_quiet_s
    assert any(e.kind is EventKind.ARMING for e in events)
    assert sm.state is AlarmState.ARMING


def test_none_switch_preserves_legacy_flow():
    # No switch signal must behave exactly as before (autonomous arming).
    sm = _sm(AlarmState.IDLE)
    sm.update(Perception(now=0.0))
    events = sm.update(Perception(now=20.0))
    assert any(e.kind is EventKind.ARMING for e in events)

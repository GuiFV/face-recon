from __future__ import annotations

from random import Random

from face_recon.core.config import Settings
from face_recon.core.models import BBox, ChallengeKind, ChallengeSignals
from face_recon.pipeline.orchestrator import Orchestrator
from face_recon.pipeline.state_machine import (
    AlarmState,
    Event,
    EventKind,
    StateMachine,
    StateMachineConfig,
)


class FakeWebhook:
    def __init__(self):
        self.events = []

    def emit(self, event, ts, extra=None):
        self.events.append(event.kind)


def _sm():
    cfg = StateMachineConfig(
        idle_quiet_s=10,
        countdown_s=5,
        eval_window_s=6,
        challenge_window_s=4,
        challenge_retries=1,
        alarm_grace_s=7,
        k_consecutive=3,
        enabled_challenges=(ChallengeKind.SMILE,),
    )
    return StateMachine(cfg=cfg, rng=Random(0))


def _orch(references=None, **settings_over):
    wh = FakeWebhook()
    orch = Orchestrator(
        Settings(_env_file=None, **settings_over), _sm(), wh, references=references
    )
    return orch, wh


class FakeStream:
    def __init__(self, frames):
        self._frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def frames(self):
        return iter(self._frames)


class FakeInference:
    def __init__(self, response):
        self.response = response

    def infer(self, frame, model_id):
        return self.response


class FakeFace:
    def __init__(self, bbox, embedding):
        self.bbox = bbox
        self.embedding = embedding


class FakeEmbedder:
    def __init__(self, face):
        self.face = face

    def best_face(self, frame):
        return self.face


class FakeLandmarks:
    def __init__(self, sig):
        self.sig = sig

    def signals(self, frame):
        return self.sig


def _pose_response(pose):
    """Turn a Pose fixture into a Roboflow-style pose response."""
    return {
        "predictions": [
            {
                "keypoints": [
                    {"x": kp.x, "y": kp.y, "confidence": kp.confidence, "class_name": name}
                    for name, kp in pose.keypoints.items()
                ]
            }
        ]
    }


def _live_orch(*, references, inference=None, face_embedder=None, landmarks=None):
    orch = Orchestrator(
        Settings(_env_file=None),
        _sm(),
        FakeWebhook(),
        references=references,
        inference=inference,
        face_embedder=face_embedder,
        landmarks=landmarks,
    )
    captured: list = []
    orch.tick = lambda p: captured.append(p) or []  # spy on the folded Perception
    return orch, captured


def test_every_event_is_webhooked():
    orch, wh = _orch()
    orch._handle_events([Event(EventKind.ARMING), Event(EventKind.DISARMED)], now=1.0)
    assert wh.events == [EventKind.ARMING, EventKind.DISARMED]


def test_challenge_issued_clears_the_signal_buffer():
    orch, _ = _orch()
    orch.add_challenge_signal(ChallengeSignals(smile=0.1))
    orch._handle_events(
        [Event(EventKind.CHALLENGE_ISSUED, challenge=ChallengeKind.SMILE)], now=1.0
    )
    assert orch._challenge_signals == []


def test_challenge_passed_evaluates_current_challenge():
    orch, _ = _orch()
    orch.sm.state = AlarmState.CHALLENGE
    orch.sm._challenge = ChallengeKind.SMILE
    assert orch.challenge_passed() is False  # no signals yet
    orch.add_challenge_signal(ChallengeSignals(smile=0.1))  # neutral
    orch.add_challenge_signal(ChallengeSignals(smile=0.7))  # smile
    assert orch.challenge_passed() is True


def test_perception_runs_identity_only_after_geometry_gate(upper_body_pose):
    orch, _ = _orch(references=[[1.0, 0.0]])
    good_face = BBox(46, 74, 54, 86)  # centred above the shoulders, plausible size
    p = orch.perception_from(
        now=1.0,
        pir=True,
        person=True,
        pose=upper_body_pose,
        face_bbox=good_face,
        embedding=[1.0, 0.0],
    )
    assert p.geometry_ok is True
    assert p.identity_match is True  # cosine 1.0 vs the reference >= threshold


def test_perception_geometry_fail_skips_identity(upper_body_pose):
    orch, _ = _orch(references=[[1.0, 0.0]])
    chest_face = BBox(46, 110, 54, 122)  # face below the shoulders: a phone held at chest
    p = orch.perception_from(
        now=1.0,
        pir=True,
        person=True,
        pose=upper_body_pose,
        face_bbox=chest_face,
        embedding=[1.0, 0.0],
    )
    assert p.geometry_ok is False
    assert p.identity_match is None  # never evaluated


# --- control API ---


def test_arm_disarm_override():
    orch, _ = _orch()
    assert orch._armed_override is None
    orch.arm()
    assert orch._armed_override is True
    orch.disarm()
    assert orch._armed_override is False
    orch.clear_arm_override()
    assert orch._armed_override is None


def test_push_signal_keeps_pir_hot_for_the_ttl():
    t = [100.0]
    orch = Orchestrator(
        Settings(_env_file=None, external_pir_ttl_seconds=5.0),
        _sm(),
        FakeWebhook(),
        clock=lambda: t[0],
    )
    assert orch._external_pir_active(t[0]) is False
    orch.push_signal()
    assert orch._external_pir_active(t[0]) is True  # hot now
    t[0] = 104.0
    assert orch._external_pir_active(t[0]) is True  # still within the 5s window
    t[0] = 106.0
    assert orch._external_pir_active(t[0]) is False  # window elapsed


# --- live frame processing ---


def test_process_frame_folds_pose_and_face(upper_body_pose):
    face = FakeFace(bbox=(46, 74, 54, 86), embedding=[1.0, 0.0])  # centred above shoulders
    orch, captured = _live_orch(
        references=[[1.0, 0.0]],
        inference=FakeInference(_pose_response(upper_body_pose)),
        face_embedder=FakeEmbedder(face),
    )
    orch.process_frame(object())
    p = captured[0]
    assert p.person is True
    assert p.geometry_ok is True
    assert p.identity_match is True  # embedding matches the reference


def test_process_frame_no_person_when_no_pose():
    orch, captured = _live_orch(
        references=[[1.0, 0.0]],
        inference=FakeInference({"predictions": []}),
        face_embedder=FakeEmbedder(None),
    )
    orch.process_frame(object())
    p = captured[0]
    assert p.person is False
    assert p.geometry_ok is None and p.identity_match is None


def test_process_frame_buffers_challenge_signals_instead_of_matching(upper_body_pose):
    face = FakeFace(bbox=(46, 74, 54, 86), embedding=[1.0, 0.0])
    sig = ChallengeSignals(smile=0.7)
    orch, captured = _live_orch(
        references=[[1.0, 0.0]],
        inference=FakeInference(_pose_response(upper_body_pose)),
        face_embedder=FakeEmbedder(face),
        landmarks=FakeLandmarks(sig),
    )
    orch.sm.state = AlarmState.CHALLENGE
    orch.sm._challenge = ChallengeKind.SMILE
    orch.process_frame(object())
    assert orch._challenge_signals == [sig]  # signal buffered for the challenge
    assert captured[0].identity_match is False  # identity not matched during a challenge


def test_run_feeds_control_state_to_each_frame():
    orch = Orchestrator(
        Settings(_env_file=None), _sm(), FakeWebhook(), stream=FakeStream([object(), object()])
    )
    seen: list = []
    orch.process_frame = lambda frame, **kw: seen.append(kw) or []
    orch.arm()
    orch.run()  # processes both frames, then the stream ends
    assert len(seen) == 2
    assert all(kw["armed_switch"] is True for kw in seen)  # arm() override threaded through
    assert all(kw["external_pir"] is False for kw in seen)  # no signal pushed


def test_process_frame_stashes_a_debug_jpeg(upper_body_pose):
    import numpy as np

    face = FakeFace(bbox=(46, 74, 54, 86), embedding=[1.0, 0.0])
    orch, _ = _live_orch(
        references=[[1.0, 0.0]],
        inference=FakeInference(_pose_response(upper_body_pose)),
        face_embedder=FakeEmbedder(face),
    )
    assert orch.latest_debug_jpeg() is None  # nothing processed yet
    orch.process_frame(np.zeros((120, 160, 3), dtype=np.uint8))
    jpeg = orch.latest_debug_jpeg()
    assert jpeg and jpeg[:2] == b"\xff\xd8"

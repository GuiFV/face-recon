from __future__ import annotations

from face_recon.core.models import ChallengeKind
from face_recon.pipeline.state_machine import Event, EventKind
from face_recon.services.webhook import WebhookEmitter


class _Resp:
    def raise_for_status(self):
        return None


class _FailResp:
    def raise_for_status(self):
        raise RuntimeError("boom")


def _recorder(responses=None):
    calls = []

    def post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return (responses or {}).get(url, _Resp())

    post.calls = calls
    return post


def test_payload_includes_event_and_ts():
    em = WebhookEmitter([], poster=_recorder())
    payload = em.build_payload(Event(EventKind.ALARM), ts=123.0)
    assert payload == {"event": "alarm", "ts": 123.0}


def test_payload_includes_challenge_kind():
    em = WebhookEmitter([], poster=_recorder())
    payload = em.build_payload(
        Event(EventKind.CHALLENGE_ISSUED, challenge=ChallengeKind.SMILE), ts=1.0
    )
    assert payload["event"] == "challenge_issued"
    assert payload["challenge"] == "smile"


def test_emit_posts_to_every_url():
    post = _recorder()
    em = WebhookEmitter(["http://a/hook", "http://b/hook"], poster=post)
    sent = em.emit(Event(EventKind.GREETED), ts=5.0, extra={"track": 7})
    assert sent == 2
    assert {c["url"] for c in post.calls} == {"http://a/hook", "http://b/hook"}
    assert post.calls[0]["json"] == {"event": "greeted", "ts": 5.0, "track": 7}


def test_emit_swallows_failures_and_counts_successes():
    post = _recorder(responses={"http://bad/hook": _FailResp()})
    em = WebhookEmitter(["http://ok/hook", "http://bad/hook"], poster=post)
    sent = em.emit(Event(EventKind.MOTION), ts=1.0)
    assert sent == 1  # the failing one is logged, not raised


def test_blank_urls_are_filtered():
    em = WebhookEmitter(["", "http://a/hook", ""], poster=_recorder())
    assert em.urls == ["http://a/hook"]

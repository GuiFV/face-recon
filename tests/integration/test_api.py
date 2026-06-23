from __future__ import annotations

from fastapi.testclient import TestClient

from face_recon.api.app import create_app
from face_recon.core.config import Settings


def _client() -> TestClient:
    return TestClient(create_app(Settings(_env_file=None)))


def test_healthz_ok():
    resp = _client().get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_metrics_exposed():
    resp = _client().get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    # The metric definitions should be present in the exposition output.
    assert "face_recon_greetings_total" in resp.text


def test_status_reports_initial_state():
    resp = _client().get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "idle"
    assert body["challenge"] is None


def test_arm_delay_get_returns_configured_default():
    resp = _client().get("/config/arm-delay")
    assert resp.status_code == 200
    assert resp.json()["seconds"] == Settings(_env_file=None).idle_quiet_seconds


def test_arm_delay_set_takes_effect():
    client = _client()
    assert client.post("/config/arm-delay", params={"seconds": 300}).status_code == 200
    assert client.get("/config/arm-delay").json()["seconds"] == 300


def test_arm_delay_rejects_non_positive():
    assert _client().post("/config/arm-delay", params={"seconds": 0}).status_code == 422

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

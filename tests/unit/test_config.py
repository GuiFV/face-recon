from __future__ import annotations

from face_recon.core.config import Settings


def test_defaults_are_sane():
    s = Settings(_env_file=None)
    assert s.k_consecutive == 5
    assert 0.0 < s.face_match_threshold < 1.0
    assert s.head_shoulder_ratio_min < s.head_shoulder_ratio_max
    assert s.face_body_scale_min < s.face_body_scale_max
    assert s.arming_countdown_seconds > 0
    assert s.enabled_challenge_list == ["smile", "blink", "turn_left", "turn_right"]


def test_roboflow_uses_plain_env_names(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "secret-key")
    monkeypatch.setenv("ROBOFLOW_WORKSPACE", "my-workspace")
    s = Settings(_env_file=None)
    assert s.roboflow_api_key == "secret-key"
    assert s.roboflow_workspace == "my-workspace"


def test_prefixed_env_override(monkeypatch):
    monkeypatch.setenv("FACE_RECON_K_CONSECUTIVE", "9")
    monkeypatch.setenv("FACE_RECON_FACE_MATCH_THRESHOLD", "0.5")
    monkeypatch.setenv("FACE_RECON_IDLE_QUIET_SECONDS", "300")
    s = Settings(_env_file=None)
    assert s.k_consecutive == 9
    assert s.face_match_threshold == 0.5
    assert s.idle_quiet_seconds == 300


def test_webhook_url_list_splits():
    s = Settings(_env_file=None, webhook_urls="http://a/,http://b/ ,  ")
    assert s.webhook_url_list == ["http://a/", "http://b/"]

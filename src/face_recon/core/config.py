"""Application settings.

Project settings use the FACE_RECON_ env prefix. The Roboflow credentials use the plain
names RF_API_KEY and RF_WORKSPACE (matching the deployment .env). Kept free of heavy CV
imports so it stays cheap to import and easy to test.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FACE_RECON_",
        env_file=".env",
        extra="ignore",
    )

    # --- Roboflow (plain env names, no prefix) ---
    roboflow_api_key: str = Field(default="", validation_alias="RF_API_KEY")
    roboflow_workspace: str = Field(default="", validation_alias="RF_WORKSPACE")
    inference_server_url: str = "http://localhost:9001"

    # --- Camera stream (set to your camera's MJPEG/RTSP URL) ---
    stream_url: str = "http://pi-host:8081"

    # --- Enrolled reference embeddings (built by enrolment/enrol.py) ---
    reference_path: str = "enrolment/reference/known.npz"

    # --- Pose model: one call gives the person + their skeleton (confirm id on deploy) ---
    pose_model_id: str = "yolov8n-pose-640"

    # --- Identity match (ArcFace cosine; reference set built by enrolment/enrol.py) ---
    # Live face matched against the reference set; the decision uses the mean of the top-k
    # similarities (robust to one lucky reference). The enrolled person's intra-cosine ~0.67
    # and strangers sit below ~0.3, so 0.42 is a safe start; tune against live frames.
    face_match_threshold: float = 0.42
    face_match_top_k: int = 5
    match_label: str = "known"  # on-screen label for a recognised face; set your name via env
    keypoint_min_confidence: float = 0.3
    k_consecutive: int = 5

    # --- Geometry gate (must pass before identity is even checked) ---
    head_shoulder_ratio_min: float = 0.25
    head_shoulder_ratio_max: float = 0.75
    face_body_scale_min: float = 0.15
    face_body_scale_max: float = 0.80

    # --- Control inputs ---
    # How long an external PIR/motion push (POST /signal) keeps motion "hot".
    external_pir_ttl_seconds: float = 5.0

    # --- State machine timing (seconds) ---
    idle_quiet_seconds: float = 60.0  # seconds of quiet before the box auto-arms
    arming_countdown_seconds: float = 10.0
    eval_window_seconds: float = 6.0
    challenge_window_seconds: float = 4.0
    challenge_retries: int = 1
    alarm_grace_seconds: float = 7.0

    # --- Liveness challenge thresholds (signals are model-normalised) ---
    smile_neutral_below: float = 0.20
    smile_above: float = 0.50
    blink_open_above: float = 0.20
    blink_closed_below: float = 0.10
    turn_centre_band: float = 0.15
    turn_threshold: float = 0.35
    enabled_challenges: str = "smile,blink,turn_left,turn_right"

    # --- Outbound webhooks (the events the box emits; handlers subscribe) ---
    webhook_urls: str = ""  # comma-separated

    # --- Observability (bonus; not in the signal path) ---
    metrics_enabled: bool = True

    # --- Live CV runtime (GPU host only; off by default so the app/tests stand up bare) ---
    live_enabled: bool = False
    face_use_gpu: bool = True  # run insightface on the GPU
    face_landmarker_model_path: str = "models/face_landmarker.task"  # MediaPipe asset

    # --- Service ---
    log_level: str = "INFO"

    @property
    def enabled_challenge_list(self) -> list[str]:
        return [c.strip() for c in self.enabled_challenges.split(",") if c.strip()]

    @property
    def webhook_url_list(self) -> list[str]:
        return [u.strip() for u in self.webhook_urls.split(",") if u.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance for the process."""
    return Settings()

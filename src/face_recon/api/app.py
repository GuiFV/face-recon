"""FastAPI application factory for the decision service.

Builds the orchestrator (state machine + webhook emitter + enrolled references + CV services)
and exposes health, the current status, the control API, Prometheus metrics, and the annotated
debug stream. The live per-frame loop (orchestrator.run) starts in a background thread when
settings.live_enabled is set; the app itself stands up without the CV stack present.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from random import Random

from fastapi import FastAPI

from face_recon.api.routes import router
from face_recon.core import metrics as _metrics  # noqa: F401  (registers metrics on import)
from face_recon.core.config import Settings, get_settings
from face_recon.core.logging import configure_logging, get_logger
from face_recon.enrolment.enrol import load_reference_set
from face_recon.pipeline.orchestrator import Orchestrator
from face_recon.pipeline.state_machine import StateMachine, StateMachineConfig
from face_recon.services.face_embed import FaceEmbedder
from face_recon.services.inference import InferenceClient
from face_recon.services.landmarks import LandmarkProvider
from face_recon.services.stream import StreamConsumer
from face_recon.services.webhook import WebhookEmitter

logger = get_logger(__name__)


def build_orchestrator(settings: Settings) -> Orchestrator:
    """Assemble the orchestrator and its services from settings.

    The CV services are constructed here but import their heavy libraries lazily and open no
    network connections until used, so this stays cheap and the app stands up without the GPU
    stack present. The per-frame loop only runs when settings.live_enabled is set (GPU host).
    """
    sm = StateMachine(cfg=StateMachineConfig.from_settings(settings), rng=Random())
    webhook = WebhookEmitter(settings.webhook_url_list)
    references = load_reference_set(settings.reference_path)
    inference = InferenceClient(settings.inference_server_url, settings.roboflow_api_key)
    pose_provider = None
    if settings.pose_provider == "local":
        from face_recon.pipeline.pose import LocalPoseProvider

        pose_provider = LocalPoseProvider(settings.local_pose_model_path)
    face_embedder = FaceEmbedder(use_gpu=settings.face_use_gpu)
    landmarks = LandmarkProvider(settings.face_landmarker_model_path)
    stream = StreamConsumer(settings.stream_url)
    return Orchestrator(
        settings,
        sm,
        webhook,
        references=references,
        inference=inference,
        pose_provider=pose_provider,
        face_embedder=face_embedder,
        landmarks=landmarks,
        stream=stream,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Accepts injected settings for testing."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    orchestrator = build_orchestrator(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "decision service starting; stream=%s references=%d live=%s",
            settings.stream_url,
            len(orchestrator.references),
            settings.live_enabled,
        )
        thread: threading.Thread | None = None
        if settings.live_enabled:
            thread = threading.Thread(
                target=orchestrator.run, name="frame-loop", daemon=True
            )
            thread.start()
        try:
            yield
        finally:
            orchestrator.stop()
            if thread is not None:
                thread.join(timeout=5.0)

    app = FastAPI(title="face-recon decision service", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.orchestrator = orchestrator
    app.include_router(router)
    return app

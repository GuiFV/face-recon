"""HTTP routes for the decision service: health and Prometheus metrics.

Kept import light so the API can come up and report health even before the CV pipeline is
wired to the live box.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe. Returns ok when the service process is up."""
    return {"status": "ok"}


@router.get("/status")
def status(request: Request) -> dict:
    """Current alarm state and any active liveness challenge."""
    orch = request.app.state.orchestrator
    challenge = orch.sm.current_challenge
    return {
        "state": str(orch.sm.state),
        "challenge": str(challenge) if challenge else None,
    }


# --- control API: an integration pushes arm/disarm and optional motion here ---


@router.post("/arm")
def arm(request: Request) -> dict:
    """Force the box armed (master override; e.g. a physical switch wired in)."""
    request.app.state.orchestrator.arm()
    return {"armed": True}


@router.post("/disarm")
def disarm(request: Request) -> dict:
    """Force the box disarmed (master override)."""
    request.app.state.orchestrator.disarm()
    return {"armed": False}


@router.post("/arm/auto")
def arm_auto(request: Request) -> dict:
    """Release the override; hand arming back to the autonomous quiet-then-arm flow."""
    request.app.state.orchestrator.clear_arm_override()
    return {"armed": None}


@router.post("/signal")
def signal(request: Request) -> dict:
    """Push an external motion/PIR pulse (kept 'hot' for a short window)."""
    request.app.state.orchestrator.push_signal()
    return {"ok": True}


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics in the text exposition format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/debug/frame")
def debug_frame(request: Request) -> Response:
    """The latest processed frame, annotated with skeleton + face box + identity label."""
    jpeg = request.app.state.orchestrator.latest_debug_jpeg()
    if not jpeg:
        return Response(content=b"no frame yet", status_code=503)
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/debug/stream")
async def debug_stream(request: Request) -> StreamingResponse:
    """Live MJPEG of the annotated frames (skeleton + face + identity). For bring-up/demo."""
    orch = request.app.state.orchestrator

    async def gen():
        while not await request.is_disconnected():
            jpeg = orch.latest_debug_jpeg()
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

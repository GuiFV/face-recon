"""Per-frame orchestrator: the runtime spine of the decision service.

Pulls frames from the stream, runs the CV pipeline (detect -> pose -> geometry gate -> face
match; during a challenge, face landmarks -> challenge signals), folds the results into a
Perception, drives the state machine, and emits each event as a webhook. The box is
self-contained: its only inputs are the video stream and a small control API (arm / disarm /
an optional external PIR/motion push), and its only outputs are webhooks plus the status and
debug endpoints. It knows nothing about GPIO, sirens, or any specific hardware; an integration
turns the emitted events into whatever it wants.

The pure parts are unit tested: the event -> webhook mapping (`_handle_events`), the
challenge-signal buffering (`add_challenge_signal` / `challenge_passed`), and how CV results
fold into a Perception (`perception_from`, which applies the geometry gate before identity).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from face_recon.core.config import Settings
from face_recon.core.logging import get_logger
from face_recon.core.models import BBox, ChallengeSignals, Pose
from face_recon.pipeline import challenges as ch
from face_recon.pipeline import faces, liveness
from face_recon.pipeline import pose as pose_stage
from face_recon.pipeline.challenges import ChallengeConfig
from face_recon.pipeline.debug import DebugSnapshot
from face_recon.pipeline.state_machine import (
    AlarmState,
    Event,
    EventKind,
    Perception,
    StateMachine,
)
from face_recon.services.face_embed import FaceEmbedder
from face_recon.services.inference import InferenceClient
from face_recon.services.landmarks import LandmarkProvider
from face_recon.services.stream import StreamConsumer
from face_recon.services.webhook import WebhookEmitter

logger = get_logger(__name__)


def challenge_config_from_settings(s: Settings) -> ChallengeConfig:
    return ChallengeConfig(
        smile_neutral_below=s.smile_neutral_below,
        smile_above=s.smile_above,
        blink_open_above=s.blink_open_above,
        blink_closed_below=s.blink_closed_below,
        turn_centre_band=s.turn_centre_band,
        turn_threshold=s.turn_threshold,
    )


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        sm: StateMachine,
        webhook: WebhookEmitter,
        *,
        references: list | None = None,
        clock: Callable[[], float] = time.monotonic,
        inference: InferenceClient | None = None,
        face_embedder: FaceEmbedder | None = None,
        landmarks: LandmarkProvider | None = None,
        stream: StreamConsumer | None = None,
    ) -> None:
        self.settings = settings
        self.sm = sm
        self.webhook = webhook
        self.references = references or []
        self.clock = clock
        # Live CV services. Left None in unit tests, which drive the pure methods.
        self._inference = inference
        self._face_embedder = face_embedder
        self._landmarks = landmarks
        self._stream = stream
        self.challenge_cfg = challenge_config_from_settings(settings)
        self._challenge_signals: list[ChallengeSignals] = []
        self._running = False
        self._last_match = None  # most recent FaceMatch, for the debug overlay
        self._debug: DebugSnapshot | None = None  # latest frame + CV results, for /debug
        # Control inputs pushed via the API (thread-safe enough as plain attribute writes).
        self._armed_override: bool | None = None  # set by arm()/disarm(); None = autonomous
        self._pir_until: float | None = None  # external PIR/motion is "hot" until this time

    # --- control API (fed by the HTTP endpoints) ---

    def arm(self) -> None:
        self._armed_override = True

    def disarm(self) -> None:
        self._armed_override = False

    def clear_arm_override(self) -> None:
        """Hand arming back to the autonomous quiet-then-arm flow."""
        self._armed_override = None

    def push_signal(self) -> None:
        """An external motion/PIR pulse; keeps PIR 'hot' for a short window."""
        self._pir_until = self.clock() + self.settings.external_pir_ttl_seconds

    def _external_pir_active(self, now: float) -> bool:
        return self._pir_until is not None and now < self._pir_until

    # --- pure coordination (unit tested) ---

    def tick(self, p: Perception) -> list[Event]:
        """Advance the state machine one step and emit the resulting events."""
        events = self.sm.update(p)
        self._handle_events(events, p.now)
        return events

    def _handle_events(self, events: list[Event], now: float) -> None:
        for ev in events:
            self.webhook.emit(ev, now)
            if ev.kind is EventKind.CHALLENGE_ISSUED:
                self._challenge_signals.clear()  # fresh window for the new challenge

    def add_challenge_signal(self, sig: ChallengeSignals) -> None:
        self._challenge_signals.append(sig)

    def challenge_passed(self) -> bool:
        kind = self.sm.current_challenge
        if kind is None:
            return False
        return ch.evaluate_challenge(kind, self._challenge_signals, self.challenge_cfg)

    def perception_from(
        self,
        now: float,
        *,
        pir: bool,
        person: bool,
        pose: Pose | None,
        face_bbox: BBox | None,
        embedding: list | None,
        armed_switch: bool | None = None,
    ) -> Perception:
        """Fold one frame of CV results into a Perception, applying the geometry gate before
        identity: identity is only evaluated when the geometry gate passes."""
        s = self.settings
        self._last_match = None
        geometry_ok: bool | None = None
        identity_match: bool | None = None
        if person and pose is not None:
            geometry_ok = liveness.geometry_gate_ok(
                pose,
                face_bbox,
                min_confidence=s.keypoint_min_confidence,
                head_shoulder_ratio_min=s.head_shoulder_ratio_min,
                head_shoulder_ratio_max=s.head_shoulder_ratio_max,
                face_body_scale_min=s.face_body_scale_min,
                face_body_scale_max=s.face_body_scale_max,
            )
            if geometry_ok:
                if embedding is not None and self.references:
                    match = faces.match_embedding(
                        embedding,
                        self.references,
                        s.face_match_threshold,
                        top_k=s.face_match_top_k,
                    )
                    self._last_match = match
                    identity_match = match.is_match
                else:
                    identity_match = False
        challenge_passed = (
            self.challenge_passed() if self.sm.state is AlarmState.CHALLENGE else None
        )
        return Perception(
            now=now,
            pir=pir,
            person=person,
            geometry_ok=geometry_ok,
            identity_match=identity_match,
            challenge_passed=challenge_passed,
            armed_switch=armed_switch,
        )

    # --- live wiring ---

    def _primary_pose(self, frame) -> Pose | None:
        """Run the pose model on the frame and return the nearest person's pose."""
        if self._inference is None:
            return None
        poses = pose_stage.poses_from_frame(frame, self._inference, self.settings.pose_model_id)
        return pose_stage.primary_pose(poses)

    def process_frame(
        self, frame, *, external_pir: bool = False, armed_switch: bool | None = None
    ) -> list[Event]:
        """Extract CV results from one frame, fold into a Perception, then tick.

        Presence comes from the camera (a detected person). `external_pir` lets an integration
        assert motion from a real PIR; when it is not supplied, presence is purely camera-driven
        (pir mirrors person). The face box and its ArcFace embedding come from insightface.
        During a challenge we collect landmark signals instead of matching identity.
        """
        now = self.clock()
        pose = self._primary_pose(frame)
        person = pose is not None
        pir = external_pir or person  # camera-driven; an external PIR can also assert motion

        face = self._face_embedder.best_face(frame) if self._face_embedder else None
        face_bbox = BBox(*face.bbox) if face is not None else None

        embedding: list | None = None
        if self.sm.state is AlarmState.CHALLENGE:
            if self._landmarks is not None:
                sig = self._landmarks.signals(frame)
                if sig is not None:
                    self.add_challenge_signal(sig)
        elif face is not None:
            embedding = list(face.embedding)

        p = self.perception_from(
            now,
            pir=pir,
            person=person,
            pose=pose,
            face_bbox=face_bbox,
            embedding=embedding,
            armed_switch=armed_switch,
        )
        events = self.tick(p)
        self._stash_debug(frame, pose, face_bbox)
        return events

    def _stash_debug(self, frame, pose: Pose | None, face_bbox: BBox | None) -> None:
        """Record the latest frame + CV results for the /debug overlay."""
        parts = [self.sm.state.value]
        if self._last_match is not None:
            who = self.settings.match_label if self._last_match.is_match else "stranger"
            parts.append(f"{who} {self._last_match.similarity:.2f}")
        if face_bbox is not None:
            parts.append(f"{int(face_bbox.width)}x{int(face_bbox.height)}px")
        self._debug = DebugSnapshot(
            frame=frame, pose=pose, face_bbox=face_bbox, label=" | ".join(parts)
        )

    def latest_debug_jpeg(self) -> bytes | None:
        """Render the latest snapshot to an annotated JPEG (None if nothing processed yet)."""
        snap = self._debug
        if snap is None:
            return None
        from face_recon.pipeline.debug import render_snapshot

        return render_snapshot(snap)

    def run(self) -> None:
        """Consume the camera stream and process frames until `stop()` is called."""
        if self._stream is None:
            raise RuntimeError("no stream configured; cannot run the live loop")
        self._running = True
        logger.info("frame loop starting")
        try:
            with self._stream as stream:
                for frame in stream.frames():
                    if not self._running:
                        break
                    try:
                        self.process_frame(
                            frame,
                            external_pir=self._external_pir_active(self.clock()),
                            armed_switch=self._armed_override,
                        )
                    except Exception:  # one bad frame must not kill the loop
                        logger.exception("frame processing failed; continuing")
        finally:
            logger.info("frame loop stopped")

    def stop(self) -> None:
        self._running = False

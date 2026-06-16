"""Outbound webhook emitter: the box's only outbound protocol.

On each state-machine event the orchestrator calls emit(); this POSTs a small JSON payload to
every configured webhook URL. Handlers subscribe simply by being a configured URL: an
integration decides what each event means (play a sound, fire a phone alert, drive a siren or
a light, log it). The box itself does none of that; it only announces what happened.

A delivery failure is logged, never raised into the decision loop: a missed webhook must not
crash the alarm. The HTTP poster is injected so the logic is unit testable without a network.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import httpx

from face_recon.core.logging import get_logger
from face_recon.pipeline.state_machine import Event

logger = get_logger(__name__)

Poster = Callable[..., Any]


class WebhookEmitter:
    def __init__(self, urls: Iterable[str], poster: Poster | None = None, timeout: float = 5.0):
        self.urls = [u for u in urls if u]
        self._post = poster or httpx.post
        self.timeout = timeout

    def build_payload(self, event: Event, ts: float, extra: dict | None = None) -> dict:
        payload: dict[str, Any] = {"event": str(event.kind), "ts": ts}
        if event.challenge is not None:
            payload["challenge"] = str(event.challenge)
        if extra:
            payload.update(extra)
        return payload

    def emit(self, event: Event, ts: float, extra: dict | None = None) -> int:
        """POST the event to every configured URL. Returns how many succeeded."""
        payload = self.build_payload(event, ts, extra)
        sent = 0
        for url in self.urls:
            try:
                resp = self._post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                sent += 1
            except Exception as exc:  # noqa: BLE001 (best-effort; never raise into the loop)
                logger.warning("webhook to %s failed: %s", url, exc)
        return sent

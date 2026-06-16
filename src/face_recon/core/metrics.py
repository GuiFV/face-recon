"""Prometheus metrics for the decision service.

Defined once at import time and registered against the default registry, which the
/metrics endpoint serialises. Per check spoof counters are recorded internally for
tuning; this is observability, not a user facing signal, so it does not breach the
fail silent rule (the stranger never learns which check failed).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

MATCH_CONFIDENCE = Histogram(
    "face_recon_match_confidence",
    "Cosine similarity of the best face match per evaluated track",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

LIVENESS_HEAD_MOTION = Histogram(
    "face_recon_liveness_head_motion",
    "Scale invariant head motion independent of the torso, per evaluated track",
    buckets=(0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32),
)

SPOOF_FLAGS = Counter(
    "face_recon_spoof_flags_total",
    "Tracks rejected by a liveness check, labelled by which check failed",
    ["check"],
)

GREETINGS = Counter(
    "face_recon_greetings_total",
    "Greetings triggered",
)

STRANGERS = Counter(
    "face_recon_strangers_total",
    "Stranger notifications sent",
)

PIPELINE_LATENCY = Histogram(
    "face_recon_pipeline_latency_seconds",
    "End to end processing latency per frame",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0),
)

FPS = Gauge(
    "face_recon_fps",
    "Processed frames per second",
)

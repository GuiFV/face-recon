"""Camera stream consumer.

Pulls decoded BGR frames from a camera stream URL (MJPEG or RTSP). The temporal liveness window
needs a reasonable frame rate, so a low-rate source is a known tuning point. opencv is a heavy
optional dependency, imported lazily.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

# On a dropped read the consumer reopens the capture with a capped backoff instead of giving up,
# so a transient camera or network outage does not end the frame loop.
RECONNECT_BACKOFF_S = 1.0
RECONNECT_MAX_S = 10.0


class StreamConsumer:
    """Yields decoded BGR frames from the camera stream URL."""

    def __init__(self, stream_url: str) -> None:
        self.stream_url = stream_url
        self._capture = None

    def __enter__(self) -> StreamConsumer:
        import cv2

        self._capture = cv2.VideoCapture(self.stream_url)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open stream: {self.stream_url}")
        return self

    def __exit__(self, *exc) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def frames(self) -> Iterator[np.ndarray | None]:
        """Yield decoded frames until the consumer stops iterating (or the context exits).

        On a dropped read (the stream stalled or the network blipped) the capture is reopened
        with a capped backoff rather than ending the loop, so a transient outage does not kill
        the frame loop. A single None is yielded per reconnect gap, so the consumer can drop any
        stale state before real frames resume.
        """
        if self._capture is None:
            raise RuntimeError("StreamConsumer must be used as a context manager")
        import cv2

        failures = 0
        while True:
            ok, frame = self._capture.read()
            if ok:
                failures = 0
                yield frame
                continue
            failures += 1
            try:
                self._capture.release()
            except Exception:
                pass
            time.sleep(min(RECONNECT_BACKOFF_S * failures, RECONNECT_MAX_S))
            self._capture = cv2.VideoCapture(self.stream_url)
            yield None

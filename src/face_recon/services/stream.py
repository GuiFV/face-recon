"""Camera stream consumer.

Pulls decoded BGR frames from a camera stream URL (MJPEG or RTSP). The temporal liveness window
needs a reasonable frame rate, so a low-rate source is a known tuning point. opencv is a heavy
optional dependency, imported lazily.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


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

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the stream ends or is closed.

        TODO(live): handle reconnection on dropped frames; a low-rate or flaky stream can
        stall, so a backoff and reopen loop belongs here.
        """
        if self._capture is None:
            raise RuntimeError("StreamConsumer must be used as a context manager")
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield frame

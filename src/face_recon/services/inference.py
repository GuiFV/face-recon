"""Client for the Roboflow self hosted inference server.

Wraps the inference HTTP API so the pipeline stages stay decoupled from the SDK. The
inference-sdk import is lazy so the rest of the package imports cheaply.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class InferenceClient:
    """Thin client over the Roboflow inference server."""

    def __init__(self, server_url: str, api_key: str) -> None:
        self.server_url = server_url
        self.api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from inference_sdk import InferenceHTTPClient

            self._client = InferenceHTTPClient(
                api_url=self.server_url,
                api_key=self.api_key,
            )
        return self._client

    def infer(self, frame: np.ndarray, model_id: str) -> dict[str, Any]:
        """Run a model on a frame and return the raw inference response.

        The SDK accepts a BGR numpy frame directly. Response shape is confirmed per model on
        deploy; `pipeline.pose.parse_poses` is defensive about the keypoint fields.
        """
        return self._ensure_client().infer(frame, model_id=model_id)

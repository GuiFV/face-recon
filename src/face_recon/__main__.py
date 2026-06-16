"""Run the decision service: python -m face_recon."""

from __future__ import annotations

import uvicorn

from face_recon.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "face_recon.api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

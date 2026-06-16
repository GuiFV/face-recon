# Decision service image for the GPU host (RTX 50-series / Blackwell).
#
# Runs the per-frame brain: pose via the inference server, insightface (face detect + ArcFace
# embedding) ON THE GPU, and MediaPipe landmarks for the liveness challenge. insightface runs on
# onnxruntime; like the inference server, Blackwell (sm_120) needs the cu128 build, so we pin
# onnxruntime-gpu==1.23.2 on a CUDA 12.8 + cuDNN base (see inference-blackwell.Dockerfile for the
# same fix and why the stock build fails with "no kernel image is available").
#
# The face model (insightface buffalo_l) and the MediaPipe face-landmarker asset are baked in at
# build time, so the container needs no network for models at runtime. The enrolled reference set
# (known.npz) is personal and is mounted at runtime, not copied into the image (see compose).
#
# Build context is the repo root (see docker-compose.yml). Build/run on the GPU host:
#   docker compose build decision-service && docker compose up -d
#
# If the tag below is unavailable, use any CUDA 12.8 cudnn-runtime tag the host can pull.

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# python plus the shared libraries opencv and mediapipe need even in headless mode.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Install the CV stack into a venv, then swap CPU onnxruntime for the Blackwell GPU build.
RUN python3 -m venv /opt/venv \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[cv]" \
 && pip uninstall -y onnxruntime \
 && pip install --no-cache-dir onnxruntime-gpu==1.23.2

# Bake the models so runtime needs no downloads: buffalo_l (face detect + ArcFace) and the
# MediaPipe face-landmarker asset. The download provider here is irrelevant; the GPU is used at
# runtime.
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=-1, det_size=(640, 640))" \
 && mkdir -p /app/models \
 && wget -qO /app/models/face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

EXPOSE 8000

CMD ["python", "-m", "face_recon"]

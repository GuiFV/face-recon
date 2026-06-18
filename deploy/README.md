# deploy (the GPU box)

The box runs as a small Docker stack on an NVIDIA GPU host: the Roboflow **inference server**
(person + pose), the **decision service** (the box itself), and optional **Prometheus +
Grafana** for observability. Wiring the box's webhooks/control API to real hardware is a
separate integration and is not part of this repo.

## Inference server

The decision service talks to a self-hosted Roboflow inference server on port 9001 for
**person detection and pose** (the skeleton the geometry gate needs). It runs models locally on
the GPU; only the model downloads and licence/auth touch Roboflow, never the video frames. Face
**recognition** does not run here: it uses insightface (ArcFace) inside the decision service,
because a general-purpose embedder could not reliably tell people apart.

### Blackwell / RTX-50 note (important)

The stock `roboflow/roboflow-inference-server-gpu` image ships a PyTorch built only up to
`sm_90` (Hopper). On a Blackwell GPU (RTX-50, `sm_120`) every model fails with
`CUDA error: no kernel image is available for execution on the device`. Use the
`inference-blackwell.Dockerfile` here instead, which layers the CUDA 12.8 (`cu128`) PyTorch +
a matching `onnxruntime-gpu` on top:

```bash
docker build -f deploy/inference-blackwell.Dockerfile -t roboflow-inference-blackwell .
docker run -d --name inference --restart unless-stopped --gpus all -p 9001:9001 \
    roboflow-inference-blackwell
```

On non-Blackwell GPUs the stock image works as-is. Verified on a Blackwell RTX-50 GPU with
`torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `onnxruntime-gpu 1.23.2`.

## Decision service

`decision-service.Dockerfile` is the per-frame brain. It runs insightface (face detect + ArcFace
embedding) **on the GPU**, so it carries the same Blackwell fix as the inference server: a CUDA
12.8 + cuDNN base with `onnxruntime-gpu==1.23.2`. The face model (`buffalo_l`) and the MediaPipe
face-landmarker asset are baked into the image, so no model downloads happen at runtime. The live
loop only runs when `FACE_RECON_LIVE_ENABLED=true` (set in compose).

Two things it needs at runtime:

- **The reference set.** `known.npz` is personal and gitignored, so it is mounted, not baked.
  Build it (see [`../enrolment/README.md`](../enrolment/README.md)), place it at
  `enrolment/reference/known.npz` on the host, and compose mounts that folder read-only.
- **The camera stream.** Set `FACE_RECON_STREAM_URL`. If the camera is on another host/network,
  give the container host networking or a route; the inference server is reachable by compose
  service name regardless.

## Full stack (compose)

`docker-compose.yml` brings up inference-server + decision-service + Prometheus + Grafana. It
reads the repo-root `.env` for `ROBOFLOW_API_KEY` / `ROBOFLOW_WORKSPACE` and the `FACE_RECON_*` settings
(run it from the repo root so `.env` is picked up).

## First-run checklist

1. `cp .env.example .env` and fill in `ROBOFLOW_API_KEY`, `FACE_RECON_STREAM_URL`, etc.
2. Build the reference set and place it at `enrolment/reference/known.npz` (see the enrolment
   guide).
3. `docker compose build && docker compose up -d`.
4. **Verify:** `curl :8000/healthz` and `:8000/status`; `:9001` for the inference server. The
   decision-service log should show `live=True` and the reference count.
5. **Calibrate** against live frames using `:8000/debug/stream`: confirm the pose model id, tune
   `FACE_RECON_FACE_MATCH_THRESHOLD`, and the head-yaw sign/scale for the turn challenge.

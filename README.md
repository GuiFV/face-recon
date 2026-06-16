# face-recon

A presence-aware, face-recognition **alarm box**. Point it at a camera stream; it watches for
people, recognises an enrolled person, runs an anti-spoofing liveness challenge, and announces
what happened as webhooks. It owns no hardware and makes no assumptions about your setup: the
only inputs are a video stream and a small control API, and the only outputs are webhooks plus
status/debug endpoints. What an event *means* (open a door, play a sound, sound a siren, send a
phone alert) is entirely up to whatever you wire to it.

> Built around the Roboflow stack (self-hosted inference server for person + pose) plus a
> dedicated face model for identity. Code quality, observability, and an honest account of the
> failure modes matter here as much as the feature.

## What it does

1. Consumes a camera stream (MJPEG/RTSP).
2. Detects a person and estimates their **pose / skeleton** (Roboflow inference server).
3. **Geometry gate (anti-spoof)** — before it even looks at identity, the body must check out:
   skeleton completeness, a plausible head-to-shoulder proportion, a face-to-body scale that
   matches, and the face sitting where a head should. A phone held at chest height fails here.
4. **Identity** — crops the face and matches it against your enrolled reference set with
   ArcFace embeddings (cosine similarity, top-k mean, against a threshold).
5. **Liveness challenge** — if it looks like the enrolled person, it issues a randomised,
   spoken challenge (smile / blink / turn your head). Randomised per arming so a recording
   can't pre-satisfy it. A printed photo or a static video cannot pass.
6. Drives an alarm **state machine** (idle → arming → armed → evaluating → challenge → alarm)
   and emits every transition as a webhook.

## Interface (the contract)

**In:**
- A camera **stream** (`FACE_RECON_STREAM_URL`).
- A small **control API** (an integration pushes these):
  - `POST /arm` / `POST /disarm` — master arm/disarm override (e.g. a physical switch).
  - `POST /arm/auto` — release the override; back to autonomous quiet-then-arm.
  - `POST /signal` — an external motion/PIR pulse (optional; presence is camera-driven by default).

**Out:**
- **Webhooks** — one JSON POST per event to every configured URL
  (`FACE_RECON_WEBHOOK_URLS`). Payload: `{"event": "...", "ts": ..., "challenge": "..."?}`.
  Events: `arming`, `arming_cancelled`, `motion`, `challenge_issued`, `greeted`, `alarm`,
  `disarmed`.
- **HTTP** — `GET /healthz`, `GET /status`, `GET /metrics` (Prometheus),
  `GET /debug/frame` and `GET /debug/stream` (the live frame annotated with the skeleton, the
  face box, and the identity label, for bring-up and demos).

A liveness failure falls **silently** into the alarm path; the system never reveals which check
failed.

## Architecture

```
  your camera                         RTX-class GPU box (Docker)
  +-------------+   MJPEG/RTSP        +-------------------------------------+
  |  stream     | ----------------->  |  Roboflow inference server (GPU)     |
  +-------------+                     |   person + pose models               |
                                      |             ^                        |
  integration  --- /arm /disarm --->  |  decision-service (FastAPI)          |
  (your glue)      /signal            |   stream -> detect -> pose ->        |
               <--- webhooks -------  |   geometry gate -> face match        |
                                      |   (insightface) -> liveness          |
                                      |   (MediaPipe) -> state machine       |
                                      |   -> webhooks + control + /debug     |
                                      |   Prometheus  <-- metrics --> Grafana|
                                      +-------------------------------------+
```

All inference and decision logic runs in the box. Frames never leave it (the inference server
runs models locally). The box is the win-side product; wiring its webhooks/control API to real
hardware (lights, sirens, sensors, notifications) is a separate integration you provide.

## Repository layout

```
src/face_recon/
  core/        config, logging, domain models, metrics
  pipeline/    pose, faces, liveness, challenges, state_machine, debug, orchestrator
  services/    boundaries: stream, inference (Roboflow), face_embed (insightface),
               landmarks (MediaPipe), webhook
  api/         FastAPI app: health, status, control API, metrics, debug stream
  enrolment/   build the reference embedding set
deploy/        docker compose, Dockerfiles (incl. a Blackwell/RTX-50 GPU build), Prometheus, Grafana
tests/         unit (pure logic) and integration (API)
```

## Enrolment (recognising *you*)

The box recognises whoever you enrol. You build a reference set once, on your side; it is never
committed.

1. **Capture** short clips/frames of the person **from your own camera**, at the distance and
   lighting they'll actually be recognised at (a mid distance where the face is ~90–180 px), a
   few angles and looks.
2. **Build** the reference set:
   ```bash
   python -m face_recon.enrolment.enrol \
       --captures /path/pass1 /path/pass2 ... \
       --out enrolment/reference/known.npz
   ```
   It keeps only mid-to-large, confident faces (small faces embed noisily; faces pressed to a
   wide lens are distorted), and reports the set's tightness.
3. **Point the box at it** with `FACE_RECON_REFERENCE_PATH` (default
   `enrolment/reference/known.npz`). Tune `FACE_RECON_FACE_MATCH_THRESHOLD` against live frames.

## Running

```bash
cp .env.example .env        # set RF_API_KEY, FACE_RECON_STREAM_URL, etc.
make up                     # docker compose: inference server, decision-service, Prometheus, Grafana
```

- decision-service: `http://localhost:8000` (`/healthz`, `/status`, `/metrics`, `/debug/stream`)
- inference server: `http://localhost:9001`
- Prometheus: `http://localhost:9090` · Grafana: `http://localhost:3000`

On Blackwell / RTX-50 GPUs the stock inference image fails (`no kernel image`); see
[`deploy/README.md`](deploy/README.md) for the cu128 build that fixes it.

## Development

```bash
make setup        # venv + light deps + dev tools
make test         # unit + integration tests
make lint         # ruff
```

The decision core (geometry, face matching, challenges, the state machine, config, the API) is
unit tested and runs without a GPU or camera. The CV service boundaries are lazy-imported, so
the package stays light to import and test.

## Non-goals

- No hardware control in the box: it emits webhooks; you wire what they do.
- No video recording / NVR.
- No custom model training (pre-trained person + pose models; identity needs only enrolment).

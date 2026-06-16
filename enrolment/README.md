# Enrolment

The box recognises whoever you enrol. Enrolment builds a **reference set**: a small bundle of
ArcFace face embeddings of the person, captured from your own camera. You do this once, on your
side. Reference frames and the built set are personal and are **never committed** (this folder's
`reference/` is gitignored).

## 1. Capture frames

Record the person **from the camera the box will actually watch**, so the reference matches the
real conditions:

- **Distance:** stand where you'll be recognised, at a distance where the **face is ~90–180 px**
  in the frame. Too far (tiny face) embeds noisily; pressed right against a wide-angle lens
  distorts the face. Mid distance is the sweet spot.
- **Lighting:** the real lighting of the spot. Keep light **on your face**, not a bright window
  directly behind you (backlight turns the face into a silhouette the detector misses).
- **Variety:** a few short passes covering the looks you want recognised (e.g. with/without
  glasses or a beard) and any lighting you expect. A handful of seconds each is plenty.

Drop the frames into folders, one per pass, anywhere on your machine (outside the repo is fine).

## 2. Build the reference set

```bash
python -m face_recon.enrolment.enrol \
    --captures /path/pass1 /path/pass2 ... \
    --out enrolment/reference/known.npz \
    [--stride 12] [--min-side 90] [--max-side 180]
```

It runs each sampled frame through the face model, keeps only **mid-to-large, confident** faces
(the `--min-side` / `--max-side` window), stacks their embeddings, and reports how tight the set
is (mean intra-similarity). A tight set well above stranger similarity (~0.3) is what you want.

## 3. Point the box at it

The box loads `FACE_RECON_REFERENCE_PATH` (default `enrolment/reference/known.npz`). In Docker,
mount the folder in (see `deploy/docker-compose.yml`). Then tune
`FACE_RECON_FACE_MATCH_THRESHOLD` against live frames using the `/debug/stream` view, which shows
the live match score.

## Adding another person

Run the same command into a separate file and load that instead; the matcher takes the best
match across the set. (v1 is built around one enrolled person.)

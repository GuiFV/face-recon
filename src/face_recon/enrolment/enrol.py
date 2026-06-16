"""Build the reference embedding set (ArcFace, via insightface) from enrolment frames.

Run against folders of frames captured from the camera, of the person to recognise, in
the real lighting and at the distance they will actually enter at. Each frame is run through
insightface and the embeddings of faces within a size window are stacked and saved. The
decision service loads this set and matches a live face against it by cosine similarity.

Why a size window and not "bigger is better": the reference set must look like the moment of
recognition. A face smaller than --min-side gives a noisy embedding; and on a wide-FOV (near
fish-eye) lens a face pressed right against the camera is barrel-distorted and forms its own
cluster, so --max-side drops those extreme close-ups. The result represents the mid distance a
person actually enters at, which is where we want recognition to work.

Adding a second person later is the same command into a separate output file.

Usage:
    python -m face_recon.enrolment.enrol \
        --captures /path/pass1 /path/pass2 ... \
        --out enrolment/reference/known.npz [--stride 12] [--min-side 90] [--max-side 180]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from face_recon.services.face_embed import FaceEmbedder

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_FACE_SIDE = 90.0  # px; faces smaller than this give noisy ArcFace embeddings
MAX_FACE_SIDE = 180.0  # px; larger faces on a wide-FOV lens are barrel-distorted close-ups
MIN_DET_SCORE = 0.6


def load_bgr(path: Path):
    """Load an image as a contiguous BGR numpy array (what insightface expects)."""
    from PIL import Image

    rgb = np.array(Image.open(path).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def list_frames(folders: list[Path], stride: int = 1) -> list[Path]:
    """Image files across the folders, sampled every `stride`-th, sorted per folder."""
    frames: list[Path] = []
    for folder in folders:
        fs = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        frames.extend(fs[::stride])
    return frames


def build_reference_set(
    frames: list[Path],
    embedder: FaceEmbedder,
    min_side: float = MIN_FACE_SIDE,
    max_side: float | None = MAX_FACE_SIDE,
    min_score: float = MIN_DET_SCORE,
    progress_every: int = 100,
) -> np.ndarray:
    """Embed the mid-to-large, confident faces across the frames into an (n, 512) array."""
    vecs: list[np.ndarray] = []
    for i, fr in enumerate(frames):
        if progress_every and i and i % progress_every == 0:
            print(f"  ...{i}/{len(frames)} scanned, {len(vecs)} kept", flush=True)
        face = embedder.best_face(load_bgr(fr))
        if not face or face.det_score < min_score or face.min_side < min_side:
            continue
        if max_side is not None and face.min_side >= max_side:
            continue
        vecs.append(face.embedding)
    return np.vstack(vecs) if vecs else np.empty((0, 512))


def save_reference_set(vectors: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, embeddings=vectors)


def load_reference_set(path: Path | str) -> list[list[float]]:
    """Load enrolled embeddings as a list of vectors. Empty list if the file is missing."""
    p = Path(path)
    if not p.exists():
        return []
    with np.load(p) as data:
        return [list(map(float, v)) for v in data["embeddings"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reference embedding set from frames.")
    parser.add_argument("--captures", required=True, nargs="+", type=Path, help="frame folders")
    parser.add_argument("--out", required=True, type=Path, help="output .npz path")
    parser.add_argument("--stride", type=int, default=12, help="sample every Nth frame")
    parser.add_argument("--min-side", type=float, default=MIN_FACE_SIDE, help="min face px")
    parser.add_argument("--max-side", type=float, default=MAX_FACE_SIDE,
                        help="max face px (drop distorted close-ups; 0 disables the cap)")
    parser.add_argument("--min-score", type=float, default=MIN_DET_SCORE, help="min det score")
    args = parser.parse_args()

    frames = list_frames(args.captures, stride=args.stride)
    print(f"scanning {len(frames)} sampled frames from {len(args.captures)} folder(s)...")
    embedder = FaceEmbedder(use_gpu=False)
    max_side = args.max_side if args.max_side and args.max_side > 0 else None
    vectors = build_reference_set(frames, embedder, args.min_side, max_side, args.min_score)
    if len(vectors) == 0:
        raise SystemExit("no faces met the size/score filter; lower --min-side or --stride")
    save_reference_set(vectors, args.out)

    # Report the set's tightness (mean pairwise cosine of the normalised embeddings).
    sims = vectors @ vectors.T
    n = len(vectors)
    off = (sims.sum() - n) / (n * n - n) if n > 1 else 1.0
    print(f"enrolled {n} reference embeddings to {args.out} (mean intra-cosine {off:.3f})")


if __name__ == "__main__":
    main()

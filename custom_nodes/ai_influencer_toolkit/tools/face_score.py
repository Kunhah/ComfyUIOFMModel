#!/usr/bin/env python3
"""Face-identity scoring with OpenCV's YuNet detector + SFace recognizer (CPU, ~40 MB of
models, downloaded to ../models/ on first use). Used by eval_lora.py; also runnable on a dataset folder to find images
whose face drifts from the rest before you pay for training.

SFace cosine similarity: OpenCV documents >= 0.363 as "same person". For a character dataset,
each image vs. the average of the others is typically well above that; the lowest scorers are the
ones most likely to teach the LoRA a different face.

Usage:
    python face_score.py input/ai_influencer/MyCharacter/dataset
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402
import numpy as np

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DETECTOR = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
RECOGNIZER = os.path.join(MODELS, "face_recognition_sface_2021dec.onnx")
# Not shipped in the repo: fetched from the OpenCV model zoo on first use and checked by hash.
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODEL_SOURCES = {
    DETECTOR: (ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
               "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    RECOGNIZER: (ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
                 "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}
SAME_PERSON = 0.363
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def ensure_models() -> None:
    import hashlib
    import urllib.request

    os.makedirs(MODELS, exist_ok=True)
    for path, (url, sha256) in MODEL_SOURCES.items():
        if os.path.isfile(path):
            continue
        print(f"downloading {os.path.basename(path)} (OpenCV model zoo) ...", file=sys.stderr)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        with open(tmp, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != sha256:
                os.remove(tmp)
                raise SystemExit(f"{os.path.basename(path)}: checksum mismatch, download discarded ({url})")
        os.replace(tmp, path)


class FaceScorer:
    def __init__(self):
        ensure_models()
        self.detector = cv2.FaceDetectorYN.create(DETECTOR, "", (320, 320), score_threshold=0.8)
        self.recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER, "")

    def embed_bgr(self, img: np.ndarray) -> np.ndarray | None:
        """Unit-length embedding of the largest face, or None if no face is found."""
        h, w = img.shape[:2]
        scale = 1024 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        feat = self.recognizer.feature(self.recognizer.alignCrop(img, face)).flatten()
        return feat / np.linalg.norm(feat)

    def embed_file(self, path: str) -> np.ndarray | None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return None if img is None else self.embed_bgr(img)


def list_images(folder: str) -> list[str]:
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)


def reference_embeddings(scorer: FaceScorer, folder: str) -> tuple[np.ndarray, list[str]]:
    names, embs = [], []
    for p in list_images(folder):
        e = scorer.embed_file(p)
        if e is not None:
            names.append(os.path.basename(p))
            embs.append(e)
    return np.stack(embs), names


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(sys.argv) == 2 else 1
    if not os.path.isdir(sys.argv[1]):
        print(f"Not a folder: {sys.argv[1]}", file=sys.stderr)
        return 1
    scorer = FaceScorer()
    paths = list_images(sys.argv[1])
    rows = []
    embs = {}
    for p in paths:
        e = scorer.embed_file(p)
        if e is None:
            rows.append((os.path.basename(p), None))
        else:
            embs[os.path.basename(p)] = e
    names = list(embs)
    matrix = np.stack([embs[n] for n in names])
    for i, n in enumerate(names):
        others = np.delete(matrix, i, axis=0).mean(0)
        rows.append((n, float(matrix[i] @ (others / np.linalg.norm(others)))))
    scored = [s for _, s in rows if s is not None]
    print(f"{len(scored)}/{len(paths)} faces found. vs-rest similarity: "
          f"mean {np.mean(scored):.3f}, min {np.min(scored):.3f} (same-person threshold {SAME_PERSON})")
    for n, s in sorted(rows, key=lambda r: -1 if r[1] is None else r[1]):
        flag = "NO FACE" if s is None else ("  <- outlier" if s < np.mean(scored) - 2 * np.std(scored) else "")
        print(f"  {'-' if s is None else f'{s:.3f}':>6}  {n}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

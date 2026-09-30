#!/usr/bin/env python3
"""Run the camera-imperfections pass over images that already exist.

Same math as the *Camera Imperfections* node (nodes_camera.py), for everything generated before
that node was wired in, or for pictures that came out of another tool: ISO grain, red/blue
fringing towards the corners, corner falloff and a per-image white balance drift. Each image gets
its own seed derived from the file name, so a folder doesn't come out sharing one grain pattern.

Usage:
    python camera_pass.py output/projects/lina/outputs/images/final
    python camera_pass.py shot.png --out regraded/ --iso-grain 0.04
    python camera_pass.py old_batch/ --in-place --jpeg-quality 92

Writes `<name>_camera.<ext>` next to each input unless --out or --in-place is given.
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from custom_nodes.ai_influencer_toolkit.nodes_camera import apply_camera_imperfections  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect(paths: list[str]) -> list[str]:
    files = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                stem, ext = os.path.splitext(name)
                if ext.lower() in IMAGE_EXTS and not stem.endswith("_camera"):
                    files.append(os.path.join(path, name))
        else:
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Image files and/or folders of images.")
    parser.add_argument("--out", default="", help="Write into this folder instead of next to the inputs.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input files.")
    parser.add_argument("--seed", type=int, default=-1, help="Fixed seed for every image. Default: derived from the file name.")
    parser.add_argument("--iso-grain", type=float, default=0.025, help="0.015 clean daylight, 0.025 phone photo (default), 0.05+ pushed high ISO.")
    parser.add_argument("--grain-size", type=float, default=0.1, help="Grain clump size, %% of the short side.")
    parser.add_argument("--chroma-noise", type=float, default=0.35, help="Color speckle, relative to --iso-grain.")
    parser.add_argument("--chromatic-aberration", type=float, default=0.05, help="Corner fringing, %% of the short side.")
    parser.add_argument("--vignette", type=float, default=0.15, help="Corner darkening.")
    parser.add_argument("--color-jitter", type=float, default=0.3, help="Per-image white balance and exposure drift.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Quality for JPEG output.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.out and args.in_place:
        parser.error("--out and --in-place are mutually exclusive")
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    files = collect(args.paths)
    if not files:
        print("no images found", file=sys.stderr)
        return 1

    for path in files:
        image = Image.open(path).convert("RGB")
        tensor = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(args.device)
        seed = args.seed if args.seed >= 0 else zlib.crc32(os.path.basename(path).encode())
        out = apply_camera_imperfections(
            tensor, seed, args.iso_grain, args.grain_size, args.chroma_noise,
            args.chromatic_aberration, args.vignette, args.color_jitter,
        )
        result = Image.fromarray((out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))

        name, ext = os.path.splitext(os.path.basename(path))
        if args.in_place:
            dest = path
        elif args.out:
            dest = os.path.join(args.out, name + ext)
        else:
            dest = os.path.join(os.path.dirname(path), f"{name}_camera{ext}")
        if ext.lower() in (".jpg", ".jpeg"):
            result.save(dest, quality=args.jpeg_quality)
        else:
            result.save(dest)
        print(dest)

    print(f"{len(files)} image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

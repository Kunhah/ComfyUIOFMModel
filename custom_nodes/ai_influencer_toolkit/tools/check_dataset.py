#!/usr/bin/env python3
"""Check a character-LoRA dataset before you pay for GPU time.

A LoRA is only as good as its dataset. For a Krea 2 character LoRA, 26-34 sharp images of the
same face (different angles, expressions, lighting, and a mix of clean studio shots and casual
phone-style shots) is the target. This script checks what a script can check:

  errors   (exit code 1): fewer than 10 images, unreadable images, images without a caption file,
           empty captions, exact duplicate images
  warnings (exit code 0): count outside 26-34, short side under --min-side, near-duplicate images,
           captions missing the trigger word, caption files without an image,
           file types AI Toolkit may not read

It cannot judge whether it's the same face in every image, or whether angles and expressions vary
enough. Check that yourself.

Usage:
    python check_dataset.py path/to/lina_dataset --trigger l1na
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SAFE_EXTS = {".jpg", ".jpeg", ".png"}
SWEET_SPOT = (26, 34)


def average_hash(img: Image.Image, size: int = 12) -> int:
    small = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = small.tobytes()
    mean = sum(pixels) / len(pixels)
    return sum(1 << i for i, p in enumerate(pixels) if p > mean)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Folder with images and same-named .txt captions.")
    parser.add_argument("--trigger", default="", help="Trigger word that should appear in every caption (e.g. l1na).")
    parser.add_argument("--min-side", type=int, default=1024, help="Warn when an image's short side is smaller than this.")
    parser.add_argument("--caption-ext", default="txt")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Not a folder: {args.folder}")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    files = sorted(os.listdir(args.folder))
    images = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    captions = {os.path.splitext(f)[0] for f in files if f.lower().endswith("." + args.caption_ext)}
    stems = {os.path.splitext(f)[0] for f in images}

    by_hash: dict[str, list[str]] = defaultdict(list)
    sha_of: dict[str, str] = {}
    ahashes: list[tuple[str, int]] = []
    for name in images:
        path = os.path.join(args.folder, name)
        stem, ext = os.path.splitext(name)
        if ext.lower() not in SAFE_EXTS:
            warnings.append(f"{name}: {ext} may not be supported by your AI Toolkit version; convert to .jpg/.png to be safe")
        with open(path, "rb") as f:
            sha_of[name] = hashlib.sha1(f.read()).hexdigest()
        by_hash[sha_of[name]].append(name)
        try:
            with Image.open(path) as img:
                img.load()
                w, h = img.size
                ahashes.append((name, average_hash(img)))
        except (UnidentifiedImageError, OSError) as e:
            errors.append(f"{name}: can't be read ({e})")
            continue
        if min(w, h) < args.min_side:
            warnings.append(f"{name}: {w}x{h}, short side under {args.min_side}px (fine if it's an intentional phone-style shot)")

        if stem not in captions:
            errors.append(f"{name}: no {stem}.{args.caption_ext} caption")
            continue
        with open(os.path.join(args.folder, f"{stem}.{args.caption_ext}"), encoding="utf-8") as f:
            caption = f.read().strip()
        if not caption:
            errors.append(f"{stem}.{args.caption_ext}: empty caption")
        elif args.trigger and args.trigger.lower() not in caption.lower() and "[trigger]" not in caption:
            warnings.append(f"{stem}.{args.caption_ext}: no '{args.trigger}' (AI Toolkit adds the trigger word if it's set in the job, but it's clearer to write it)")

    for stem in sorted(captions - stems):
        warnings.append(f"{stem}.{args.caption_ext}: caption without an image")
    for names in by_hash.values():
        if len(names) > 1:
            errors.append(f"exact duplicates: {', '.join(names)}")
    for i, (a, ha) in enumerate(ahashes):
        for b, hb in ahashes[i + 1:]:
            if bin(ha ^ hb).count("1") <= 6 and sha_of[a] != sha_of[b]:
                warnings.append(f"near-duplicates: {a} ~ {b} (similar shots teach less than varied ones)")

    n = len(images)
    if n < 10:
        errors.append(f"only {n} images; a character LoRA needs far more (aim for {SWEET_SPOT[0]}-{SWEET_SPOT[1]})")
    elif not SWEET_SPOT[0] <= n <= SWEET_SPOT[1]:
        warnings.append(f"{n} images; {SWEET_SPOT[0]}-{SWEET_SPOT[1]} strong images is the target, quality beats quantity")

    print(f"{n} images, {len(captions)} captions in {args.folder}")
    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")
    print("OK" if not errors else f"{len(errors)} error(s): fix before training")
    print("Not checkable here: same face in every image, varied angles/expressions/lighting.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

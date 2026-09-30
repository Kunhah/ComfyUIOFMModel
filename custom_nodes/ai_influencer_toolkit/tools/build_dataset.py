#!/usr/bin/env python3
"""Build a clean LoRA training folder from a curation manifest (datasets/<name>.json): copies
each selected image at native resolution as NN.<ext> with a same-stem NN.txt caption, and
writes a numbered contact sheet so you can eyeball the final set before paying for training.

The same folder works for fal.ai's Krea 2 trainer (train_lora.py zips it) and for AI Toolkit
(upload the folder as a dataset). It's also what Workflow 07 scores checkpoints against.

Usage:
    python build_dataset.py ../datasets/MyCharacter_v1.json
    python build_dataset.py ../datasets/MyCharacter_v1.json --out /some/other/folder
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def contact_sheet(paths: list[str], out: str, tile: int = 300, cols: int = 6) -> None:
    rows = -(-len(paths) // cols)
    sheet = Image.new("RGB", (cols * tile, rows * tile), "white")
    font = ImageFont.load_default(size=26)
    for k, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((tile, tile))
            x, y = (k % cols) * tile, (k // cols) * tile
            sheet.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
        ImageDraw.Draw(sheet).text((x + 4, y + 2), os.path.splitext(os.path.basename(p))[0],
                                   fill="red", font=font, stroke_width=3, stroke_fill="white")
    sheet.save(out, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--out", help="Target folder (default: input/ai_influencer/<project>/dataset).")
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        m = json.load(f)
    src_dir = os.path.join(REPO_ROOT, m["source_dir"])
    out = args.out or os.path.join(REPO_ROOT, "input", "ai_influencer", m["project"], "dataset")

    missing = [i["src"] for i in m["images"] if not os.path.isfile(os.path.join(src_dir, i["src"]))]
    if missing:
        print("Missing source images:\n  " + "\n  ".join(missing))
        return 1
    no_trigger = [i["src"] for i in m["images"] if m["trigger"] not in i["caption"]]
    if no_trigger:
        print("Captions without the trigger word:\n  " + "\n  ".join(no_trigger))
        return 1

    if os.path.isdir(out) and os.listdir(out):
        print(f"{out} is not empty; move it aside first so nothing is mixed in.")
        return 1
    os.makedirs(out, exist_ok=True)

    written = []
    for n, item in enumerate(m["images"], start=1):
        ext = os.path.splitext(item["src"])[1].lower()
        dst = os.path.join(out, f"{n:02d}{ext}")
        shutil.copy2(os.path.join(src_dir, item["src"]), dst)
        with open(os.path.join(out, f"{n:02d}.txt"), "w", encoding="utf-8") as f:
            f.write(item["caption"].strip() + "\n")
        written.append(dst)

    sheet = os.path.join(os.path.dirname(out), f"{os.path.basename(out)}_contact_sheet.jpg")
    contact_sheet(written, sheet)
    print(f"{len(written)} images + captions -> {out}\ncontact sheet -> {sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

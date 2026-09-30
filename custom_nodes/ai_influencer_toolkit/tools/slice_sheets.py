#!/usr/bin/env python3
"""Cut character sheets you already have into one image per view, with a caption each -- the same
code as Workflow 01's *Slice Character Sheet* node (sheet_slicing.py), without ComfyUI.

Workflow 01 saves each sheet under outputs/images/sheets/<sheet name>/, so the template (layout and
captions) is taken from the folder name; pass --sheet for files kept anywhere else.

Usage:
    python slice_sheets.py ../../../output/projects/lina/outputs/images/sheets/pose-grid
    python slice_sheets.py some_sheet.png --sheet lighting-grid --trigger l1na --out /tmp/crops

Each sheet becomes a folder <out>/<sheet>_<file stem>/ with 00.png (the large view), 01.png, ...
and a same-named .txt caption: "<trigger>, <that panel's wording from the template>, plain grey
studio background". Look at the crops before copying them into the dataset: the captions assume GPT
kept the template's panel order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheet_slicing import DEFAULT_DIR, list_sheets, load_template, save_slices, slice_sheet  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Sheet images or folders of them.")
    ap.add_argument("--sheet", choices=list_sheets(), help="Template that made them (default: the folder's name).")
    ap.add_argument("--template", help="Path to the template JSON that made them (e.g. a custom one), instead of --sheet.")
    ap.add_argument("--project", default="", help="Reads the trigger word from output/projects/<project>/character.json.")
    ap.add_argument("--trigger", default="", help="Trigger word for the captions (overrides --project). Default: [trigger].")
    ap.add_argument("--out", help="Where the crop folders go (default: <project or sheet's project>/outputs/images/sheet_crops).")
    ap.add_argument("--templates-dir", default=DEFAULT_DIR)
    ap.add_argument("--caption-prefix", default="", help="Start of every caption (default: the trigger word).")
    ap.add_argument("--caption-suffix", default="plain grey studio background", help="End of every caption ('' for none).")
    ap.add_argument("--threshold", type=float, default=0.09, help="How different from the background counts as the person.")
    ap.add_argument("--padding", type=float, default=0.05, help="Extra room around each figure, fraction of its size.")
    args = ap.parse_args()

    trigger = args.trigger
    if not trigger and args.project:
        path = os.path.join(REPO_ROOT, "output", "projects", args.project, "character.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                trigger = json.load(f).get("trigger", "")

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith(EXTS)]
        else:
            files.append(p)
    if not files:
        print("No sheet images found.")
        return 1

    for path in files:
        if args.template:
            with open(args.template, encoding="utf-8") as f:
                template = json.load(f)
            sheet = os.path.splitext(os.path.basename(args.template))[0]
        else:
            sheet = args.sheet or os.path.basename(os.path.dirname(os.path.abspath(path)))
            if sheet not in list_sheets(args.templates_dir):
                print(f"{path}: can't tell which sheet this is (folder '{sheet}'); pass --sheet or --template")
                return 1
            template = load_template(sheet, args.templates_dir)
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        slices = slice_sheet(img, template, trigger, args.threshold, args.padding, args.caption_prefix, args.caption_suffix)
        if args.out:
            out_root = args.out
        else:  # .../outputs/images/sheets/<sheet>/x.png -> .../outputs/images/sheet_crops
            out_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "sheet_crops")
        out = os.path.join(out_root, f"{sheet}_{os.path.splitext(os.path.basename(path))[0]}")
        save_slices(slices, out)
        fallback = sum(1 for *_, found in slices if not found)
        print(f"{path}: {len(slices)} crops -> {out}" + (f" ({fallback} by grid position only)" if fallback else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

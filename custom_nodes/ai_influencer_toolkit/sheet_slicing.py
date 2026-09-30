"""Pure slicing/template code shared by nodes_sheet.py and tools/slice_sheets.py (no ComfyUI
imports, so the tool runs without ComfyUI). See nodes_sheet.py for what it does and why."""
from __future__ import annotations

import json
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DIR = "character-sheet-studio/.claude/skills/character-sheet-studio/templates"
OWN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sheet_templates")
# Your own templates: any <name>.json here shows up in every sheet dropdown (press R to refresh),
# and a name that exists elsewhere replaces that one. It sits in ComfyUI's user/ folder, so node
# pack updates never touch it. The Custom Character Sheet node can save into it.
USER_DIR = os.path.join(REPO_ROOT, "user", "ai_influencer", "sheet_templates")
STUDIO_SHEETS = ["face-turnaround", "expression-grid", "full-body-360", "upper-body"]

DEFAULT_LOCKS = [
    "Soft grey seamless studio background, the same grey behind every view.",
    "Flat even softbox lighting, no hard shadows.",
    "Eye level, identical framing and scale.",
    "Keep the face and identity exactly as in the reference images.",
    "No text, no labels, no watermark.",
]
DEFAULT_AVOID = [
    "No borders and no drop shadows between the panels.",
    "Leave a clear strip of empty grey background between neighbouring panels; no figure overlaps another panel.",
    "Do not crop the person at a panel edge.",
]


def _studio_dir(templates_dir: str) -> str:
    return templates_dir if os.path.isabs(templates_dir) else os.path.join(REPO_ROOT, templates_dir)


def _names(folder: str) -> list[str]:
    try:
        return sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".json"))
    except OSError:
        return []


def list_sheets(templates_dir: str = DEFAULT_DIR) -> list[str]:
    """The studio's four first, then this pack's, then yours -- each name once."""
    seen, out = set(), []
    for name in STUDIO_SHEETS + _names(OWN_DIR) + _names(USER_DIR) + _names(_studio_dir(templates_dir)):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def template_path(sheet: str, templates_dir: str = DEFAULT_DIR) -> str:
    for folder in (USER_DIR, OWN_DIR, _studio_dir(templates_dir)):  # yours wins
        path = os.path.join(folder, f"{sheet}.json")
        if os.path.isfile(path):
            return path
    return os.path.join(_studio_dir(templates_dir), f"{sheet}.json")


def load_template(sheet: str, templates_dir: str = DEFAULT_DIR) -> dict:
    path = template_path(sheet, templates_dir)
    if not os.path.isfile(path):
        hint = " (unzip Character Sheet Studio into the ComfyUI folder)" if sheet in STUDIO_SHEETS else ""
        raise FileNotFoundError(f"Character sheet template not found: {path}{hint}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def grid_shape(n: int) -> tuple[int, int]:
    """rows x cols for n grid panels in the right half (roughly square: 16:9 halved is ~8:9)."""
    cols = max(1, math.ceil(math.sqrt(n)))
    return math.ceil(n / cols), cols


def build_template(title: str, hero: str, panels: list[str], locks: list[str] | None = None,
                   consistency: str = "", avoid: list[str] | None = None) -> dict:
    """A template in the studio's format: one large view in the left half, the panels in a grid in the right half."""
    panels = [p.strip() for p in panels if p.strip()]
    if not panels:
        raise ValueError("A sheet needs at least one panel (one per line)")
    rows, cols = grid_shape(len(panels))
    locks = [x.strip() for x in (DEFAULT_LOCKS if locks is None else locks) if x.strip()]
    return {
        "task": f"{title.strip() or 'Character Sheet'}, one single canvas, 16:9",
        "identity": {"source": "the attached reference images only",
                     "rule": "The person is the person in the reference images. Only the views described below change."},
        "layout": {"hero": {"position": "left half", "content": hero.strip()},
                   "grid": {"position": "right half", "rows": rows, "cols": cols,
                            "reading_order": "left to right, top to bottom",
                            "cells": [{"view": p} for p in panels]}},
        "locks": {f"lock_{i + 1}": x for i, x in enumerate(locks)},
        "consistency": consistency.strip() or (
            "Every panel shows the same person from the reference images at the same scale. Only what the panel "
            f"descriptions say changes, and no two panels repeat.{' Grid cells after the last description stay empty grey.' if rows * cols > len(panels) else ''}"),
        "must_avoid": [x.strip() for x in (DEFAULT_AVOID if avoid is None else avoid) if x.strip()],
    }


def panel_texts(template: dict) -> list[str]:
    """The hero's wording, then each grid cell's, in the template's reading order."""
    layout = template["layout"]
    return [layout["hero"]["content"], *(c["view"] for c in layout["grid"]["cells"])]


def _cells(w: int, h: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1): the left-half hero, then the right-half grid left to right, top to bottom."""
    half = w // 2
    out = [(0, 0, half, h)]
    for r in range(rows):
        for c in range(cols):
            out.append((half + c * (w - half) // cols, r * h // rows, half + (c + 1) * (w - half) // cols, (r + 1) * h // rows))
    return out


def slice_boxes(img: np.ndarray, rows: int, cols: int, threshold: float = 0.09, pad: float = 0.05) -> list[tuple[tuple[int, int, int, int], bool]]:
    """img: HxWx3 floats 0..1. Returns one (box, found) per panel; found=False means the cell fallback."""
    h, w = img.shape[:2]
    cells = _cells(w, h, rows, cols)
    fg = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in cells:
        cell = img[y0:y1, x0:x1]
        ring = max(2, int(0.03 * min(x1 - x0, y1 - y0)))
        border = np.concatenate([cell[:ring].reshape(-1, 3), cell[-ring:].reshape(-1, 3),
                                 cell[:, :ring].reshape(-1, 3), cell[:, -ring:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        fg[y0:y1, x0:x1] = np.abs(cell - bg).max(axis=2) > threshold
    # clean up: close small gaps (hair strands, a thin arm), drop specks
    step = max(1, min(h, w) // 540)
    small = fg[::step, ::step]
    small = ndimage.binary_closing(small, iterations=2)
    small = ndimage.binary_opening(small, iterations=1)
    labels, n = ndimage.label(small)
    boxes: list[list[int] | None] = [None] * len(cells)
    min_area = 0.002 * small.size
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        area = int((labels[sl] == i).sum())
        if area < min_area:
            continue
        ys, xs = sl
        cy = (ys.start + ys.stop) / 2 * step
        cx = (xs.start + xs.stop) / 2 * step
        k = next((j for j, (x0, y0, x1, y1) in enumerate(cells) if x0 <= cx < x1 and y0 <= cy < y1), None)
        if k is None:
            continue
        b = [xs.start * step, ys.start * step, xs.stop * step, ys.stop * step]
        boxes[k] = b if boxes[k] is None else [min(boxes[k][0], b[0]), min(boxes[k][1], b[1]), max(boxes[k][2], b[2]), max(boxes[k][3], b[3])]
    out = []
    for cell, box in zip(cells, boxes):
        x0, y0, x1, y1 = cell
        if box is None:
            out.append((cell, False))
            continue
        # a shape may reach over the cell line, but not deep into the neighbour (merged figures)
        mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        bx0, by0 = max(box[0], x0 - mx), max(box[1], y0 - my)
        bx1, by1 = min(box[2], x1 + mx), min(box[3], y1 + my)
        px, py = pad * (bx1 - bx0), pad * (by1 - by0)
        out.append(((int(max(0, bx0 - px)), int(max(0, by0 - py)), int(min(w, bx1 + px)), int(min(h, by1 + py))), True))
    return out


def slice_sheet(img: np.ndarray, template: dict, trigger: str, threshold: float = 0.09, pad: float = 0.05,
                caption_prefix: str = "", caption_suffix: str = "plain grey studio background") -> list[tuple[np.ndarray, str, bool]]:
    """(crop, caption, found) per panel, hero first. Caption = prefix (default: the trigger word, or
    [trigger]), the panel's wording, suffix -- empty parts are left out."""
    grid = template["layout"]["grid"]
    texts = panel_texts(template)
    boxes = slice_boxes(img, int(grid["rows"]), int(grid["cols"]), threshold, pad)
    if len(boxes) < len(texts):
        raise ValueError(f"template has {len(texts)} panels but its layout only has room for {len(boxes)}")
    head = caption_prefix.strip() or trigger.strip() or "[trigger]"
    out = []
    for ((x0, y0, x1, y1), found), text in zip(boxes, texts):  # empty trailing grid cells are skipped
        caption = ", ".join(p for p in (head, text.strip(), caption_suffix.strip()) if p)
        out.append((img[y0:y1, x0:x1], caption, found))
    return out


def save_slices(slices, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for i, (crop, caption, _found) in enumerate(slices):
        stem = f"{i:02d}"
        Image.fromarray((np.clip(crop, 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(out_dir, stem + ".png"))
        with open(os.path.join(out_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(caption + "\n")
        names.append(stem + ".png")
    return names

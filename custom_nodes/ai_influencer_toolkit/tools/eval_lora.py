#!/usr/bin/env python3
"""Compare trained Krea 2 LoRAs side by side on fal.ai (fal-ai/krea-2/turbo/lora, $0.01/megapixel).

Renders a fixed prompt set with the SAME seed in every column, so the only difference between
cells is the LoRA (and scale). Columns: an optional no-LoRA baseline, then each run x scale.
Rows: prompts with outfits, places, poses and hairstyles that are NOT in the dataset, plus a
last "leak" row that describes a different woman WITHOUT the trigger word.

Each image is face-scored locally (face_score.py, OpenCV SFace) against the dataset:
  identity  mean similarity to the dataset face centroid over the trigger rows. Higher = more her.
            The baseline column shows what "not her" scores, so the metric is checked every run.
  leak      the same score on the no-trigger row. It should stay near baseline; if it climbs,
            the LoRA pushes her face into every image (overtrained / too strong).
  faces     how many trigger rows produced a detectable face.
Scores are a tie-breaker. Look at the grid: the winner keeps her face in every row AND still
follows each prompt (outfit, hair, place). Past the sweet spot, runs start copying the dataset
(gray studio, pink cardigan) even when the prompt asks for something else.

Rendered cells are cached by (lora, scale, prompt, seed, size), so re-running never pays twice.

Usage:
    python eval_lora.py MyCharacter --runs mychr1_s600_lr5e-04_r1024,mychr1_s1000_lr5e-04_r1024 --dry-run
    python eval_lora.py MyCharacter --runs <tag>,<tag> --scales 0.8,1.0 --seeds 7,21
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import env_config  # noqa: E402
import logging_util  # noqa: E402
import pricing  # noqa: E402
from face_score import FaceScorer, reference_embeddings  # noqa: E402
from train_lora import load_runs  # noqa: E402

ENDPOINT = "fal-ai/krea-2/turbo/lora"

TRIGGER_PROMPTS = [
    "close-up portrait photo of {t}, a young woman, looking at the camera, soft window light, natural skin texture",
    "candid smartphone photo of {t}, a young woman, laughing at a cafe table, black leather jacket, afternoon light",
    "full body photo of {t}, a young woman, walking down a rainy Tokyo street at night, red raincoat, neon reflections",
    "side profile photo of {t}, a young woman, hiking on a mountain trail, green windbreaker, overcast daylight",
    "mirror selfie of {t}, a young woman, hair in a high ponytail, gray hoodie, at a gym, fluorescent light",
    "photo of {t}, a young woman, sitting on a beach at golden hour, white linen dress, wind in her hair",
    "over-the-shoulder photo of {t}, a young woman, in a library, green sweater, round reading glasses",
]
LEAK_PROMPT = "photo of a young woman with short blonde hair and blue eyes, smiling at the camera, modern office background"


def cell_key(path: str | None, scale: float, prompt: str, seed: int, w: int, h: int) -> str:
    return hashlib.sha1(json.dumps([path, scale, prompt, seed, w, h]).encode()).hexdigest()[:16]


def render(fal_client, cache_dir: str, lora_url: str | None, scale: float, prompt: str, seed: int, w: int, h: int) -> tuple[str, bool]:
    key = cell_key(lora_url, scale, prompt, seed, w, h)
    path = os.path.join(cache_dir, f"{key}.jpg")
    if os.path.isfile(path):
        return path, False
    result = fal_client.subscribe(ENDPOINT, arguments={
        "prompt": prompt,
        "loras": [{"path": lora_url, "scale": scale}] if lora_url else [],
        "image_size": {"width": w, "height": h},
        "seed": seed,
        "num_images": 1,
        "output_format": "jpeg",
    })
    urllib.request.urlretrieve(result["images"][0]["url"], path)
    if (result.get("has_nsfw_concepts") or [False])[0]:
        print(f"  note: safety checker flagged a cell ({prompt[:40]}...); it may be blank")
    return path, True


def build_grid(cells: list[list[str]], col_labels: list[str], row_labels: list[str], out: str, tile: int = 320) -> None:
    font = ImageFont.load_default(size=18)
    head, side = 170, 260
    th = int(tile * 1.5)
    grid = Image.new("RGB", (side + tile * len(col_labels), head + th * len(row_labels)), "white")
    d = ImageDraw.Draw(grid)
    for c, label in enumerate(col_labels):
        d.multiline_text((side + c * tile + 6, 6), label, fill="black", font=font)
    for r, label in enumerate(row_labels):
        d.multiline_text((6, head + r * th + 6), label, fill="black", font=font)
        for c, p in enumerate(cells[r]):
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((tile, th))
                grid.paste(im, (side + c * tile, head + r * th))
    grid.save(out, quality=90)


def wrap(text: str, width: int = 24) -> str:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return "\n".join(lines + [line])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project")
    parser.add_argument("--runs", required=True, help="Comma-separated run tags from train_lora.py --list.")
    parser.add_argument("--scales", default="1.0", help="Comma-separated LoRA scales to test per run, e.g. 0.8,1.0.")
    parser.add_argument("--seeds", default="7", help="Comma-separated seeds; each prompt is rendered once per seed.")
    parser.add_argument("--size", default="1024x1536", help="WxH. Judge at >=1.5K: fine detail is invisible at 1024.")
    parser.add_argument("--no-baseline", action="store_true", help="Skip the no-LoRA control column.")
    parser.add_argument("--dataset", help="Reference faces (default: input/ai_influencer/<project>/dataset).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--budget", type=float, default=float(env_config.get("FAL_BUDGET_USD") or 0),
                         help="Refuse to run if logged fal.ai spend + this eval would exceed this many dollars "
                              "(default: FAL_BUDGET_USD from .env, 0 = no guard).")
    args = parser.parse_args()

    project = logging_util.sanitize_project_name(args.project)
    input_dir, output_dir = logging_util.ensure_project(project)
    runs = load_runs(project)
    tags = [t.strip() for t in args.runs.split(",") if t.strip()]
    unknown = [t for t in tags if t not in runs]
    if unknown:
        print(f"Unknown run(s): {unknown}. Known: {list(runs) or 'none'}")
        return 1
    scales = [float(s) for s in args.scales.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    w, h = (int(x) for x in args.size.lower().split("x"))
    trigger = runs[tags[0]]["trigger"]

    columns = [] if args.no_baseline else [{"label": "baseline\n(no LoRA)", "url": None, "scale": 0.0, "tag": None}]
    for t in tags:
        for s in scales:
            columns.append({"label": f"{runs[t]['steps']} steps\nlr {runs[t]['learning_rate']:g}\nscale {s:g}", "url": runs[t]["lora_url"], "scale": s, "tag": t})
    prompts = [p.format(t=trigger) for p in TRIGGER_PROMPTS] + [LEAK_PROMPT]
    rows = [(p, seed) for p in prompts for seed in seeds]

    cache_dir = os.path.join(output_dir, "lora_evals", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    jobs = [(r, c) for r in range(len(rows)) for c in range(len(columns))]
    new_jobs = [(r, c) for r, c in jobs if not os.path.isfile(os.path.join(
        cache_dir, cell_key(columns[c]["url"], columns[c]["scale"], rows[r][0], rows[r][1], w, h) + ".jpg"))]
    cost = pricing.estimate_fal_krea2_image_cost_usd(w, h, len(new_jobs))
    print(f"{len(rows)} rows x {len(columns)} columns = {len(jobs)} cells, {len(new_jobs)} new "
          f"(~${cost:.2f}; up to ${len(new_jobs) * 0.01 * -(-w * h // 1_000_000):.2f} if fal rounds up per megapixel)")
    spent = logging_util.provider_spend_to_date("fal_ai")
    if args.budget:
        print(f"Budget: ${spent:.2f} logged so far + ${cost:.2f} = ${spent + cost:.2f} of ${args.budget:.2f}")
    if args.dry_run:
        return 0
    logging_util.check_budget("fal_ai", cost, args.budget)

    if new_jobs:
        env_config.require_cli("FAL_KEY")
        if not args.yes and input(f"Spend ~${cost:.2f}? Type yes: ").strip().lower() != "yes":
            print("Cancelled, nothing spent.")
            return 1
    import fal_client

    cells: list[list[str | None]] = [[None] * len(columns) for _ in rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(render, fal_client, cache_dir, columns[c]["url"], columns[c]["scale"], rows[r][0], rows[r][1], w, h): (r, c) for r, c in jobs}
        for fut in concurrent.futures.as_completed(futs):
            r, c = futs[fut]
            path, fresh = fut.result()
            cells[r][c] = path
            if fresh:
                logging_util.append_generation_log({
                    "provider": "fal_ai", "model": ENDPOINT, "operation": "lora_eval", "project": project,
                    "stage": "lora_eval", "resolution_or_aspect": f"{w}x{h}", "num_outputs": 1,
                    "seed": rows[r][1], "estimated_cost_usd": pricing.estimate_fal_krea2_image_cost_usd(w, h),
                    "actual_cost_usd": None, "prompt_id": logging_util.make_prompt_id(rows[r][0]),
                    "output_path": path, "notes": columns[c]["tag"] or "baseline",
                })

    scorer = FaceScorer()
    refs, _ = reference_embeddings(scorer, args.dataset or os.path.join(input_dir, "dataset"))
    centroid = refs.mean(0)
    centroid /= np.linalg.norm(centroid)
    for c, col in enumerate(columns):
        trig, leak, found = [], [], 0
        for r, (prompt, _) in enumerate(rows):
            e = scorer.embed_file(cells[r][c])
            s = None if e is None else float(e @ centroid)
            (leak if prompt == LEAK_PROMPT else trig).append(s)
            if prompt != LEAK_PROMPT and s is not None:
                found += 1
        scored = [s for s in trig if s is not None]
        col["identity"] = round(float(np.mean(scored)), 3) if scored else None
        col["identity_min"] = round(float(np.min(scored)), 3) if scored else None
        col["leak"] = round(float(np.mean([s for s in leak if s is not None])), 3) if any(s is not None for s in leak) else None
        col["faces"] = f"{found}/{len(trig)}"
        col["per_row"] = trig + leak
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        col["label"] += f"\nid {fmt(col['identity'])} (min {fmt(col['identity_min'])})\nleak {fmt(col['leak'])}  faces {col['faces']}"

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_dir, "lora_evals", stamp)
    os.makedirs(out_dir, exist_ok=True)
    row_labels = [wrap(("[LEAK, no trigger] " if p == LEAK_PROMPT else "") + p.replace(f"{trigger}, ", "").replace(f" of {trigger}", ""), 26) + f"\nseed {s}" for p, s in rows]
    grid_path = os.path.join(out_dir, "grid.jpg")
    build_grid(cells, [c["label"] for c in columns], row_labels, grid_path)
    report = {
        "timestamp": stamp, "size": [w, h], "seeds": seeds, "prompts": prompts,
        "columns": [{k: v for k, v in c.items()} for c in columns],
        "cells": cells,
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'column':<34} {'identity':>9} {'min':>6} {'leak':>6} {'faces':>6}")
    for col in columns:
        name = col["tag"] and f"{col['tag']}@{col['scale']:g}" or "baseline (no LoRA)"
        fmt = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"{name:<34} {fmt(col['identity']):>9} {fmt(col['identity_min']):>6} {fmt(col['leak']):>6} {col['faces']:>6}")
    print(f"\nGrid: {grid_path}\nLook at it before picking: python train_lora.py {project} --pick <tag> --scale <s>")
    return 0


if __name__ == "__main__":
    sys.exit(main())

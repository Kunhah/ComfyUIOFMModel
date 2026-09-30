#!/usr/bin/env python3
"""Train Krea 2 character LoRAs on fal.ai (fal-ai/krea-2-trainer) from a captioned dataset folder,
optionally several step counts at once, so you can compare them with eval_lora.py.

fal's trainer returns only the final weights (no intermediate checkpoints), so "see the result at
set step counts" means one short run per step count. Runs are cheap ($0.003/step) and are submitted
in parallel from one upload.

Every run is recorded in output/projects/<project>/lora_runs/<tag>.json, its weights downloaded to
models/loras/<project>/<tag>.safetensors, and logged to logs/generations.jsonl. A tag that already
exists is never re-trained (so you can't pay twice by accident) unless you pass --force.

Usage:
    python train_lora.py MyCharacter --trigger mychr1 --steps 600,1000 --dry-run
    python train_lora.py MyCharacter --trigger mychr1 --steps 600,1000
    python train_lora.py MyCharacter --list
    python train_lora.py MyCharacter --pick mychr1_s1000_lr5e-04_r1024 [--scale 0.9]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import folder_paths  # noqa: E402
import env_config  # noqa: E402
import logging_util  # noqa: E402
import pricing  # noqa: E402

ENDPOINT = "fal-ai/krea-2-trainer"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def runs_dir(project: str) -> str:
    d = os.path.join(logging_util.get_project_output_dir(project), "lora_runs")
    os.makedirs(d, exist_ok=True)
    return d


def load_runs(project: str) -> dict[str, dict]:
    d = runs_dir(project)
    runs = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                runs[f[:-5]] = json.load(fh)
    return runs


def build_zip(folder: str, trigger: str) -> tuple[bytes, int, str]:
    """Zips images at native resolution with their same-stem .txt captions. Refuses to build a
    zip where an image has no caption or a caption lacks the trigger, since fal would silently
    fall back to the bare trigger phrase for those."""
    files = sorted(os.listdir(folder))
    images = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    problems = []
    buf = io.BytesIO()
    digest = hashlib.sha1()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name in images:
            stem = os.path.splitext(name)[0]
            cap_path = os.path.join(folder, stem + ".txt")
            if not os.path.isfile(cap_path):
                problems.append(f"{name}: no {stem}.txt")
                continue
            with open(cap_path, encoding="utf-8") as f:
                caption = f.read().strip()
            if trigger not in caption:
                problems.append(f"{stem}.txt: missing trigger '{trigger}'")
            with open(os.path.join(folder, name), "rb") as f:
                data = f.read()
            digest.update(data + caption.encode())
            zf.writestr(name, data)
            zf.writestr(stem + ".txt", caption + "\n")
    if problems:
        raise SystemExit("Dataset problems:\n  " + "\n  ".join(problems))
    return buf.getvalue(), len(images), digest.hexdigest()[:12]


def tag_for(trigger: str, steps: int, lr: float, resolution: int) -> str:
    return f"{trigger}_s{steps}_lr{lr:.0e}_r{resolution}"


def train_one(fal_client, project: str, url: str, tag: str, args, steps: int, n_images: int, dataset_hash: str) -> dict:
    started = time.time()

    def on_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs or []:
                print(f"  [{tag}] {log.get('message', '')}", flush=True)

    result = fal_client.subscribe(
        ENDPOINT,
        arguments={
            "images_data_url": url,
            "trigger_phrase": args.trigger,
            "steps": steps,
            "learning_rate": args.lr,
            "resolution": args.resolution,
        },
        with_logs=True,
        on_queue_update=on_update,
    )
    lora_url = (result.get("lora_file") or {}).get("url")
    if not lora_url:
        raise RuntimeError(f"{tag}: no lora_file in response: {result!r}")

    lora_dir = os.path.join(folder_paths.get_folder_paths("loras")[0], project)
    os.makedirs(lora_dir, exist_ok=True)
    local_path = os.path.join(lora_dir, f"{tag}.safetensors")
    urllib.request.urlretrieve(lora_url, local_path)

    cost = pricing.estimate_fal_krea2_training_cost_usd(steps)
    record = {
        "tag": tag,
        "base": "krea-2",
        "endpoint": ENDPOINT,
        "trigger": args.trigger,
        "steps": steps,
        "learning_rate": args.lr,
        "resolution": args.resolution,
        "dataset": args.dataset,
        "dataset_hash": dataset_hash,
        "n_images": n_images,
        "lora_url": lora_url,
        "config_url": (result.get("config_file") or {}).get("url"),
        "local_path": local_path,
        "estimated_cost_usd": cost,
        "minutes": round((time.time() - started) / 60, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(runs_dir(project), f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    logging_util.append_generation_log({
        "provider": "fal_ai", "model": ENDPOINT, "operation": "lora_training", "project": project,
        "stage": "lora_training", "num_outputs": 1, "estimated_cost_usd": cost, "actual_cost_usd": None,
        "output_path": local_path, "notes": f"{tag}, {n_images} images, dataset {dataset_hash}",
    })
    return record


def cmd_pick(project: str, tag: str, scale: float) -> int:
    runs = load_runs(project)
    if tag not in runs:
        print(f"No run '{tag}'. Known: {', '.join(runs) or 'none'}")
        return 1
    run = runs[tag]
    path = logging_util.character_json_path(project)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.update(lora_url=run["lora_url"], lora_base=run["base"], lora_tag=tag, lora_scale=scale,
                lora_trigger_phrase=run["trigger"])
    if not data.get("trigger"):
        data["trigger"] = run["trigger"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"{project}: using {tag} @ {scale} (saved to {path})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project")
    parser.add_argument("--trigger", help="Unique made-up word, e.g. mychr1 (must appear in every caption).")
    parser.add_argument("--dataset", help="Captioned dataset folder (default: input/ai_influencer/<project>/dataset).")
    parser.add_argument("--steps", default="1000", help="One or more step counts, comma-separated, e.g. 600,1000.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Krea recommends 3e-4 to 7e-4 (constant). Default 5e-4.")
    parser.add_argument("--resolution", type=int, choices=[768, 1024], default=1024)
    parser.add_argument("--dry-run", action="store_true", help="Build and check the zip, print the cost, spend nothing.")
    parser.add_argument("--yes", action="store_true", help="Skip the spending confirmation.")
    parser.add_argument("--force", action="store_true", help="Re-train tags that already exist.")
    parser.add_argument("--budget", type=float, default=float(env_config.get("FAL_BUDGET_USD") or 0),
                         help="Refuse to run if logged fal.ai spend + this run would exceed this many dollars "
                              "(default: FAL_BUDGET_USD from .env, 0 = no guard).")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--pick", metavar="TAG", help="Record this run as the character's LoRA in character.json.")
    parser.add_argument("--scale", type=float, default=1.0, help="LoRA scale stored with --pick.")
    args = parser.parse_args()

    project = logging_util.sanitize_project_name(args.project)
    input_dir, _ = logging_util.ensure_project(project)

    if args.list:
        for tag, r in load_runs(project).items():
            print(f"{tag:<40} {r['n_images']} imgs  ${r['estimated_cost_usd']:.2f}  {r['minutes']} min  {r['finished_at']}")
        return 0
    if args.pick:
        return cmd_pick(project, args.pick, args.scale)
    if not args.trigger:
        parser.error("--trigger is required to train")

    args.dataset = os.path.abspath(os.path.expanduser(args.dataset or os.path.join(input_dir, "dataset")))
    steps_list = sorted({int(s) for s in args.steps.split(",")})
    existing = load_runs(project)
    todo = []
    for s in steps_list:
        tag = tag_for(args.trigger, s, args.lr, args.resolution)
        if tag in existing and not args.force:
            print(f"skip {tag}: already trained ({existing[tag]['lora_url']})")
        else:
            todo.append((tag, s))
    if not todo:
        return 0

    zip_bytes, n_images, dataset_hash = build_zip(args.dataset, args.trigger)
    total = sum(pricing.estimate_fal_krea2_training_cost_usd(s) for _, s in todo)
    print(f"Dataset: {n_images} captioned images, {len(zip_bytes) / 1e6:.1f} MB, hash {dataset_hash}")
    for tag, s in todo:
        print(f"  {tag}: {s} steps, lr {args.lr:g}, res {args.resolution} -> ${pricing.estimate_fal_krea2_training_cost_usd(s):.2f}")
    print(f"Total estimated cost: ${total:.2f}  (fal.ai, billed to your fal account)")
    spent = logging_util.provider_spend_to_date("fal_ai")
    if args.budget:
        print(f"Budget: ${spent:.2f} logged so far + ${total:.2f} = ${spent + total:.2f} of ${args.budget:.2f}")
    if args.dry_run:
        return 0
    logging_util.check_budget("fal_ai", total, args.budget)

    env_config.require_cli("FAL_KEY")
    if not args.yes and input(f"Spend ~${total:.2f}? Type yes: ").strip().lower() != "yes":
        print("Cancelled, nothing spent.")
        return 1

    import fal_client

    print("Uploading dataset...")
    url = fal_client.upload(zip_bytes, "application/zip", f"{project}_{dataset_hash}.zip")
    print(f"Training {len(todo)} run(s) in parallel (Krea 2 training takes a while)...")
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(todo)) as pool:
        futures = {pool.submit(train_one, fal_client, project, url, tag, args, s, n_images, dataset_hash): tag for tag, s in todo}
        for fut in concurrent.futures.as_completed(futures):
            tag = futures[fut]
            try:
                r = fut.result()
                print(f"DONE {tag} in {r['minutes']} min -> {r['local_path']}")
            except Exception as e:
                failed += 1
                print(f"FAILED {tag}: {e}", file=sys.stderr)
    print("\nNext: python eval_lora.py", project, "--runs", ",".join(t for t, _ in todo))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

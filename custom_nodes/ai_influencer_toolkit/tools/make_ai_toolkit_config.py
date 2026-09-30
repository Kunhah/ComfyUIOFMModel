#!/usr/bin/env python3
"""Write an Ostris AI Toolkit job config for a Krea 2 character LoRA.

The defaults follow the recipe in README_AI_INFLUENCER.md ("Local Krea 2 pipeline"):
  - architecture Krea 2 RAW (krea/Krea-2-Raw). Never train on Turbo: it lowers quality.
    You train on RAW and generate on Turbo.
  - 3000 steps, AdamW8bit, learning rate 1e-4 (the toolkit default)
  - a checkpoint every 250 steps, and max_step_saves_to_keep=100 so the toolkit keeps every
    checkpoint instead of deleting old ones (the default keeps only 4)
  - batch size 1 (use 1 on an RTX 5090 to avoid running out of memory)

You can load the generated file in the AI Toolkit UI (New Job -> Show Advanced -> paste) or run it
directly on the pod:  python run.py config/<name>.yaml

Usage:
    python make_ai_toolkit_config.py lina --trigger l1na \
        --dataset-path /workspace/ai-toolkit/datasets/lina -o lina_krea2.yaml
"""
from __future__ import annotations

import argparse
import sys

import yaml

SAMPLE_PROMPTS = [
    "photo of [trigger], close-up portrait, looking at the camera, soft window light",
    "photo of [trigger] laughing at a cafe table, candid smartphone photo",
    "full body photo of [trigger] walking down a city street at night, wearing a red coat",
    "photo of [trigger] in profile, hiking on a mountain trail, overcast daylight",
]


def build_config(args: argparse.Namespace) -> dict:
    return {
        "job": "extension",
        "config": {
            "name": args.name,
            "process": [{
                "type": "diffusion_trainer",
                "training_folder": "output",
                "device": "cuda",
                "trigger_word": args.trigger,
                "network": {"type": "lora", "linear": args.rank, "linear_alpha": args.rank},
                "save": {
                    "dtype": "bf16",
                    "save_every": args.save_every,
                    "max_step_saves_to_keep": args.max_saves,
                    "save_format": "diffusers",
                    "push_to_hub": False,
                },
                "datasets": [{
                    "folder_path": args.dataset_path,
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "cache_latents_to_disk": True,
                    "resolution": [512, 768, 1024],
                }],
                "train": {
                    "batch_size": args.batch_size,
                    "steps": args.steps,
                    "gradient_accumulation": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit",
                    "timestep_type": "linear",
                    "content_or_style": "balanced",
                    "optimizer_params": {"weight_decay": 1e-4},
                    "lr": args.lr,
                    "dtype": "bf16",
                },
                "model": {
                    "name_or_path": "krea/Krea-2-Raw",
                    "arch": "krea2",
                    "quantize": True,
                    "qtype": "qfloat8",
                    "quantize_te": True,
                    "qtype_te": "qfloat8",
                    "low_vram": args.low_vram,
                },
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": args.save_every,
                    "width": 832,
                    "height": 1024,
                    "prompts": args.sample_prompts or SAMPLE_PROMPTS,
                    # ai-toolkit defaults `neg` to False (a bool) and Krea 2's text encoder
                    # concatenates it to a string -> TypeError on the very first sample.
                    "neg": "",
                    "seed": 42,
                    "walk_seed": False,
                    "guidance_scale": 4,
                    "sample_steps": 25,
                },
            }],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="LoRA/job name; checkpoints are saved as <name>_<step>.safetensors.")
    parser.add_argument("--trigger", required=True, help="A unique made-up word the base model doesn't know, e.g. l1na.")
    parser.add_argument("--dataset-path", required=True, help="Dataset folder as seen from the pod (images + .txt captions).")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--max-saves", type=int, default=100, help="Keep this many intermediate checkpoints. Keep it high so none get deleted.")
    parser.add_argument("--batch-size", type=int, default=1, help="Use 1 on an RTX 5090 (32GB).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--low-vram", action="store_true", help="For GPUs under ~40GB (e.g. RTX 5090).")
    parser.add_argument("--sample-prompt", action="append", default=[], dest="sample_prompts",
                         help="Validation prompt rendered at every checkpoint (repeatable). Use [trigger] for the "
                              "trigger word. Defaults to the built-in set. Use outfits/places NOT in the dataset.")
    parser.add_argument("-o", "--output", default="", help="Output .yaml path (default: <name>.yaml).")
    args = parser.parse_args()

    if len(args.trigger.split()) != 1:
        print("--trigger should be a single made-up word (e.g. l1na), so it only means your character.")
        return 1
    if args.max_saves * args.save_every < args.steps:
        print(f"--max-saves {args.max_saves} x --save-every {args.save_every} < --steps {args.steps}: "
              "the toolkit would delete early checkpoints. Raise --max-saves.")
        return 1

    out = args.output or f"{args.name}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(build_config(args), f, sort_keys=False)
    n = args.steps // args.save_every
    print(f"Wrote {out}: Krea 2 RAW, {args.steps} steps, {n} checkpoints (every {args.save_every}), trigger '{args.trigger}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

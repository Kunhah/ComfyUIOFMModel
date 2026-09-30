#!/usr/bin/env python3
"""List a character's LoRA checkpoints and past checkpoint tests, and record the one you picked
in character.json ("local_lora"), which the Apply Character LoRA node reads.

Usage:
    python pick_checkpoint.py lina --filter l1na            # list checkpoints + test results
    python pick_checkpoint.py lina --filter l1na --step 2500
    python pick_checkpoint.py lina --lora lina/l1na_000002500.safetensors --strength 0.9
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import folder_paths  # noqa: E402,F401  (needs REPO_ROOT on sys.path first)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import logging_util  # noqa: E402


# Same naming rule as nodes_local_lora (not imported: it pulls in ComfyUI's model code).
STEP_RE = re.compile(r"_(\d{3,})$")


def step_of(name: str) -> int | None:
    m = STEP_RE.search(os.path.splitext(os.path.basename(name))[0])
    return int(m.group(1)) if m else None


def find_checkpoints(needle: str) -> list[tuple[str, int | None]]:
    names = [n for n in folder_paths.get_filename_list("loras") if needle.lower() in n.replace("\\", "/").lower()]
    return sorted(((n, step_of(n)) for n in names), key=lambda x: (x[1] is None, x[1] or 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project")
    parser.add_argument("--filter", default="", help="Part of the checkpoint filenames, e.g. l1na.")
    parser.add_argument("--step", type=int, help="Pick the checkpoint saved at this step (needs --filter).")
    parser.add_argument("--final", action="store_true", help="Pick the final (unnumbered) checkpoint (needs --filter).")
    parser.add_argument("--lora", default="", help="Pick this exact file (relative to models/loras).")
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()

    project = logging_util.sanitize_project_name(args.project)
    logging_util.ensure_project(project)
    checkpoints = find_checkpoints(args.filter) if args.filter else []

    choice = args.lora
    if args.step is not None or args.final:
        matches = [n for n, s in checkpoints if (s is None if args.final else s == args.step)]
        if len(matches) != 1:
            print(f"Expected one checkpoint, found {len(matches)}: {matches}")
            return 1
        choice = matches[0]

    if not choice:
        print(f"Checkpoints matching '{args.filter}':" if args.filter else "Pass --filter to list checkpoints.")
        for n, s in checkpoints:
            print(f"  {'final' if s is None else s:>8}  {n}")
        tests = sorted(glob.glob(os.path.join(logging_util.get_project_output_dir(project), "lora_tests", "*", "report.json")))
        for path in tests[-5:]:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
            print(f"\nTest {report['timestamp']}  ({os.path.dirname(path)}/grid.png)")
            for col in report["columns"]:
                score = f"identity {col['identity']:.3f}  copy {col['copy']:.3f}" if "identity" in col else "(not scored)"
                print(f"  {col['label']:<22} {score}")
            if report.get("best_by_identity"):
                print(f"  best by identity: {report['best_by_identity']}")
        return 0

    if choice not in folder_paths.get_filename_list("loras"):
        print(f"'{choice}' is not in models/loras.")
        return 1
    path = logging_util.character_json_path(project)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["local_lora"] = choice
    data["local_lora_strength"] = args.strength
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"{project}: local_lora = {choice} @ {args.strength}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

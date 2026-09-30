#!/usr/bin/env python3
"""Scaffold a new AI-influencer character project: creates the input/ and output/ folder tree
and a character.json you can then hand-edit, without needing to open ComfyUI first.

Usage:
    python new_project.py jane_doe --trigger "photo of sks woman" \
        --feature "olive skin, dark brown wavy hair, brown eyes" \
        --feature "small scar above left eyebrow"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import folder_paths  # noqa: E402  (needs REPO_ROOT on sys.path first)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import logging_util  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Project/character name (used as the folder name).")
    parser.add_argument("--trigger", default="", help="Short identity phrase prepended to every prompt.")
    parser.add_argument("--feature", action="append", default=[], dest="permanent_features",
                         help="A permanent physical feature; repeat --feature for more.")
    args = parser.parse_args()

    project = logging_util.sanitize_project_name(args.name)
    input_dir, output_dir = logging_util.ensure_project(project)

    if args.trigger or args.permanent_features:
        path = logging_util.character_json_path(project)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if args.trigger:
            data["trigger"] = args.trigger
        if args.permanent_features:
            data["permanent_features"] = args.permanent_features
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    print(f"Project '{project}' ready.")
    print(f"  References to upload (LoadImage will browse these): {input_dir}")
    print(f"  Generated output / character.json:                  {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

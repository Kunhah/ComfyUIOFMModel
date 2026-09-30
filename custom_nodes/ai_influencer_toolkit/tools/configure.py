#!/usr/bin/env python3
"""Set the toolkit's API keys and settings in .env (at the ComfyUI root, never committed).

    python custom_nodes/ai_influencer_toolkit/tools/configure.py            # walk through every setting
    python custom_nodes/ai_influencer_toolkit/tools/configure.py --show     # what is set; keys are masked
    python custom_nodes/ai_influencer_toolkit/tools/configure.py --set HF_TOKEN=hf_xxx --set FAL_KEY=

In the walk-through, Enter keeps the current value and "-" clears it. Keys are typed without echo.
Lines in .env that aren't toolkit settings (comments, your own variables) are kept as they are.
Restart ComfyUI afterwards so the nodes see the new values.
"""
from __future__ import annotations

import argparse
import getpass
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_config  # noqa: E402


def write_env(updates: dict[str, str]) -> None:
    """Replace the KEY= line of each updated setting in place, append the ones .env doesn't have."""
    path = env_config.ENV_FILE
    if not os.path.exists(path):
        example = os.path.join(env_config.REPO_ROOT, ".env.example")
        if os.path.exists(example):
            shutil.copyfile(example, path)
        else:
            open(path, "w").close()
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    pending = dict(updates)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip().removeprefix("export ").strip() if "=" in line else ""
        if key in pending and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={pending.pop(key)}"
    lines += [f"{key}={value}" for key, value in pending.items()]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(tmp, 0o600)  # keys: readable by you only
    os.replace(tmp, path)


def show() -> None:
    current = env_config.parse_env_file()
    print(f"{env_config.ENV_FILE}\n")
    for s in env_config.SETTINGS:
        value = current.get(s.name, "")
        shown = env_config.mask(value) if s.secret else (value or f"(default: {s.default or 'blank'})")
        print(f"  {s.name:<31} {shown}")
    overridden = [s.name for s in env_config.SETTINGS if os.environ.get(s.name) and os.environ[s.name] != current.get(s.name)]
    if overridden:
        print(f"\nSet in your shell environment, which wins over .env: {', '.join(overridden)}")


def walk() -> dict[str, str]:
    current = env_config.parse_env_file()
    updates: dict[str, str] = {}
    print("Enter keeps the current value, '-' clears it.\n")
    for s in env_config.SETTINGS:
        value = current.get(s.name, "")
        print(f"{s.name}: {s.help}" + (f"\n  get it at {s.url}" if s.url else ""))
        shown = env_config.mask(value) if s.secret else (value or f"default {s.default or 'blank'}")
        prompt = f"  [{shown}] > "
        answer = (getpass.getpass(prompt) if s.secret else input(prompt)).strip()
        if answer == "-":
            updates[s.name] = ""
        elif answer:
            updates[s.name] = answer
        print()
    return updates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="List the settings (keys masked) and exit.")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="Set one value without prompting; repeatable. NAME= clears it.")
    args = ap.parse_args()

    if args.show:
        show()
        return 0
    if args.set:
        updates = {}
        for item in args.set:
            name, sep, value = item.partition("=")
            if not sep or name not in env_config.BY_NAME:
                raise SystemExit(f"--set wants NAME=VALUE with NAME one of: {', '.join(env_config.BY_NAME)}")
            updates[name] = value.strip()
    else:
        try:
            updates = walk()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled, .env unchanged.")
            return 1
    if updates:
        write_env(updates)
        print(f"Saved {', '.join(updates)} to {env_config.ENV_FILE}. Restart ComfyUI to pick them up.")
    else:
        print("Nothing changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

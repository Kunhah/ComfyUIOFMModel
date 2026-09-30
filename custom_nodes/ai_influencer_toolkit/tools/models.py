#!/usr/bin/env python3
"""The browser's "Models" window from a terminal: what this PC can run, what a workflow needs, downloads.

    python custom_nodes/ai_influencer_toolkit/tools/models.py                     # GPU advice + every model, installed or not
    python custom_nodes/ai_influencer_toolkit/tools/models.py 09b                 # what Workflow 09b needs (any part of the file name)
    python custom_nodes/ai_influencer_toolkit/tools/models.py 09b --download      # ... and download what's missing
    python custom_nodes/ai_influencer_toolkit/tools/models.py --version gguf      # choose the Krea 2 version (fp8 | nvfp4 | gguf | bf16)

Files go into ComfyUI/models/<folder>/ and an interrupted download resumes where it stopped. The gguf version (8 GB
cards) also needs the ComfyUI-GGUF add-on; --download installs it into custom_nodes/ (restart ComfyUI afterwards).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(PACK, "..", "..")))  # ComfyUI's folder_paths
import env_config  # noqa: E402
import model_catalog as cat  # noqa: E402
from launcher import nvidia_gpu  # noqa: E402

WORKFLOWS = os.path.join(env_config.REPO_ROOT, "user", "default", "workflows", "ai_influencer")


def gpu() -> dict | None:
    import subprocess

    found = nvidia_gpu()
    if not found:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        vram = float(out.splitlines()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        vram = 0.0
    return {"name": found[0], "vram_gb": vram, "cc": found[1] or 7.5}


def find_workflow(query: str) -> str:
    hits = [p for p in sorted(glob.glob(os.path.join(WORKFLOWS, "*.json"))) if query.lower() in os.path.basename(p).lower()]
    exact = [p for p in hits if os.path.basename(p).split("_")[0] == query]
    if len(exact) == 1 or len(hits) == 1:
        return (exact or hits)[0]
    names = ", ".join(os.path.basename(p) for p in hits) or "none"
    raise SystemExit(f"{query!r} matches {names}. Give more of the workflow's file name.")


def show(names: list[str]) -> list[str]:
    missing = []
    for n in names:
        m = cat.BY_NAME[n]
        ok = cat.installed(m)
        missing += [] if ok else [n]
        print(f"  {'ok  ' if ok else '    '}{m.size / cat.GB:6.1f} GB  {m.what}\n{'':16}{m.name} -> models/{m.folder}")
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workflow", nargs="?", help="Workflow number or part of its file name, e.g. 09b.")
    ap.add_argument("--download", action="store_true", help="Download the missing files the workflow needs.")
    ap.add_argument("--version", choices=list(cat.VERSIONS), help="Save the Krea 2 version to use.")
    args = ap.parse_args()

    if args.version:
        from configure import write_env

        write_env({"AI_INFLUENCER_KREA_VERSION": args.version})
        os.environ["AI_INFLUENCER_KREA_VERSION"] = args.version
        print(f"Krea 2 version: {cat.VERSIONS[args.version]['label']}. The workflows switch to it when opened.")

    recommended, advice = cat.recommend(gpu())
    version = cat.preferred_version()
    print(f"{advice}\nKrea 2 version: {cat.VERSIONS[version]['label']}"
          + (f" (recommended here: {cat.VERSIONS[recommended]['label']}, change with --version {recommended})"
             if recommended != version else "") + "\n")

    if not args.workflow:
        print("Every model the local workflows use (ok = downloaded):")
        show([m.name for m in cat.MODELS])
        if args.download:
            raise SystemExit("\n--download needs a workflow, so only what it uses is downloaded.")
        return 0

    path = find_workflow(args.workflow)
    names = list(dict.fromkeys(cat.swap_for_version(n, version) for n in cat.workflow_models(path)))
    print(f"{os.path.basename(path)} uses:" if names else f"{os.path.basename(path)} uses no local model files.")
    missing = show(names)
    addon = any(cat.BY_NAME[n].loader for n in names) and not cat.addon_present()
    if addon:
        print(f"  also needs the {cat.GGUF_ADDON} add-on (not installed)")
    if args.download and addon:
        print(f"\nInstalling {cat.GGUF_ADDON} into {cat.addon_dir()} ...")
        cat.install_addon()
        print("Installed. Restart ComfyUI to load it.")
    if not missing:
        print("\nEverything it needs is downloaded." if names else "")
        return 0
    total = sum(cat.BY_NAME[n].size for n in missing)
    print(f"\nMissing: {total / cat.GB:.1f} GB. Free disk: {cat.free_bytes(cat.BY_NAME[missing[0]]) / cat.GB:.0f} GB.")
    if not args.download:
        print(f"Download them with:  python {os.path.relpath(__file__)} {args.workflow} --download")
        return 0

    cat.DOWNLOADS.start(missing)
    while True:
        state = cat.DOWNLOADS.snapshot()
        done = sum(s["done"] for s in state.values())
        print(f"\r  {done / cat.GB:6.1f} of {total / cat.GB:.1f} GB", end="", flush=True)
        if all(s["status"] != "running" for s in state.values()):
            break
        time.sleep(1)
    print()
    failed = {n: s["error"] for n, s in state.items() if s["status"] == "failed"}
    for n, err in failed.items():
        print(f"  {n}: {err}")
    print("Some downloads failed; run the same command again to resume." if failed else "All downloaded.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

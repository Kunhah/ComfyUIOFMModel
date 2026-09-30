#!/usr/bin/env python3
"""What START_COMFYUI.bat runs, inside the venv it created: install what's missing, then start ComfyUI.

    venv\\Scripts\\python.exe custom_nodes\\ai_influencer_toolkit\\tools\\windows_launcher.py            # set up if needed, then start
    venv\\Scripts\\python.exe custom_nodes\\ai_influencer_toolkit\\tools\\windows_launcher.py --reinstall  # redo the installs

On the first run it picks the PyTorch build for this PC (NVIDIA GPU or CPU only), installs
requirements.txt, and opens the API-key window once. It writes a stamp file in the venv, so later runs skip
straight to starting ComfyUI, unless requirements.txt changed (after downloading a newer copy of the repo).
Written for Windows, but it runs anywhere, which is how it is tested.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_config  # noqa: E402

ROOT = env_config.REPO_ROOT
STAMP = os.path.join(sys.prefix, ".ai_influencer_setup")
TORCH_INDEX = "https://download.pytorch.org/whl/"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def nvidia_gpu() -> tuple[str, float] | None:
    """(name, compute capability) of the first NVIDIA GPU, or None. The capability is 0 when the driver is too
    old to report it."""
    for query in ("name,compute_cap", "name"):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode == 0 and out.stdout.strip():
            fields = [f.strip() for f in out.stdout.splitlines()[0].split(",")]
            try:
                cap = float(fields[1]) if len(fields) > 1 else 0.0
            except ValueError:
                cap = 0.0
            return fields[0], cap
    return None


def torch_build() -> str:
    """cu130 for current NVIDIA cards, cu126 for GTX 10-series and older (compute < 7.5), cpu without one.
    Matches the choices in ComfyUI's own README."""
    gpu = nvidia_gpu()
    if gpu is None:
        return "cpu"
    name, cap = gpu
    return "cu126" if 0 < cap < 7.5 else "cu130"


def requirements_hash(build: str) -> str:
    with open(os.path.join(ROOT, "requirements.txt"), "rb") as f:
        return hashlib.sha256(f.read() + build.encode()).hexdigest()


def pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args]
    if subprocess.call(cmd, cwd=ROOT) != 0:
        raise SystemExit("\nThe install above failed. Check the internet connection and run START_COMFYUI.bat again.\n"
                         "If it keeps failing, send the text above to whoever shared this folder with you.")


def install(build: str) -> None:
    gpu = nvidia_gpu()
    say("=" * 70)
    say("First-time setup. This downloads a few GB and can take 10-30 minutes. Keep this window open.")
    say(f"Graphics card: {gpu[0] if gpu else 'no NVIDIA card found, ComfyUI will run on the CPU (slow)'}")
    say("=" * 70 + "\n")
    pip("--upgrade", "pip")
    pip("torch", "torchvision", "torchaudio", "--index-url", TORCH_INDEX + build)
    pip("-r", "requirements.txt")
    with open(STAMP, "w", encoding="utf-8") as f:
        f.write(requirements_hash(build))
    say("\nSetup finished.\n")


def installed_hash() -> str:
    try:
        with open(STAMP, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reinstall", action="store_true", help="Run the installs again even if nothing changed.")
    args = ap.parse_args()

    build = torch_build()
    first_run = not installed_hash()
    if args.reinstall or installed_hash() != requirements_hash(build):
        install(build)

    if first_run and not os.path.exists(env_config.ENV_FILE):
        say("Opening the API-key window. Fill in what you have; every box can stay empty and be set later\n"
            "with SETTINGS_API_KEYS.bat.")
        subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "configure_gui.py")])

    cmd = [sys.executable, "main.py", "--auto-launch"]
    if build == "cpu":
        cmd.append("--cpu")
    say("Starting ComfyUI. The browser opens by itself when it is ready (or go to http://127.0.0.1:8188).")
    say("Leave this window open while you use ComfyUI; closing it stops ComfyUI.\n")
    try:
        return subprocess.call(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""What START_COMFYUI.bat (Windows) and start_comfyui.sh (Linux) run, inside the venv they created:
install what's missing, then start ComfyUI.

    venv/bin/python custom_nodes/ai_influencer_toolkit/tools/launcher.py              # set up if needed, then start
    venv/bin/python custom_nodes/ai_influencer_toolkit/tools/launcher.py --reinstall  # redo the installs, torch too
    venv/bin/python custom_nodes/ai_influencer_toolkit/tools/launcher.py -- --port 8190   # extra main.py arguments
    venv/bin/python custom_nodes/ai_influencer_toolkit/tools/launcher.py --keys       # only the API-key window

PyTorch is installed only when the venv doesn't have it (or with --reinstall), as the build that fits this PC:
cu130, cu126 for GTX 10-series and older, or CPU when there is no NVIDIA card. requirements.txt is installed on
the first run and again whenever it changes; a stamp file in the venv remembers which version is installed.
The first run also opens the API-key window once when there is no .env yet.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_config  # noqa: E402

ROOT = env_config.REPO_ROOT
TOOLS = os.path.dirname(os.path.abspath(__file__))
STAMP = os.path.join(sys.prefix, ".ai_influencer_setup")
TORCH_INDEX = "https://download.pytorch.org/whl/"
BAT = "START_COMFYUI.bat" if os.name == "nt" else "./start_comfyui.sh"


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


def torch_build(gpu: tuple[str, float] | None) -> str:
    """Matches the choices in ComfyUI's own README."""
    if gpu is None:
        return "cpu"
    return "cu126" if 0 < gpu[1] < 7.5 else "cu130"


def cuda_works() -> bool:
    """Whether the installed torch can run on this GPU: a CUDA build can still lack kernels for an old card."""
    probe = "import torch; assert torch.cuda.is_available(); print((torch.zeros(1, device='cuda') + 1).item())"
    return subprocess.run([sys.executable, "-c", probe], capture_output=True).returncode == 0


def requirements_hash() -> str:
    with open(os.path.join(ROOT, "requirements.txt"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def installed_hash() -> str:
    try:
        with open(STAMP, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args]
    if subprocess.call(cmd, cwd=ROOT) != 0:
        raise SystemExit(f"\nThe install above failed. Check the internet connection and run {BAT} again.\n"
                         "If it keeps failing, send the text above to whoever shared this folder with you.")


def install(gpu: tuple[str, float] | None, with_torch: bool) -> None:
    say("=" * 70)
    if with_torch:
        say("Installing. This downloads a few GB and can take 10-30 minutes. Keep this window open.")
        say(f"Graphics card: {gpu[0] if gpu else 'no NVIDIA card found, ComfyUI will run on the CPU (slow)'}")
    else:
        say("Installing the Python packages ComfyUI needs (requirements.txt).")
    say("=" * 70 + "\n")
    pip("--upgrade", "pip")
    if with_torch:
        pip("--force-reinstall", "torch", "torchvision", "torchaudio", "--index-url", TORCH_INDEX + torch_build(gpu))
    pip("-r", "requirements.txt")
    with open(STAMP, "w", encoding="utf-8") as f:
        f.write(requirements_hash())
    say("\nSetup finished.\n")


def open_key_window() -> None:
    """The window (configure_gui.py) when Tk is available, else configure.py's questions in the terminal."""
    if importlib.util.find_spec("tkinter") is not None and subprocess.call(
            [sys.executable, "-c", "import tkinter; tkinter.Tk().destroy()"], stderr=subprocess.DEVNULL) == 0:
        say("Opening the API-key window. Fill in what you have; every box can stay empty.")
        subprocess.call([sys.executable, os.path.join(TOOLS, "configure_gui.py")])
    else:
        say("API keys: answer what you have, press Enter to skip any of them.\n")
        subprocess.call([sys.executable, os.path.join(TOOLS, "configure.py")])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reinstall", action="store_true", help="Reinstall PyTorch and requirements.txt.")
    ap.add_argument("--keys", action="store_true", help="Only open the API-key window (or terminal questions).")
    ap.add_argument("comfy_args", nargs="*", help="Passed on to main.py (put them after --).")
    args = ap.parse_args()
    if args.keys:
        open_key_window()
        return 0

    gpu = nvidia_gpu()
    first_run = not installed_hash()
    need_torch = args.reinstall or importlib.util.find_spec("torch") is None
    if need_torch or installed_hash() != requirements_hash():
        install(gpu, need_torch)

    if first_run and not os.path.exists(env_config.ENV_FILE):
        open_key_window()

    cmd = [sys.executable, "main.py", "--auto-launch", *args.comfy_args]
    if (gpu is None or not cuda_works()) and "--cpu" not in cmd:
        say("The graphics card can't be used by this PyTorch, so ComfyUI runs on the CPU (slow for local models).")
        cmd.append("--cpu")
    say("Starting ComfyUI. The browser opens by itself when it is ready (or go to http://127.0.0.1:8188).")
    say("Leave this window open while you use ComfyUI; closing it (or Ctrl+C) stops ComfyUI.\n")
    try:
        return subprocess.call(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

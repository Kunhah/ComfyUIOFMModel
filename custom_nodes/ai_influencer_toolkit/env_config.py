"""Every key and per-user setting the toolkit reads, in one place.

Values come from the process environment first, then from `.env` at the ComfyUI root (never
committed; `.env.example` is the template). Edit them with

    python custom_nodes/ai_influencer_toolkit/tools/configure.py          # interactive
    python custom_nodes/ai_influencer_toolkit/tools/configure.py --show   # what is set (masked)

Nothing in this file may hold a real value: the defaults below are the public ones.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

PACK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PACK_DIR, "..", ".."))
ENV_FILE = os.path.join(REPO_ROOT, ".env")


@dataclass(frozen=True)
class Setting:
    name: str
    help: str
    default: str = ""
    secret: bool = False
    url: str = ""


SETTINGS = [
    Setting("COMFY_API_KEY", "Comfy account key for the partner API nodes in Workflows 01-05 (Krea, GPT Image, "
            "Seedance). Not needed if you sign in inside the ComfyUI web UI instead.",
            secret=True, url="https://platform.comfy.org/login"),
    Setting("COMFY_API_BASE_URL", "Comfy API gateway.", default="https://api.comfy.org"),
    Setting("COMFY_SERVER_URL", "Your running ComfyUI, used by tools/queue_workflow.py.",
            default="http://127.0.0.1:8188"),
    Setting("AI_INFLUENCER_DEFAULT_PROJECT", "Character used when a node's project field is blank "
            "(folder name under input/ai_influencer/ and output/projects/).", default="default"),
    Setting("AI_INFLUENCER_KREA_VERSION", "Which Krea 2 version the local workflows use: fp8 (standard, 12 GB+ "
            "cards), gguf (low memory, 8 GB cards), nvfp4 (small, RTX 50-series) or bf16 (full quality, 40 GB+). "
            "Set it in the Models window.",
            default="fp8"),
    Setting("FAL_KEY", "fal.ai key, only for the fal LoRA nodes (Workflow 06) and train_lora.py / eval_lora.py.",
            secret=True, url="https://fal.ai/dashboard/keys"),
    Setting("FAL_BUDGET_USD", "Refuse fal.ai jobs estimated above this many dollars (0 = no limit).", default="0"),
    Setting("VAST_API_KEY", "vast.ai key for renting GPU pods (vast_pod.py, comfy_pod.py, the Pod queue).",
            secret=True, url="https://cloud.vast.ai/manage-keys/"),
    Setting("HF_TOKEN", "Hugging Face token with write access: downloads gated Krea 2 and moves datasets, "
            "checkpoints and the pod kit through a private HF dataset repo.",
            secret=True, url="https://huggingface.co/settings/tokens"),
    Setting("AI_INFLUENCER_HF_REPO", "That private HF dataset repo, as user/name. "
            "Blank = <your HF user>/<project>-lora-transfer, created on first use."),
    Setting("AI_INFLUENCER_LORA_PATH", "Path inside the HF repo of the character LoRA the ComfyUI pod loads, "
            "e.g. results/<instance id>/<name>_000002500.safetensors (vast_pod.py watch prints it)."),
]
BY_NAME = {s.name: s for s in SETTINGS}

_loaded: float | None | bool = False  # mtime of .env when last read; False = never read
_from_file: set[str] = set()  # keys load_env() put into os.environ (the shell's own stay untouched)


def parse_env_file(path: str = ENV_FILE) -> dict[str, str]:
    """KEY=VALUE lines; blank lines, # comments and an optional `export ` prefix are ignored,
    matching surrounding quotes are stripped."""
    values: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


def load_env() -> None:
    """Copy `.env` into os.environ, without overriding variables set in the shell (fal_client and
    huggingface_hub read FAL_KEY / HF_TOKEN from the environment themselves). Read again whenever
    `.env` changes, so a key saved while ComfyUI is running is used without a restart."""
    global _loaded
    try:
        mtime = os.path.getmtime(ENV_FILE)
    except OSError:
        mtime = None
    if _loaded is not False and _loaded == mtime:
        return
    _loaded = mtime
    for key, value in parse_env_file(ENV_FILE).items():
        if key in _from_file or not os.environ.get(key):
            if value:
                os.environ[key] = value
                _from_file.add(key)
            elif key in _from_file:
                os.environ.pop(key, None)  # cleared in .env
                _from_file.discard(key)


def get(name: str) -> str:
    load_env()
    default = BY_NAME[name].default if name in BY_NAME else ""
    return os.environ.get(name, "").strip() or default


def require(name: str) -> str:
    """The value, or ValueError naming the setting, where to get it and how to set it."""
    value = get(name)
    if value:
        return value
    s = BY_NAME.get(name)
    where = f" Get one at {s.url}." if s and s.url else ""
    raise ValueError(f"{name} is not set.{where} Set it with: "
                     f"python custom_nodes/ai_influencer_toolkit/tools/configure.py (or add {name}=... to .env)")


def require_cli(*names: str) -> None:
    """require() for command-line tools: exits with the message instead of a traceback."""
    for name in names:
        try:
            require(name)
        except ValueError as e:
            raise SystemExit(str(e)) from None


def hf_repo(project: str) -> str:
    """AI_INFLUENCER_HF_REPO, else <HF user>/<project>-lora-transfer."""
    repo = get("AI_INFLUENCER_HF_REPO")
    if repo:
        return repo
    from huggingface_hub import HfApi

    return f"{HfApi(token=require("HF_TOKEN")).whoami()['name']}/{project.lower()}-lora-transfer"


def mask(value: str) -> str:
    if not value:
        return "(not set)"
    return "*" * 8 if len(value) < 20 else f"...{value[-4:]}"

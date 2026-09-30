"""Every model file the local workflows (07-10, 09b, 12) load: where it goes, where it comes from, how big it is,
and what it is for in plain words. Used by model_routes.py (the "Models" window in the browser) and
tools/models.py (the same from a terminal).

Krea 2 comes in three versions of the same model. The workflows are saved with the FP8 one; the browser swaps in
whichever version is installed (AI_INFLUENCER_KREA_VERSION first) when a workflow is opened.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import urllib.request
from dataclasses import asdict, dataclass

try:
    from . import env_config
except ImportError:  # tools/models.py puts this folder on sys.path
    import env_config

GB = 1e9


@dataclass(frozen=True)
class Model:
    name: str
    folder: str      # ComfyUI models/ subfolder
    url: str
    size: int        # bytes
    what: str        # one line for people who don't know what a VAE is
    group: str = ""  # files that are versions of each other share a group
    loader: str = ""  # node type that loads it when not ComfyUI's usual one (GGUF files: UnetLoaderGGUF)


HF = "https://huggingface.co/Comfy-Org/"
MODELS = [
    Model("krea2_turbo_fp8_scaled.safetensors", "diffusion_models", HF + "Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors",
          13141730784, "Krea 2, the model that draws the image (FP8 version)", "krea2"),
    Model("krea2_turbo_nvfp4.safetensors", "diffusion_models", HF + "Krea-2/resolve/main/diffusion_models/krea2_turbo_nvfp4.safetensors",
          7673668448, "Krea 2, the model that draws the image (small NVFP4 version)", "krea2"),
    Model("krea2_turbo_bf16.safetensors", "diffusion_models", HF + "Krea-2/resolve/main/diffusion_models/krea2_turbo_bf16.safetensors",
          26283332608, "Krea 2, the model that draws the image (full-quality BF16 version)", "krea2"),
    Model("krea2_turbo-Q4_K_M.gguf", "diffusion_models", "https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF/resolve/main/krea2_turbo-Q4_K_M.gguf",
          7486289184, "Krea 2, the model that draws the image (low-memory GGUF Q4 version)", "krea2", loader="UnetLoaderGGUF"),
    Model("qwen3vl_4b_fp8_scaled.safetensors", "text_encoders", HF + "Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
          5242467968, "Reads your prompt and reference pictures for Krea 2 (FP8 version)", "krea2_te"),
    Model("qwen3vl_4b_bf16.safetensors", "text_encoders", HF + "Krea-2/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors",
          8875719384, "Reads your prompt and reference pictures for Krea 2 (full-quality BF16 version)", "krea2_te"),
    Model("qwen_image_vae.safetensors", "vae", HF + "Krea-2/resolve/main/vae/qwen_image_vae.safetensors",
          253806246, "Turns Krea 2's result into a normal picture"),
    Model("krea2_style_reference.safetensors", "loras", HF + "Krea-2/resolve/main/loras/krea2_style_reference.safetensors",
          457111760, "Lets Krea 2 follow the reference photos of her"),
    Model("sdpose_wholebody_fp16.safetensors", "checkpoints", HF + "SDPose/resolve/main/checkpoints/sdpose_wholebody_fp16.safetensors",
          1916645792, "Finds body poses in photos (the skeleton she copies)"),
    Model("birefnet.safetensors", "background_removal", HF + "BiRefNet/resolve/main/background_removal/birefnet.safetensors",
          444473596, "Separates her from the background, for the background blur"),
    Model("seedvr2_3b_int8_convrot.safetensors", "diffusion_models", HF + "SeedVR2/resolve/main/diffusion_models/seedvr2_3b_int8_convrot.safetensors",
          3458259704, "Upscaler that makes the final picture bigger and sharper"),
    Model("seedvr2_ema_vae_fp16.safetensors", "vae", HF + "SeedVR2/resolve/main/vae/seedvr2_ema_vae_fp16.safetensors",
          501324814, "Part of the upscaler"),
    Model("sigclip_vision_patch14_384.safetensors", "clip_vision", HF + "sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors",
          856505640, "Scores how much each test image looks like her (Workflow 07)"),
]
BY_NAME = {m.name: m for m in MODELS}

# The Krea 2 versions. Each picks one file from the "krea2" and "krea2_te" groups.
VERSIONS = {
    "fp8": {"label": "Standard (FP8)", "krea2": "krea2_turbo_fp8_scaled.safetensors", "krea2_te": "qwen3vl_4b_fp8_scaled.safetensors",
            "who": "Recommended for most graphics cards with 12 GB or more. Near full quality at half the size. "
                   "On 12-16 GB cards part of it waits in normal memory, which is slower but works."},
    "nvfp4": {"label": "Small (NVFP4)", "krea2": "krea2_turbo_nvfp4.safetensors", "krea2_te": "qwen3vl_4b_fp8_scaled.safetensors",
              "who": "Smallest and fastest, slightly lower quality. Only fast on RTX 50-series cards; "
                     "other cards should use Standard."},
    "gguf": {"label": "Low memory (GGUF Q4)", "krea2": "krea2_turbo-Q4_K_M.gguf", "krea2_te": "qwen3vl_4b_fp8_scaled.safetensors",
             "addon": "ComfyUI-GGUF",
             "who": "For 8-11 GB cards such as the RTX 3060 Ti, 3070 or 4060. A compressed Krea 2 (7.5 GB) that fits in the "
                    "card's memory; pictures lose a little fine detail. Needs the free ComfyUI-GGUF add-on, "
                    "which is installed with it."},
    "bf16": {"label": "Full quality (BF16)", "krea2": "krea2_turbo_bf16.safetensors", "krea2_te": "qwen3vl_4b_bf16.safetensors",
             "who": "The original model. Needs 40 GB of graphics memory or more to run well (RTX 6000 Pro class)."},
}
DEFAULT_VERSION = "fp8"


def preferred_version() -> str:
    v = env_config.get("AI_INFLUENCER_KREA_VERSION").lower()
    return v if v in VERSIONS else DEFAULT_VERSION


def gpu_fits(gpu: dict | None) -> bool:
    """Whether this card can run Krea 2 at all (Turing or newer, 8 GB+)."""
    return bool(gpu) and gpu["cc"] >= 7.5 and gpu["vram_gb"] >= 8


def recommend(gpu: dict | None) -> tuple[str, str]:
    """(version, one sentence about this computer) for gpu = {"name", "vram_gb", "cc"} or None (CPU)."""
    if not gpu:
        return "fp8", ("ComfyUI is running without a graphics card. Local image generation would take a very long "
                       "time per picture; use the online workflows (01, 04, 05) or a rented GPU instead.")
    name, vram, cc = gpu["name"], gpu["vram_gb"], gpu["cc"]
    if not gpu_fits(gpu):
        return "fp8", (f"{name} ({vram:.0f} GB) is too weak for Krea 2: expect errors or many minutes per picture. "
                       "The online workflows (01, 04, 05) or a rented GPU are the better choice.")
    if vram < 12:
        return "gguf", (f"{name} ({vram:.0f} GB) can run Krea 2 with the Low memory version: slower than on bigger "
                        "cards, and 16 GB of RAM or more helps.")
    if cc >= 12 and vram < 40:
        return "nvfp4", f"{name} ({vram:.0f} GB) is an RTX 50-series card: the Small version runs fastest on it."
    if vram >= 40:
        return "bf16", f"{name} ({vram:.0f} GB) can run the Full quality version."
    if vram < 16:
        return "fp8", (f"{name} ({vram:.0f} GB) can run the Standard version, but slower: the model is bigger than "
                       "the card's memory, so part of it waits in normal memory (32 GB of RAM or more recommended).")
    return "fp8", f"{name} ({vram:.0f} GB) runs the Standard version well."


def swap_for_version(name: str, version: str) -> str:
    """The file of `version` that stands in for `name` when both are versions of the same model."""
    m = BY_NAME.get(name)
    return VERSIONS[version].get(m.group, name) if m and m.group in ("krea2", "krea2_te") else name


# --- where files are -------------------------------------------------------------------------------------------

def _folder_paths():
    import folder_paths  # ComfyUI's; importable from the repo root
    return folder_paths


def installed(m: Model) -> bool:
    try:
        return _folder_paths().get_full_path(m.folder, m.name) is not None
    except Exception:
        return False


def target_dir(m: Model) -> str:
    return _folder_paths().get_folder_paths(m.folder)[0]


def free_bytes(m: Model) -> int:
    d = target_dir(m)
    os.makedirs(d, exist_ok=True)
    return shutil.disk_usage(d).free


def workflow_models(path: str) -> list[str]:
    """Catalog files named by the switched-on nodes of a saved workflow (bypassed and muted ones don't need them)."""
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    names: list[str] = []
    for node in wf.get("nodes", []):
        if node.get("mode", 0) in (2, 4):
            continue
        values = node.get("widgets_values") or []
        for v in values.values() if isinstance(values, dict) else values:
            if isinstance(v, str) and v in BY_NAME and v not in names:
                names.append(v)
    return names


def describe(name: str) -> dict:
    m = BY_NAME[name]
    return {**asdict(m), "installed": installed(m)}


# --- downloads -------------------------------------------------------------------------------------------------

class Downloads:
    """One background thread per file: <file>.part, resumed after an interruption, renamed when complete."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state: dict[str, dict] = {}  # name -> {done, total, status: running|finished|failed, error}

    def start(self, names: list[str]) -> None:
        for name in names:
            m = BY_NAME[name]
            with self.lock:
                if self.state.get(name, {}).get("status") == "running" or installed(m):
                    continue
                self.state[name] = {"done": 0, "total": m.size, "status": "running", "error": ""}
            threading.Thread(target=self._run, args=(m,), daemon=True).start()

    def snapshot(self) -> dict:
        with self.lock:
            return {k: dict(v) for k, v in self.state.items()}

    def _update(self, name: str, **kw) -> None:
        with self.lock:
            self.state[name].update(kw)

    def _run(self, m: Model) -> None:
        final = os.path.join(target_dir(m), m.name)
        part = final + ".part"
        try:
            if free_bytes(m) < m.size - (os.path.getsize(part) if os.path.exists(part) else 0):
                raise OSError(f"not enough free disk space for {m.size / GB:.1f} GB in {os.path.dirname(final)}")
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(m.url, headers={"User-Agent": "ai-influencer-toolkit"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            token = env_config.get("HF_TOKEN")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored the range: start over
                    have = 0
                self._update(m.name, done=have)
                with open(part, "ab" if have else "wb") as f:
                    while chunk := r.read(4 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        self._update(m.name, done=have)
            if have != m.size:
                raise OSError(f"download stopped at {have / GB:.2f} of {m.size / GB:.2f} GB; start it again to resume")
            os.replace(part, final)
            self._update(m.name, status="finished")
        except Exception as e:
            self._update(m.name, status="failed", error=str(e))


DOWNLOADS = Downloads()


# --- the ComfyUI-GGUF add-on (needed only by the Low memory version) -------------------------------------------

GGUF_ADDON = "ComfyUI-GGUF"
GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"  # city96/ComfyUI-GGUF, pinned: a new version is a deliberate update
GGUF_ZIP = f"https://github.com/city96/ComfyUI-GGUF/archive/{GGUF_COMMIT}.zip"


def addon_dir() -> str:
    try:
        base = _folder_paths().get_folder_paths("custom_nodes")[0]
    except Exception:
        base = os.path.join(env_config.REPO_ROOT, "custom_nodes")
    return os.path.join(base, GGUF_ADDON)


def addon_present() -> bool:
    """Its files are there (it may still need a ComfyUI restart to load)."""
    return os.path.exists(os.path.join(addon_dir(), "nodes.py"))


def install_addon() -> str:
    """Download the pinned ComfyUI-GGUF into custom_nodes/ and pip-install its one requirement (gguf)."""
    import io
    import subprocess
    import sys
    import zipfile

    if not addon_present():
        with urllib.request.urlopen(urllib.request.Request(GGUF_ZIP, headers={"User-Agent": "ai-influencer-toolkit"}),
                                    timeout=120) as r:
            archive = zipfile.ZipFile(io.BytesIO(r.read()))
        tmp = addon_dir() + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        archive.extractall(tmp)
        (top,) = os.listdir(tmp)  # ComfyUI-GGUF-<commit>/
        os.replace(os.path.join(tmp, top), addon_dir())
        shutil.rmtree(tmp, ignore_errors=True)
    out = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "gguf>=0.13.0"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("installing the gguf package failed: " + (out.stderr or out.stdout)[-500:])
    return addon_dir()

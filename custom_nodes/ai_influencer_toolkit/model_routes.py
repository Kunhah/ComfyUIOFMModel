"""Routes for web/model_helper.js, the "Models" window: what this computer can run, which model files a workflow
still needs, the Krea 2 version choice, and downloads with progress.

    GET  /ai_influencer/models            GPU, recommended + chosen Krea 2 version, the catalog with installed flags
    POST /ai_influencer/models/check      {"names": [...]} -> the ones of those that aren't installed
    POST /ai_influencer/models/download   {"names": [...]} -> starts the downloads
    GET  /ai_influencer/models/progress   per file: done / total bytes, running | finished | failed
    POST /ai_influencer/models/version    {"version": "fp8" | "nvfp4" | "gguf" | "bf16"} -> saved to .env
    POST /ai_influencer/models/addon      installs the ComfyUI-GGUF add-on the Low memory version needs (then restart)
"""
from __future__ import annotations

import os
import sys

from aiohttp import web

from server import PromptServer

from . import env_config, model_catalog as cat

routes = PromptServer.instance.routes


def gpu_info() -> dict | None:
    """The card ComfyUI actually uses, or None when it runs on the CPU."""
    try:
        import torch

        import comfy.model_management as mm

        dev = mm.get_torch_device()
        if dev.type != "cuda":
            return None
        p = torch.cuda.get_device_properties(dev)
        return {"name": p.name, "vram_gb": p.total_memory / 1e9, "cc": float(f"{p.major}.{p.minor}")}
    except Exception:
        return None


@routes.get("/ai_influencer/models")
async def models_info(request):
    gpu = gpu_info()
    version, advice = cat.recommend(gpu)
    return web.json_response({
        "gpu": gpu, "recommended": version, "advice": advice, "fits": cat.gpu_fits(gpu), "version": cat.preferred_version(),
        "versions": cat.VERSIONS, "models": [cat.describe(m.name) for m in cat.MODELS],
        "free_gb": cat.free_bytes(cat.MODELS[0]) / cat.GB,
        "addon": {"present": cat.addon_present(), "loaded": gguf_loaded()},
    })


def gguf_loaded() -> bool:
    """ComfyUI-GGUF's nodes are registered, i.e. it was there when ComfyUI started."""
    import nodes

    return "UnetLoaderGGUF" in nodes.NODE_CLASS_MAPPINGS


@routes.post("/ai_influencer/models/addon")
async def models_addon(request):
    try:
        path = await PromptServer.instance.loop.run_in_executor(None, cat.install_addon)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response({"installed": path, "restart": not gguf_loaded()})


@routes.post("/ai_influencer/models/check")
async def models_check(request):
    names = [n for n in (await request.json()).get("names", []) if n in cat.BY_NAME]
    return web.json_response({"missing": [cat.describe(n) for n in names if not cat.installed(cat.BY_NAME[n])]})


@routes.post("/ai_influencer/models/download")
async def models_download(request):
    names = [n for n in (await request.json()).get("names", []) if n in cat.BY_NAME]
    cat.DOWNLOADS.start(names)
    return web.json_response({"started": names})


@routes.get("/ai_influencer/models/progress")
async def models_progress(request):
    return web.json_response(cat.DOWNLOADS.snapshot())


@routes.post("/ai_influencer/models/version")
async def models_version(request):
    version = (await request.json()).get("version", "")
    if version not in cat.VERSIONS:
        return web.json_response({"error": f"unknown version {version!r}"}, status=400)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    from configure import write_env

    write_env({"AI_INFLUENCER_KREA_VERSION": version})
    os.environ["AI_INFLUENCER_KREA_VERSION"] = version  # env_config.get() reads the process environment
    return web.json_response({"version": version})

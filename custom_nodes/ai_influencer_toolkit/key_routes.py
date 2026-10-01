"""Routes for web/key_helper.js: pressing Run on a workflow whose nodes need a key that isn't set asks for it, instead
of the run failing with "FAL_KEY is not set".

    POST /ai_influencer/keys/check   {"types": [node types]} -> {"missing": [{name, help, url}]}
    POST /ai_influencer/keys         {"values": {NAME: value}} -> saved to .env (like tools/configure.py) and used at once
"""
from __future__ import annotations

import os
import sys

from aiohttp import web

from server import PromptServer

from . import env_config

routes = PromptServer.instance.routes

# Node type -> the key it can't run without. Partner API nodes (Krea, GPT Image, Seedance) need no entry here:
# signing in to the Comfy account in the browser is enough for them.
NODE_KEYS = {
    "AIInfluencerTrainCharacterLoRA": "FAL_KEY",
    "AIInfluencerFalLoraImage": "FAL_KEY",
}


@routes.post("/ai_influencer/keys/check")
async def keys_check(request):
    types = (await request.json()).get("types", [])
    names = sorted({NODE_KEYS[t] for t in types if t in NODE_KEYS})
    missing = [env_config.BY_NAME[n] for n in names if not env_config.get(n)]
    return web.json_response({"missing": [{"name": s.name, "help": s.help, "url": s.url} for s in missing]})


@routes.post("/ai_influencer/keys")
async def keys_save(request):
    values = (await request.json()).get("values", {})
    updates = {}
    for name, value in values.items():
        value = str(value).strip()
        if name not in NODE_KEYS.values():
            return web.json_response({"error": f"{name} can't be set here"}, status=400)
        if not value or any(c.isspace() for c in value):
            return web.json_response({"error": f"{name}: paste the key without spaces or line breaks"}, status=400)
        updates[name] = value
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    from configure import write_env

    write_env(updates)
    os.environ.update(updates)  # env_config.get() and fal_client read the process environment
    return web.json_response({"saved": sorted(updates)})

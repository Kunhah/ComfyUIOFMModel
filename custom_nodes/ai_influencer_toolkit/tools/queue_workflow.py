#!/usr/bin/env python3
"""Queue an already API-exported ComfyUI prompt (File -> Export (API) in the ComfyUI web UI,
NOT a regular saved workflow -- the graph format under user/default/workflows/ has visual
layout only and is not what /prompt accepts) against a running ComfyUI server, attaching
COMFY_API_KEY from .env for any partner API nodes in the graph.

Usage:
    python queue_workflow.py path/to/exported_api_prompt.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt_json", help="Path to an API-format prompt exported from ComfyUI's web UI.")
    args = parser.parse_args()

    with open(args.prompt_json, encoding="utf-8") as f:
        prompt = json.load(f)
    if "prompt" not in prompt:
        prompt = {"prompt": prompt}

    api_key = env_config.get("COMFY_API_KEY")
    if api_key:
        prompt.setdefault("extra_data", {})["api_key_comfy_org"] = api_key
    else:
        print("Warning: COMFY_API_KEY not set (tools/configure.py) -- partner API nodes will fail unless "
              "you're signed in to a Comfy account in the browser this server was opened from.",
              file=sys.stderr)

    server_url = env_config.get("COMFY_SERVER_URL").rstrip("/")
    data = json.dumps(prompt).encode("utf-8")
    req = urllib.request.Request(f"{server_url}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"Server rejected the prompt ({e.code}): {e.read().decode()}", file=sys.stderr)
        return 1

    print(f"Queued. prompt_id={result.get('prompt_id')}")
    if result.get("node_errors"):
        print("node_errors:", json.dumps(result["node_errors"], indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/bin/sh
# Get the newest version of this ComfyUI fork:  ./update_comfyui.sh   (close ComfyUI first)
# A Git checkout is updated with git pull; a downloaded ZIP folder downloads the new files.
# Your venv, models, input, output and .env are kept. All the real work is in tools/update.py.
set -e
cd "$(dirname "$0")"
PY=venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" custom_nodes/ai_influencer_toolkit/tools/update.py "$@"

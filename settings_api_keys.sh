#!/bin/sh
# Set the API keys and settings in .env (never shared): a window if Tk is installed, else questions in the terminal.
set -e
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
    echo "Run ./start_comfyui.sh once first: it installs what this needs."
    exit 1
fi
exec venv/bin/python custom_nodes/ai_influencer_toolkit/tools/launcher.py --keys

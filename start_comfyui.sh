#!/bin/sh
# Start ComfyUI on Linux:  ./start_comfyui.sh   (or: sh start_comfyui.sh)
# The first run creates ./venv and installs what's missing (a few GB, 10-30 min); later runs start right away.
#   ./start_comfyui.sh --reinstall          redo the installs, PyTorch too
#   ./start_comfyui.sh -- --port 8190       extra arguments for main.py go after --
# All the real work is in custom_nodes/ai_influencer_toolkit/tools/launcher.py.
set -e
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
    PY=""
    for candidate in python3.13 python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
    done
    if [ -z "$PY" ]; then
        echo "Python 3 is not installed. Install it with your package manager, for example:"
        echo "  Ubuntu/Debian:  sudo apt install python3 python3-venv python3-tk"
        echo "  Fedora:         sudo dnf install python3 python3-tkinter"
        echo "  Arch:           sudo pacman -S python tk"
        echo "then run $0 again."
        exit 1
    fi
    echo "Creating the private Python folder (venv) with $PY..."
    if ! "$PY" -m venv venv; then
        rm -rf venv
        echo
        echo "Creating the venv failed. On Ubuntu/Debian install it with:  sudo apt install python3-venv"
        echo "then run $0 again."
        exit 1
    fi
fi

exec venv/bin/python custom_nodes/ai_influencer_toolkit/tools/launcher.py "$@"

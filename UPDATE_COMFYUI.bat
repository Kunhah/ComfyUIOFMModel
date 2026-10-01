@echo off
rem Double-click to get the newest version of this ComfyUI fork (close ComfyUI first).
rem Your venv, models, input, output and .env are kept. All the real work is in
rem custom_nodes\ai_influencer_toolkit\tools\update.py.
setlocal
cd /d "%~dp0"
title Update ComfyUI
set "PY="
if exist "venv\Scripts\python.exe" set "PY="venv\Scripts\python.exe""
if not defined PY py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY (
    echo Python was not found. Double-click START_COMFYUI.bat once first: it installs it.
    pause
    exit /b 1
)
rem One line on purpose: this file may be replaced by the update while it runs.
%PY% custom_nodes\ai_influencer_toolkit\tools\update.py %* & pause & exit /b

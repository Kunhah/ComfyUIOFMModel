@echo off
rem Double-click to open the window for API keys and settings (saved in .env, which is never shared).
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Run START_COMFYUI.bat once first: it installs what this window needs.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" custom_nodes\ai_influencer_toolkit\tools\configure_gui.py
if errorlevel 1 pause

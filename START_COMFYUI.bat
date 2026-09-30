@echo off
rem Double-click to start ComfyUI. The first run installs Python packages into .\venv (a few GB, 10-30 min).
rem All the real work is in custom_nodes\ai_influencer_toolkit\tools\launcher.py.
setlocal
cd /d "%~dp0"
title ComfyUI

if exist "venv\Scripts\python.exe" goto run

echo Looking for Python...
call :find_python
if defined PY goto make_venv

echo.
echo Python is not installed on this computer. It is needed to run ComfyUI.
choice /c YN /m "Install Python 3.13 now (free, from python.org via Windows' winget)"
if errorlevel 2 goto no_python
winget install -e --id Python.Python.3.13 --scope user --accept-source-agreements --accept-package-agreements
call :find_python
if defined PY goto make_venv

:no_python
echo.
echo Could not find Python. Install Python 3.13 from https://www.python.org/downloads/windows/
echo (tick "Add python.exe to PATH" in the installer), then double-click START_COMFYUI.bat again.
pause
exit /b 1

:make_venv
echo Using %PY%
echo Creating the private Python folder (venv)...
%PY% -m venv venv
if not exist "venv\Scripts\python.exe" (
    echo Creating the venv failed. See the message above.
    pause
    exit /b 1
)

:run
"venv\Scripts\python.exe" custom_nodes\ai_influencer_toolkit\tools\launcher.py %*
echo.
echo ComfyUI has stopped. If that was not on purpose, the reason is in the text above.
pause
exit /b

rem Sets PY to a working Python 3.12/3.13 (preferred by ComfyUI), else any Python 3 found.
:find_python
set "PY="
for %%V in (3.13 3.12) do (
    if not defined PY py -%%V --version >nul 2>&1 && set "PY=py -%%V"
)
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python313\python.exe""
if not defined PY py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"
exit /b 0

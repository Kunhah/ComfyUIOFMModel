#!/usr/bin/env python3
"""What UPDATE_COMFYUI.bat (Windows) and update_comfyui.sh (Linux) run: get the newest version of this fork.

    python custom_nodes/ai_influencer_toolkit/tools/update.py

A folder cloned with Git is updated with `git pull --ff-only` (it stops instead when files were edited locally).
A folder downloaded as a ZIP downloads the newest ZIP of AI_INFLUENCER_UPDATE_REPO / AI_INFLUENCER_UPDATE_BRANCH
from GitHub and writes the files that changed over this folder. Nothing outside the ZIP is touched: venv, models,
input, output, .env and your own workflows stay as they are. A workflow that ships with the fork and was edited
here is replaced, so save your changes under a new name. Files removed upstream are left in place.
GITHUB_TOKEN is only needed while the repo is private.

Only the standard library is used, so it runs before the venv exists too. New Python packages are installed by
START_COMFYUI.bat / start_comfyui.sh the next time ComfyUI starts.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_config  # noqa: E402

ROOT = env_config.REPO_ROOT
STAMP = os.path.join(ROOT, ".update_version")  # commit of the last ZIP update; gitignored
START = "START_COMFYUI.bat" if os.name == "nt" else "./start_comfyui.sh"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def git_update() -> int:
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        say("Files that came with the fork were changed in this folder, so nothing was updated:")
        say(git("status", "--short", "--untracked-files=no").stdout)
        say("Commit or undo those changes (git stash), then run the update again.")
        return 1
    say("Updating with git pull...")
    result = git("pull", "--ff-only")
    say((result.stdout + result.stderr).strip())
    if result.returncode:
        say("\ngit pull failed; the reason is above. Nothing was changed.")
    return result.returncode


def github(path: str, accept: str) -> bytes:
    repo = env_config.get("AI_INFLUENCER_UPDATE_REPO")
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/{path}",
                                 headers={"Accept": accept, "User-Agent": "ai-influencer-update"})
    token = env_config.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            hint = ("The GITHUB_TOKEN in the settings was refused: make a new one." if token else
                    "If the repo is private, ask its owner for access and put a GitHub token in GITHUB_TOKEN "
                    "(SETTINGS_API_KEYS.bat, or settings_api_keys.sh on Linux).")
            raise SystemExit(f"GitHub answered {e.code} for {repo}. {hint}") from None
        raise


def zip_update() -> int:
    branch = env_config.get("AI_INFLUENCER_UPDATE_BRANCH")
    say(f"Checking {env_config.get('AI_INFLUENCER_UPDATE_REPO')} ({branch}) for a new version...")
    sha = github(f"commits/{branch}", "application/vnd.github.sha").decode().strip()
    try:
        with open(STAMP, encoding="utf-8") as f:
            if f.read().strip() == sha:
                say("You already have the newest version.")
                return 0
    except FileNotFoundError:
        pass

    say("Downloading the new version...")
    archive = zipfile.ZipFile(io.BytesIO(github(f"zipball/{sha}", "application/vnd.github+json")))
    changed = []
    for info in archive.infolist():
        rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""  # drop the "<owner>-<repo>-<sha>/" folder
        if not rel or info.is_dir():
            continue
        dest = os.path.abspath(os.path.join(ROOT, rel))
        if os.path.commonpath([dest, ROOT]) != ROOT:
            continue
        data = archive.read(info)
        try:
            with open(dest, "rb") as f:
                if f.read() == data:
                    continue
        except FileNotFoundError:
            pass
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".update-tmp", "wb") as f:
            f.write(data)
        os.replace(dest + ".update-tmp", dest)
        if os.name != "nt" and (info.external_attr >> 16) & 0o111:
            os.chmod(dest, 0o755)  # keep the .sh scripts runnable
        changed.append(rel)
    with open(STAMP, "w", encoding="utf-8") as f:
        f.write(sha + "\n")
    say(f"Updated {len(changed)} file(s) to version {sha[:7]}.")
    for rel in changed[:30]:
        say(f"  {rel}")
    if len(changed) > 30:
        say(f"  ... and {len(changed) - 30} more")
    return 0


def main() -> int:
    say("Close ComfyUI before updating (the black window that runs it).\n")
    if os.path.isdir(os.path.join(ROOT, ".git")) and shutil.which("git"):
        code = git_update()
    else:
        try:
            code = zip_update()
        except SystemExit as e:
            say(str(e))
            return 1
        except (OSError, zipfile.BadZipFile) as e:
            say(f"The update failed: {e}\nCheck the internet connection and try again.")
            return 1
    if code == 0:
        say(f"\nDone. Start ComfyUI with {START}; new Python packages, if any, are installed then.")
    return code


if __name__ == "__main__":
    sys.exit(main())

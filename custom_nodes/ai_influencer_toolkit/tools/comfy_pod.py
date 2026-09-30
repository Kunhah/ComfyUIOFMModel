#!/usr/bin/env python3
"""Rent a vast.ai GPU running ComfyUI with this project's setup on it: the node pack, workflows
01-11, character.json, the dataset, the Krea 2 models and your trained LoRA. Unlike vast_pod.py
(which trains and exits), this pod stays up for you to use in the browser.

Everything it runs is local open weights, so nothing generated on the pod is sent to a provider
for moderation; add --with-minimax for the MiniMax H3 video models (Workflow 11).

Three ways it can stop billing, so a forgotten pod can't drain the balance:
  1. `destroy` from here;
  2. a hard timer inside the pod (--hours), which fires even with your PC off;
  3. `watch --idle-minutes N`, which destroys the pod after N minutes with no new image written.

    python comfy_pod.py launch --hours 2          # rent + set up, prints the ComfyUI URL
    python comfy_pod.py launch --with-minimax    # ... plus the MiniMax H3 video models (~45 GB)
    python comfy_pod.py launch --with-upscale    # ... plus SeedVR2 (~4 GB), for the upscale in Workflows 08/10
    python comfy_pod.py url | status | destroy
    python comfy_pod.py watch --idle-minutes 25   # auto-destroy when you stop generating
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import env_config  # noqa: E402
import gpu_select  # noqa: E402
import logging_util  # noqa: E402
from vast_pod import destroy_instance, instance_info, load_state, save_state, search, vast  # noqa: E402

IMAGE = "vastai/comfy:v0.35.0-cuda-12.9-py312"  # matches the local ComfyUI version; pinned on purpose
STATE_KEY = "comfy_instance_id"

ONSTART = r"""#!/bin/bash
mkdir -p /var/podlogs
exec > >(tee -a /var/podlogs/onstart.log) 2>&1
set -x
destroy() { curl -s -X DELETE -H "Authorization: Bearer $VAST_API_KEY" \
  "https://console.vast.ai/api/v0/instances/$CONTAINER_ID" ; }
# MAX_HOURS is usually fractional ("2.0"), which bash $((...)) can't multiply, so awk does it
LIMIT_S=$(awk -v h="${MAX_HOURS:-2}" 'BEGIN{s=int(h*3600); print (s>0 ? s : 7200)}')
echo "WATCHDOG: armed for $LIMIT_S s"
( sleep "$LIMIT_S"; echo "WATCHDOG: hard time limit"; destroy ) &

COMFY=$(dirname "$(find / -maxdepth 6 -name main.py -path '*[Cc]omfy*' 2>/dev/null | head -1)")
[ -d "$COMFY" ] || { echo "FATAL: ComfyUI not found in image"; destroy; exit 1; }
echo "COMFY_DIR=$COMFY"
HFBASE="https://huggingface.co/datasets/$HF_REPO/resolve/main"
set +x  # keep the token out of the log
curl -fLsS --retry 8 --retry-delay 5 --retry-all-errors -H "Authorization: Bearer $HF_TOKEN" "$HFBASE/pod_kit.tar.gz" -o /tmp/kit.tgz || { echo "FATAL: kit download"; destroy; exit 1; }
mkdir -p "$COMFY/models/loras/$PROJECT"
LORA_OUT="$COMFY/models/loras/$PROJECT/$(basename $LORA_PATH)"
for attempt in 1 2 3; do  # resumable, into .part: a half-downloaded LoRA must never look like a real one
  curl -fLsS --retry 8 --retry-delay 5 --retry-all-errors -C - -H "Authorization: Bearer $HF_TOKEN" \
    "$HFBASE/$LORA_PATH" -o "$LORA_OUT.part" && mv "$LORA_OUT.part" "$LORA_OUT" && break
  sleep 10
done
[ -s "$LORA_OUT" ] && echo "LORA_OK" || echo "WARN: lora download failed"
set -x
tar xzf /tmp/kit.tgz -C /tmp
COMFY="$COMFY" PROJECT="$PROJECT" WITH_UPSCALE="${WITH_UPSCALE:-0}" WITH_MINIMAX="${WITH_MINIMAX:-0}" bash /tmp/kit/pod_bootstrap.sh || echo "WARN: bootstrap had errors"

# custom nodes only load at startup, so restart ComfyUI -- and if nothing supervises it
# (this image has no supervisor), start it ourselves or the pod comes up with no web UI.
(supervisorctl restart comfyui || supervisorctl restart all) 2>/dev/null || true
sleep 5
if ! ss -ltn 2>/dev/null | grep -q ':8188'; then
  pkill -f "main.py" 2>/dev/null || true
  sleep 3
  PY=$([ -x /venv/main/bin/python ] && echo /venv/main/bin/python || echo python3)
  cd "$COMFY" && setsid nohup $PY main.py --listen 0.0.0.0 --port 8188 > /var/podlogs/comfy.log 2>&1 < /dev/null &
  sleep 20
fi
ss -ltn 2>/dev/null | grep ':8188' && echo "COMFYUI_LISTENING" || echo "WARN: ComfyUI not listening"
echo "SETUP_DONE"
"""


def comfy_url(info: dict) -> str | None:
    ip = info.get("public_ipaddr") or info.get("ssh_host")
    ports = info.get("ports") or {}
    mapped = ports.get("8188/tcp") or ports.get("8188/tcp ")
    if ip and mapped:
        return f"http://{str(ip).strip()}:{mapped[0]['HostPort']}"
    return None


def profile_of(args) -> gpu_select.Profile:
    return gpu_select.PROFILES["minimax" if args.with_minimax else "krea2"]


def cmd_search(args) -> int:
    profile = profile_of(args)
    offers = search(args, profile)
    print(f"{profile.label}: {len(offers)} offers qualify, best estimated total first")
    for o in offers[: args.limit]:
        print(gpu_select.describe(o))
    return 0


def cmd_launch(args) -> int:
    project = logging_util.sanitize_project_name(args.project)
    env_config.require_cli("VAST_API_KEY", "HF_TOKEN")
    if not args.lora_path:
        raise SystemExit("No character LoRA to load: pass --lora-path or set AI_INFLUENCER_LORA_PATH "
                         "(python custom_nodes/ai_influencer_toolkit/tools/configure.py).")
    args.hf_repo = args.hf_repo or env_config.hf_repo(project)
    offers = search(args, profile_of(args))
    if not offers:
        raise SystemExit("No offers matched. Loosen --max-price / --gpu / --min-inet.")
    for o in offers[:3]:
        print("  candidate " + gpu_select.describe(o))
    offer = next((o for o in offers if o["id"] == args.offer), offers[0]) if args.offer else offers[0]
    hourly = offer["dph_total"]
    print(f"offer {offer['id']}: {offer['gpu_name']} {offer['gpu_ram']/1024:.0f}GB ${hourly:.3f}/hr, "
          f"net {offer.get('inet_down', 0):.0f}Mbps")
    print(f"max spend at --hours {args.hours}: ${hourly * args.hours:.2f} (pod destroys itself then)")
    if args.dry_run:
        return 0
    if not args.yes and input(f"Rent it (up to ${hourly * args.hours:.2f})? Type yes: ").strip().lower() != "yes":
        return 1

    # The vastai/comfy image puts ComfyUI behind a Caddy proxy on 8188 that wants HTTPS and a password.
    # Its own token can't be read from outside, but a WEB_PASSWORD set at launch works as the password
    # (browser: user "vastai") and as an "Authorization: Bearer" token for scripts.
    token = secrets.token_urlsafe(24)
    env = " ".join([
        "-p 8188:8188",
        f"-e WEB_PASSWORD={token}",
        f"-e WITH_MINIMAX={1 if args.with_minimax else 0}",
        f"-e WITH_UPSCALE={1 if args.with_upscale else 0}",
        f"-e VAST_API_KEY={os.environ['VAST_API_KEY']}",
        f"-e HF_TOKEN={os.environ['HF_TOKEN']}",
        f"-e HF_REPO={args.hf_repo}",
        f"-e LORA_PATH={args.lora_path}",
        f"-e PROJECT={project}",
        f"-e MAX_HOURS={args.hours}",
    ])
    res = vast("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(args.disk),
               "--env", env, "--onstart-cmd", ONSTART, "--ssh", "--direct", "--label", f"comfy-{project}")
    iid = (res or {}).get("new_contract") if isinstance(res, dict) else None
    if not iid:
        raise SystemExit(f"launch failed: {res!r}")
    st = load_state(project)
    st.update({STATE_KEY: iid, "comfy_started": time.time(), "comfy_dph": hourly, "comfy_hours": args.hours,
               "comfy_token": token})
    save_state(project, st)
    print(f"instance {iid} starting. Setup takes ~5-10 min (models). Then:\n"
          f"  python comfy_pod.py url --project {project}")
    return 0


def _iid(project: str) -> int:
    st = load_state(project)
    iid = st.get(STATE_KEY)
    if not iid:
        raise SystemExit("no ComfyUI pod recorded; launch one first")
    return iid


def cmd_url(args) -> int:
    project = logging_util.sanitize_project_name(args.project)
    info = instance_info(_iid(project))
    if info is None:
        print("instance is gone")
        return 1
    url = comfy_url(info)
    print(f"status: {info.get('actual_status')}")
    if url:
        token = load_state(project).get("comfy_token")
        print(f"ComfyUI: {url.replace('http://', 'https://')}  (self-signed certificate: accept the browser warning)")
        if token:
            print(f"login: user vastai, password {token}")
    else:
        print("port 8188 not mapped yet; try again in a minute")
    return 0


def cmd_status(args) -> int:
    project = logging_util.sanitize_project_name(args.project)
    st = load_state(project)
    iid = _iid(project)
    info = instance_info(iid)
    hrs = (time.time() - st["comfy_started"]) / 3600
    print(f"instance {iid}: {'gone' if info is None else info.get('actual_status')}  "
          f"{hrs:.2f}h  ~${hrs * st['comfy_dph']:.2f} spent (limit {st['comfy_hours']}h)")
    return 0


def cmd_destroy(args) -> int:
    project = logging_util.sanitize_project_name(args.project)
    print(destroy_instance(_iid(project)))
    return 0


def cmd_watch(args) -> int:
    """Destroys the pod once it has been idle (no new generated image) for --idle-minutes."""
    project = logging_util.sanitize_project_name(args.project)
    st = load_state(project)
    iid = _iid(project)
    last_change, last_count = time.time(), -1
    while True:
        info = instance_info(iid)
        if info is None:
            print("instance gone")
            break
        hrs = (time.time() - st["comfy_started"]) / 3600
        try:
            out = str(vast("execute", str(iid), "ls /opt/ComfyUI/output -R | wc -l", raw=False))
            count = int("".join(c for c in out if c.isdigit()) or -1)
        except SystemExit:
            count = last_count
        if count != last_count:
            last_change, last_count = time.time(), count
        idle_min = (time.time() - last_change) / 60
        print(f"[{hrs:.2f}h ~${hrs * st['comfy_dph']:.2f}] {info.get('actual_status')} idle {idle_min:.0f}m", flush=True)
        if idle_min > args.idle_minutes:
            print(f"idle {args.idle_minutes}m -> destroying")
            print(destroy_instance(iid))
            break
        if hrs > st["comfy_hours"]:
            print("time limit -> destroying")
            print(destroy_instance(iid))
            break
        time.sleep(args.interval)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["search", "launch", "url", "status", "destroy", "watch"])
    ap.add_argument("--project", default=env_config.get("AI_INFLUENCER_DEFAULT_PROJECT"))
    gpu_select.add_arguments(ap)
    ap.add_argument("--offer", type=int)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--hf-repo", default=env_config.get("AI_INFLUENCER_HF_REPO"),
                    help="Private HF dataset repo (default: AI_INFLUENCER_HF_REPO, else <user>/<project>-lora-transfer).")
    ap.add_argument("--lora-path", default=env_config.get("AI_INFLUENCER_LORA_PATH"),
                    help="Character LoRA inside the HF repo (default: AI_INFLUENCER_LORA_PATH).")
    ap.add_argument("--idle-minutes", type=int, default=25)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--with-minimax", action="store_true",
                    help="Also install the MiniMax H3 models for Workflow 11 (~45 GB); the GPU search then "
                         "only considers Blackwell cards with 32 GB+, since H3's text encoder ships in nvfp4.")
    ap.add_argument("--with-upscale", action="store_true",
                    help="Also download SeedVR2 (~4 GB) for the upscale in Workflows 08/10 and held Pod queue jobs that use it.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    # Krea 2 alone fits a 24 GB Ampere card; MiniMax H3 needs Blackwell (see gpu_select.py).
    gpu_select.apply_defaults(args, profile_of(args))

    env_config.load_env()  # the commands below read VAST_API_KEY / HF_TOKEN from os.environ
    return {"search": cmd_search, "launch": cmd_launch, "url": cmd_url, "status": cmd_status,
            "destroy": cmd_destroy, "watch": cmd_watch}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Rent a vast.ai GPU, train the character LoRA on it unattended, and destroy it automatically.

Two independent safety nets, because a forgotten instance bills by the hour:
  1. this script's `watch` loop destroys the instance when training finishes or --hours elapses;
  2. the pod's own onstart script destroys itself (it knows its CONTAINER_ID) when training ends,
     and runs a background timer that destroys it after --hours even if training hangs.
Net 2 works even with this machine switched off.

Moving files (--transfer): by default (ssh) `watch` copies the dataset straight onto the pod and the
results (the last checkpoints + the sample images) back off it, so the dataset only ever exists on
this PC and the rented machine. The finished pod waits --pull-grace minutes for that download before
destroying itself, so keep `watch` running. --transfer hf moves both through your private HF repo
instead, which works with this machine off but puts the dataset under Hugging Face's content policy.

    python vast_pod.py search
    python vast_pod.py launch --dry-run                 # show the plan, rent nothing
    python vast_pod.py launch --hours 4                 # rent + start training
    python vast_pod.py watch                            # follow, download results, destroy
    python vast_pod.py status | logs | destroy

Needs VAST_API_KEY and HF_TOKEN (set them with tools/configure.py); HF_TOKEN because Krea 2 is gated.
--transfer ssh also needs ssh/scp; it makes ~/.ssh/ai_influencer_vast and attaches it to each pod only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import env_config  # noqa: E402
import gpu_select  # noqa: E402
import logging_util  # noqa: E402

IMAGE = "vastai/ostris-ai-toolkit:ef8110c-2026-09-17-cuda-12.9"
"""Pinned: this repo has no :latest tag, and an unpullable image leaves the instance stuck in
"loading" while still billing. Check https://hub.docker.com/v2/repositories/vastai/ostris-ai-toolkit/tags
for a newer date tag."""
STATE = "vast_run.json"

ONSTART = r"""#!/bin/bash
mkdir -p /var/podlogs
exec > >(tee -a /var/podlogs/onstart.log) 2>&1
set -x
destroy() { curl -fsSL -X DELETE -H "Authorization: Bearer $VAST_API_KEY" \
  "https://console.vast.ai/api/v0/instances/$CONTAINER_ID" ; }
# net 2a: hard timer, fires even if training hangs or the renter's machine is off
# MAX_HOURS is usually fractional ("2.0"), which bash $((...)) can't multiply, so awk does it
LIMIT_S=$(awk -v h="${MAX_HOURS:-2}" 'BEGIN{s=int(h*3600); print (s>0 ? s : 7200)}')
echo "WATCHDOG: armed for $LIMIT_S s"
( sleep "$LIMIT_S"; echo "WATCHDOG: time limit reached"; destroy ) &

RUNPY=$(find / -name run.py -path "*ai-toolkit*" 2>/dev/null | head -1)
[ -z "$RUNPY" ] && { echo "FATAL: ai-toolkit not found in image"; destroy; exit 1; }
AT=$(dirname "$RUNPY"); cd "$AT"
for v in /venv/main/bin/activate venv/bin/activate /opt/venv/bin/activate; do
  [ -f "$v" ] && . "$v" && break
done
export HF_HOME=/workspace/hf
pip install -q huggingface_hub
mkdir -p datasets/$PROJECT config
#@FETCH@
unzip -qo /tmp/ds.zip -d datasets/$PROJECT
# the config's dataset path is written for a different image layout; point it at where we actually unzipped
sed -i "s|^\( *-\? *folder_path:\).*|\1 $AT/datasets/$PROJECT|" config/job.yaml
grep -n "folder_path" config/job.yaml
ls "$AT/datasets/$PROJECT" | head -3
[ "$(ls "$AT/datasets/$PROJECT"/*.txt 2>/dev/null | wc -l)" -ge 10 ] || { echo "FATAL: dataset missing/empty at $AT/datasets/$PROJECT"; destroy; exit 1; }
set -x
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FATAL: no usable CUDA device"; destroy; exit 1; }
echo "TRAINING_START $(date -Is)"
python "$RUNPY" config/job.yaml 2>&1 | tee /var/podlogs/train.log
echo "TRAINING_EXIT=${PIPESTATUS[0]}"

tar czf /tmp/samples.tgz output/*/samples 2>/dev/null
#@DELIVER@
[ "$AUTO_DESTROY" = "1" ] && destroy
"""

FETCH_HF = r"""HFBASE="https://huggingface.co/datasets/$HF_REPO/resolve/main"
set +x  # keep the token out of the instance log (the host can read it)
curl -fL -H "Authorization: Bearer $HF_TOKEN" "$HFBASE/dataset.zip" -o /tmp/ds.zip || { echo "FATAL: dataset download"; destroy; exit 1; }
curl -fL -H "Authorization: Bearer $HF_TOKEN" "$HFBASE/job.yaml" -o config/job.yaml || { echo "FATAL: config download"; destroy; exit 1; }
set -x"""

# `watch` on the renter's machine pushes the dataset in and pulls results out over SSH, so the
# dataset never leaves this PC and the rented machine. Both waits are bounded; the hard timer applies too.
FETCH_SSH = r"""IN=/workspace/incoming; mkdir -p $IN
for i in $(seq 1 $((${WAIT_MIN:-30}*6))); do [ -f $IN/READY ] && break; sleep 10; done
[ -f $IN/READY ] || { echo "FATAL: no dataset arrived over SSH (is vast_pod.py watch running?)"; destroy; exit 1; }
cp $IN/dataset.zip /tmp/ds.zip && cp $IN/job.yaml config/job.yaml || { echo "FATAL: incoming files"; destroy; exit 1; }"""

DELIVER_SSH = r"""OUT=/workspace/out; mkdir -p $OUT
cp /var/podlogs/onstart.log /var/podlogs/train.log /tmp/samples.tgz $OUT/ 2>/dev/null
ls output/*/*.safetensors 2>/dev/null | LC_ALL=C sort | tail -n "${KEEP_CHECKPOINTS:-6}" | xargs -r cp -t $OUT/
touch $OUT/DONE
echo "RESULTS_READY"
for i in $(seq 1 $((${PULL_GRACE_MIN:-60}*6))); do [ -f $OUT/PULLED ] && break; sleep 10; done"""

DELIVER_HF = r"""python - <<'PYINNER'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["HF_REPO"]
keep = int(os.environ.get("KEEP_CHECKPOINTS", "6"))
files = (["/var/podlogs/onstart.log", "/var/podlogs/train.log"]           # logs first: a crash must stay diagnosable
         + sorted(glob.glob("output/*/*.safetensors"))[-keep:] + ["/tmp/samples.tgz"])
for f in files:
    if os.path.exists(f):
        try:
            api.upload_file(path_or_fileobj=f,
                            path_in_repo=f"results/{os.environ['CONTAINER_ID']}/" + os.path.basename(f),
                            repo_id=repo, repo_type="dataset")
            print("UPLOADED", f, flush=True)
        except Exception as e:
            print("UPLOAD_FAILED", f, e, flush=True)
api.upload_file(path_or_fileobj=b"done", path_in_repo=f"results/{os.environ['CONTAINER_ID']}/DONE",
                repo_id=repo, repo_type="dataset")
PYINNER
echo "RESULTS_UPLOADED"
"""


def onstart(transfer: str) -> str:
    fetch, deliver = (FETCH_SSH, DELIVER_SSH) if transfer == "ssh" else (FETCH_HF, DELIVER_HF)
    return ONSTART.replace("#@FETCH@", fetch).replace("#@DELIVER@", deliver)


SSH_KEY = os.path.expanduser("~/.ssh/ai_influencer_vast")
"""A key just for training pods: attached to each instance, never added to the vast.ai account."""


def ensure_ssh_key() -> str:
    if not os.path.isfile(SSH_KEY):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "ai-influencer-vast", "-f", SSH_KEY], check=True)
    with open(SSH_KEY + ".pub", encoding="utf-8") as f:
        return f.read().strip()


def ssh_target(instance_id: int) -> tuple[str, int] | None:
    from urllib.parse import urlparse

    try:
        url = urlparse(str(vast("ssh-url", str(instance_id), raw=False)).strip())
    except SystemExit:
        return None
    return (f"{url.username or 'root'}@{url.hostname}", url.port or 22) if url.hostname else None


def ssh_opts(project: str, instance_id: int) -> list[str]:
    # per-instance known_hosts: vast reuses host:port pairs across instances
    known = os.path.join(logging_util.get_project_output_dir(project), "pod", f"known_hosts_{instance_id}")
    return ["-i", SSH_KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known}"]


def ssh(project: str, instance_id: int, target: tuple[str, int], command: str) -> bool:
    return subprocess.run(["ssh", *ssh_opts(project, instance_id), "-p", str(target[1]), target[0], command],
                          capture_output=True).returncode == 0


def scp(project: str, instance_id: int, target: tuple[str, int], sources: list[str], dest: str) -> bool:
    return subprocess.run(["scp", *ssh_opts(project, instance_id), "-P", str(target[1]), *sources, dest]).returncode == 0


def vastai_bin() -> str:
    """The vastai CLI installed next to this Python (the venv's bin/, which isn't on PATH when ComfyUI
    was started as venv/bin/python without activating the venv), else whatever PATH has."""
    here = os.path.join(os.path.dirname(sys.executable), "vastai")
    found = here if os.access(here, os.X_OK) else shutil.which("vastai")
    if not found:
        raise SystemExit("The vastai CLI isn't installed: pip install vastai (in ComfyUI's venv).")
    return found


def vast(*args: str, raw: bool = True) -> object:
    key = env_config.get("VAST_API_KEY")
    cmd = [vastai_bin(), *args] + (["--raw"] if raw else []) + (["--api-key", key] if key else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"vastai {' '.join(args)} failed:\n{p.stderr or p.stdout}")
    if not raw:
        return p.stdout
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return p.stdout


def state_path(project: str) -> str:
    return os.path.join(logging_util.get_project_output_dir(project), "pod", STATE)


def load_state(project: str) -> dict:
    p = state_path(project)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}


def save_state(project: str, data: dict) -> None:
    """Refuses to overwrite a newer run: an exiting watcher used to clobber the state file of the
    run that had just been launched."""
    p = state_path(project)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    current = load_state(project)
    if current and current.get("instance_id", 0) > data.get("instance_id", 0):
        return
    json.dump(data, open(p, "w", encoding="utf-8"), indent=2)


def search(args, profile: gpu_select.Profile, work_hours: float | None = None) -> list[dict]:
    """Every offer that meets the job's requirements, cheapest estimated total first (gpu_select.py)."""
    offers = vast("search", "offers", gpu_select.query(args), "-o", "dph_total", "--limit", "300")
    return gpu_select.rank(offers if isinstance(offers, list) else [], profile, work_hours)


def cmd_search(args) -> int:
    offers = search(args, gpu_select.PROFILES["train"], args.est_train_hours)
    print(f"{gpu_select.PROFILES['train'].label}: {len(offers)} offers qualify, best estimated total first")
    for o in offers[: args.limit]:
        print(gpu_select.describe(o))
    return 0


def cmd_launch(args) -> int:
    project = logging_util.sanitize_project_name(args.project)
    pod_dir = os.path.join(logging_util.get_project_output_dir(project), "pod")
    def find_one(pattern: str) -> str:
        hits = sorted(glob.glob(os.path.join(pod_dir, pattern)))
        if len(hits) != 1:
            raise SystemExit(f"expected exactly one {pattern} in {pod_dir}, found {len(hits)}")
        return hits[0]

    dataset_zip = args.dataset or find_one("*_dataset.zip")
    config_yaml = args.config or find_one("*.yaml")

    offers = search(args, gpu_select.PROFILES["train"], args.est_train_hours)
    if not offers:
        raise SystemExit("No offers matched. Loosen --max-price / --min-inet / --gpu.")
    for o in offers[:3]:
        print("  candidate " + gpu_select.describe(o))
    offer = next((o for o in offers if o["id"] == args.offer), offers[0]) if args.offer else offers[0]
    hourly = offer["dph_total"]
    print(f"offer {offer['id']}: {offer['gpu_name']} ${hourly:.3f}/hr, net {offer.get('inet_down', 0):.0f}Mbps, "
          f"disk {offer.get('disk_space', 0):.0f}GB")
    print(f"max spend at --hours {args.hours}: ${hourly * args.hours:.2f} (destroyed automatically at that point)")
    print(f"dataset: {os.path.basename(dataset_zip)}  config: {os.path.basename(config_yaml)}")
    if args.dry_run:
        return 0
    env_config.require_cli("VAST_API_KEY", "HF_TOKEN")
    if not args.yes and input(f"Rent it (up to ${hourly * args.hours:.2f})? Type yes: ").strip().lower() != "yes":
        return 1

    repo = ""
    if args.transfer == "hf":
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        repo = args.hf_repo or env_config.hf_repo(project)
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"uploading dataset + config to private HF repo {repo} ...")
        api.upload_file(path_or_fileobj=dataset_zip, path_in_repo="dataset.zip", repo_id=repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=config_yaml, path_in_repo="job.yaml", repo_id=repo, repo_type="dataset")
    else:
        pubkey = ensure_ssh_key()

    env = " ".join([
        f"-e VAST_API_KEY={os.environ['VAST_API_KEY']}",
        f"-e HF_TOKEN={os.environ['HF_TOKEN']}",  # still needed: the Krea 2 base model is gated
        f"-e HF_REPO={repo}",
        f"-e PROJECT={project}", f"-e MAX_HOURS={args.hours}",
        f"-e KEEP_CHECKPOINTS={args.keep_checkpoints}",
        f"-e AUTO_DESTROY={0 if args.no_auto_destroy else 1}",
        f"-e WAIT_MIN={args.loading_timeout}", f"-e PULL_GRACE_MIN={args.pull_grace}",
    ])
    res = vast("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(args.disk),
               "--env", env, "--onstart-cmd", onstart(args.transfer),
               "--ssh", "--direct", "--label", f"lora-{project}")
    instance_id = (res or {}).get("new_contract") if isinstance(res, dict) else None
    if not instance_id:
        raise SystemExit(f"launch failed: {res!r}")
    state = {"instance_id": instance_id, "offer": offer["id"], "dph": hourly, "started": time.time(),
             "hours": args.hours, "hf_repo": repo, "transfer": args.transfer}
    if args.transfer == "ssh":
        state.update(dataset=os.path.abspath(dataset_zip), config=os.path.abspath(config_yaml))
        try:
            vast("attach", "ssh", str(instance_id), pubkey, raw=False)
            state["key_attached"] = True
        except SystemExit as e:
            print(f"couldn't attach the SSH key yet ({e}); watch will retry")
    save_state(project, state)
    print(f"instance {instance_id} starting. Follow it:\n  python vast_pod.py watch --project {project}")
    if args.transfer == "ssh":
        print(f"Start watch now: it sends the dataset. The pod gives up after {args.loading_timeout} min without it.")
    return 0


def instance_info(instance_id: int) -> dict | None:
    data = vast("show", "instances")
    rows = data if isinstance(data, list) else data.get("instances", [])
    return next((r for r in rows if r.get("id") == instance_id), None)


def cmd_status(args) -> int:
    st = load_state(logging_util.sanitize_project_name(args.project))
    if not st:
        print("no run recorded")
        return 0
    info = instance_info(st["instance_id"])
    hrs = (time.time() - st["started"]) / 3600
    print(f"instance {st['instance_id']}: {'gone (destroyed)' if not info else info.get('actual_status')}  "
          f"{hrs:.2f}h elapsed  ~${hrs * st['dph']:.2f} spent  (limit {st['hours']}h / ~${st['hours'] * st['dph']:.2f})")
    return 0


def cmd_logs(args) -> int:
    st = load_state(logging_util.sanitize_project_name(args.project))
    if not st:
        raise SystemExit("no run recorded")
    print(vast("logs", str(st["instance_id"]), raw=False))
    return 0


def destroy_instance(instance_id: int) -> str:
    """REST, not the CLI: `vastai destroy` prompts for confirmation and aborts when there's no tty."""
    import httpx

    r = httpx.delete(f"https://console.vast.ai/api/v0/instances/{instance_id}",
                     headers={"Authorization": f"Bearer {os.environ['VAST_API_KEY'].strip()}"}, timeout=30,
                     follow_redirects=True)
    return f"destroy {instance_id}: {r.status_code} {r.text[:120]}"


def cmd_destroy(args) -> int:
    project = logging_util.sanitize_project_name(args.project)
    st = load_state(project)
    iid = args.id or st.get("instance_id")
    if not iid:
        raise SystemExit("no instance id")
    print(destroy_instance(iid))
    st["destroyed"] = time.time()
    save_state(project, st)
    return 0


def cmd_watch(args) -> int:
    project = logging_util.sanitize_project_name(args.project)
    st = load_state(project)
    if not st:
        raise SystemExit("no run recorded")
    iid = st["instance_id"]
    out_dir = os.path.join(logging_util.get_project_output_dir(project), "pod", "results")
    os.makedirs(out_dir, exist_ok=True)
    over_ssh = st.get("transfer", "hf") == "ssh"
    if not over_ssh:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=os.environ["HF_TOKEN"])
        repo = st["hf_repo"]

    def results_in_repo() -> list[str]:
        if over_ssh:
            return []
        try:
            prefix = f"results/{iid}/"
            return [f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(prefix)]
        except Exception:
            return []

    def ssh_step(status: str) -> bool:
        """Push the dataset once the pod is up; pull the results once it says DONE. True when pulled."""
        if status != "running":
            return False
        if not st.get("key_attached"):
            try:
                vast("attach", "ssh", str(iid), ensure_ssh_key(), raw=False)
                st["key_attached"] = True
            except SystemExit:
                return False
        target = ssh_target(iid)
        if target is None:
            return False
        remote = f"{target[0]}:"
        if not st.get("uploaded"):
            print("  sending dataset + config over SSH")
            if (ssh(project, iid, target, "mkdir -p /workspace/incoming")
                    and scp(project, iid, target, [st["dataset"]], remote + "/workspace/incoming/dataset.zip")
                    and scp(project, iid, target, [st["config"]], remote + "/workspace/incoming/job.yaml")
                    and ssh(project, iid, target, "touch /workspace/incoming/READY")):
                st["uploaded"] = True
                save_state(project, st)
            return False
        if not ssh(project, iid, target, "test -f /workspace/out/DONE"):
            return False
        print("  downloading results over SSH")
        if scp(project, iid, target, [remote + "/workspace/out/*"], out_dir):
            ssh(project, iid, target, "touch /workspace/out/PULLED")
            return True
        return False

    while True:
        elapsed = (time.time() - st["started"]) / 3600
        info = instance_info(iid)
        files = results_in_repo()
        status = "gone" if info is None else info.get("actual_status")
        tail = ""
        if info is not None:
            try:
                full = str(vast("logs", str(iid), raw=False))
                if len(full) > 50:
                    with open(os.path.join(out_dir, f"pod_{iid}.log"), "w", encoding="utf-8") as fh:
                        fh.write(full)
            except SystemExit:
                pass
            try:
                log = str(vast("logs", str(iid), raw=False)).strip().splitlines()
                tail = log[-1][:110] if log else ""
            except SystemExit:
                tail = ""
        print(f"[{elapsed:.2f}h ~${elapsed * st['dph']:.2f}] {status}  results:{len(files)}  {tail}", flush=True)

        if over_ssh and info is not None and ssh_step(status):
            print("training finished -> destroying instance")
            print(destroy_instance(iid))
            break
        if f"results/{iid}/DONE" in files:
            for f in files:
                name = os.path.basename(f)
                if name != "DONE" and not os.path.exists(os.path.join(out_dir, name)):
                    print(f"  downloading {name}")
                    src = hf_hub_download(repo, f, repo_type="dataset", token=os.environ["HF_TOKEN"])
                    shutil.copy(src, os.path.join(out_dir, name))
            if info is not None:
                print("training finished -> destroying instance")
                print(destroy_instance(iid))
            break
        if info is None:
            print("instance gone before results appeared; check the logs above" + ("" if over_ssh else " / HF repo"))
            break
        if status == "loading" and elapsed > args.loading_timeout / 60:
            print(f"still 'loading' after {args.loading_timeout} min -- image pull is failing -> destroying")
            print(destroy_instance(iid))
            break
        if elapsed > st["hours"]:
            print("time limit reached -> destroying")
            print(destroy_instance(iid))
            break
        time.sleep(args.interval)
    st["finished"] = time.time()
    save_state(project, st)
    print(f"results in {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["search", "launch", "status", "logs", "destroy", "watch"])
    ap.add_argument("--project", default=env_config.get("AI_INFLUENCER_DEFAULT_PROJECT"))
    gpu_select.add_arguments(ap)
    ap.add_argument("--est-train-hours", type=float, default=gpu_select.PROFILES["train"].work_hours,
                    help="Training time on an RTX 5090; scaled by each GPU's speed to rank offers by total cost.")
    ap.add_argument("--offer", type=int, help="Specific offer id from `search`.")
    ap.add_argument("--hours", type=float, default=4.0, help="Hard limit; the pod destroys itself at this point.")
    ap.add_argument("--keep-checkpoints", type=int, default=6, help="How many of the last checkpoints to upload.")
    ap.add_argument("--dataset"); ap.add_argument("--config")
    ap.add_argument("--id", type=int, help="Instance id for destroy.")
    ap.add_argument("--transfer", choices=["ssh", "hf"], default="ssh",
                    help="How the dataset goes in and checkpoints come out: ssh (straight between this PC and the pod, "
                         "needs `watch` running) or hf (through a private Hugging Face repo, subject to HF's content policy).")
    ap.add_argument("--pull-grace", type=int, default=60,
                    help="ssh: minutes the finished pod waits for watch to download results before destroying itself.")
    ap.add_argument("--hf-repo", help="Private HF dataset repo used to move files (default: AI_INFLUENCER_HF_REPO, else <user>/<project>-lora-transfer).")
    ap.add_argument("--interval", type=int, default=120, help="watch poll seconds.")
    ap.add_argument("--loading-timeout", type=int, default=30, help="Destroy if still stuck in 'loading' after this many minutes (failed image pull).")
    ap.add_argument("--no-auto-destroy", action="store_true", help="Pod keeps running after training (you pay until destroyed).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    gpu_select.apply_defaults(args, gpu_select.PROFILES["train"])

    env_config.load_env()  # the commands below read VAST_API_KEY / HF_TOKEN from os.environ
    return {"search": cmd_search, "launch": cmd_launch, "status": cmd_status,
            "logs": cmd_logs, "destroy": cmd_destroy, "watch": cmd_watch}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

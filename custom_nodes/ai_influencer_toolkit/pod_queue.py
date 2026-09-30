"""Hold ComfyUI's Queue presses on this PC, then run all of them on a vast.ai pod in one go.

The browser side (web/pod_queue.js) adds a "Pod queue" panel to the canvas. While "Hold for the
pod" is on, pressing ComfyUI's normal Queue button (batch count included, seeds randomize as
usual) stores the prompt here instead of running it. "Start pod & run all" then:

  1. if the pod kit on the HF repo is older than this PC's node pack / workflows / setup script /
     character.json, rebuilds it and uploads it (a pod runs whatever kit is on HF, not these files);
     then rents a pod with tools/comfy_pod.py (or uses the one already running), adding
     --with-upscale / --with-minimax when a held prompt needs SeedVR2 / MiniMax H3;
  2. waits until ComfyUI there has finished setting up (the node pack is registered only after the
     bootstrap's restart, which comes after every model download);
  3. checks every held prompt against the pod's node and model lists, uploads the input images
     they reference (same file names, so the prompts are sent unchanged), and queues them;
  4. downloads every output into this PC's output folder, at the same path it had on the pod,
     retrying failed downloads;
  5. destroys the pod -- also when the run fails or is stopped, after fetching whatever finished --
     unless you untick that. If a file still can't be downloaded, the pod is kept (its own --hours
     timer still ends it) rather than destroyed with your image on it.

Every image a held prompt saves goes through the background blur and then the camera pass as its
last step: ensure_filters() adds whichever of the two a Save node's image is missing.

Prompts that finished are removed from the held list; ones that failed stay, so they can be fixed
and re-run. Everything here is local open weights on the pod: no partner API, no moderation.
"""
from __future__ import annotations

import json
import logging
import os
import http.client
import random
import re
import ssl
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter

from aiohttp import web

import folder_paths
from server import PromptServer

from . import env_config

PACK_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(PACK_DIR, "tools")
REPO_ROOT = os.path.abspath(os.path.join(PACK_DIR, "..", ".."))
DEFAULT_PROJECT = env_config.get("AI_INFLUENCER_DEFAULT_PROJECT")
MEDIA_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".mp4", ".webm", ".mov",
              ".wav", ".mp3", ".flac", ".ogg")
READY_NODE = "AIInfluencerSaveImage"
BIREFNET = "birefnet.safetensors"
CAMERA, BLUR = "AIInfluencerCameraImperfections", "AIInfluencerBlurBackground"
IMAGE_SAVERS = ("AIInfluencerSaveImage", "SaveImage")  # nodes whose images end up in output/ (and get downloaded)


# ---------------------------------------------------------------------------------------------
# the held list


_store_lock = threading.Lock()


def _store_path() -> str:
    return os.path.join(folder_paths.get_user_directory(), "ai_influencer", "pod_queue.json")


def load_items() -> list[dict]:
    try:
        with open(_store_path(), encoding="utf-8") as f:
            return json.load(f).get("items", [])
    except (OSError, ValueError):
        return []


def save_items(items: list[dict]) -> None:
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"items": items}, f)
    os.replace(path + ".tmp", path)


def summarize(prompt: dict) -> str:
    """A one-line reminder of what this held prompt is: its prompt text and seed."""
    text, seed = "", None
    for node in prompt.values():
        inputs = node.get("inputs", {})
        if not text and node.get("class_type") == "AIInfluencerPromptBuilder":
            text = ", ".join(str(inputs[k]) for k in ("subject_action", "clothing", "environment")
                             if isinstance(inputs.get(k), str) and inputs[k].strip())
        if not text and isinstance(inputs.get("text"), str):
            text = inputs["text"]
        for k in ("seed", "noise_seed"):
            if seed is None and isinstance(inputs.get(k), int):
                seed = inputs[k]
    text = " ".join(text.split())[:140] or "(no prompt text)"
    return f"{text}  ·  seed {seed}" if seed is not None else text


def project_of(items: list[dict]) -> str:
    names = Counter(
        node["inputs"]["project"] for it in items for node in it["prompt"].values()
        if isinstance(node.get("inputs", {}).get("project"), str) and node["inputs"]["project"].strip()
    )
    return names.most_common(1)[0][0] if names else DEFAULT_PROJECT


def all_strings(items: list[dict]):
    for it in items:
        for node in it["prompt"].values():
            for v in node.get("inputs", {}).values():
                if isinstance(v, str):
                    yield v


# ---------------------------------------------------------------------------------------------
# every saved image gets the background blur and then the camera pass


def _ancestors(prompt: dict, nid: str) -> set[str]:
    seen, todo = set(), [nid]
    while todo:
        cur = todo.pop()
        if cur in seen or cur not in prompt:
            continue
        seen.add(cur)
        todo += [v[0] for v in prompt[cur].get("inputs", {}).values() if isinstance(v, list) and len(v) == 2]
    return {prompt[n]["class_type"] for n in seen}


def ensure_filters(prompt: dict) -> list[str]:
    """Makes every image a Save node writes pass through Blur Background and then, as the very last
    step, Camera Imperfections -- adding whichever is missing, with Workflow 08's defaults. Mutates
    `prompt`; returns what it changed or found. Idempotent, so it is safe to run on every run."""
    notes: list[str] = []
    next_id = max((int(k) for k in prompt if str(k).isdigit()), default=0) + 1000

    def new(class_type: str, **inputs) -> str:
        nonlocal next_id
        next_id += 1
        prompt[str(next_id)] = {"class_type": class_type, "inputs": inputs}
        return str(next_id)

    savers = [n for n in prompt.values() if n.get("class_type") in IMAGE_SAVERS]
    if not savers:
        notes.append("no Save node: nothing would be saved or downloaded")
    for node in savers:
        src = node.get("inputs", {}).get("images")
        if not isinstance(src, list):
            continue
        last = prompt.get(src[0], {})
        camera_last = last.get("class_type") == CAMERA
        feed = last["inputs"].get("image") if camera_last else src
        if BLUR not in _ancestors(prompt, src[0]):
            bg = new("LoadBackgroundRemovalModel", bg_removal_name=BIREFNET)
            mask = new("RemoveBackground", bg_removal_model=[bg, 0], image=feed)
            feed = [new(BLUR, image=feed, subject_mask=[mask, 0], blur_amount=1.5, feather=0.4,
                        bokeh_highlights=0.3, subject_edge_shift=0.0), 0]
            notes.append("added the background blur")
        if camera_last:
            last["inputs"]["image"] = feed
        else:
            node["inputs"]["images"] = [new(CAMERA, image=feed, seed=random.randrange(2**31), iso_grain=0.025,
                                            grain_size=0.1, chroma_noise=0.35, chromatic_aberration=0.05,
                                            vignette=0.15, color_jitter=0.3), 0]
            notes.append("added the camera pass as the last step")
    for n in prompt.values():  # switched on but turned down to nothing: say so, don't overrule it
        ins = n.get("inputs", {})
        if n.get("class_type") == BLUR and ins.get("blur_amount") == 0:
            notes.append("background blur is set to 0 (sharp background)")
        if n.get("class_type") == CAMERA and ins.get("iso_grain") == 0:
            notes.append("camera pass grain is set to 0")
    return sorted(set(notes))


# ---------------------------------------------------------------------------------------------
# the pod kit on HF (what a new pod installs: node pack, workflows, character.json, setup script)


def kit_sources(project: str) -> list[tuple[str, str]]:
    """(file on this PC, path in the tarball), the same layout as the README's kit recipe."""
    pairs: list[tuple[str, str]] = []

    def tree(src: str, arc: str) -> None:
        for dirpath, dirnames, files in os.walk(src):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for f in files:
                if not f.endswith(".pyc"):
                    full = os.path.join(dirpath, f)
                    pairs.append((full, os.path.join(arc, os.path.relpath(full, src))))

    tree(PACK_DIR, "kit/ai_influencer_toolkit")
    tree(os.path.join(REPO_ROOT, "user", "default", "workflows", "ai_influencer"), "kit/workflows/ai_influencer")
    tree(os.path.join(REPO_ROOT, "input", "ai_influencer", project, "dataset"), "kit/dataset")
    pairs.append((os.path.join(REPO_ROOT, "output", "projects", project, "character.json"), "kit/project/character.json"))
    pairs.append((os.path.join(TOOLS_DIR, "pod_bootstrap.sh"), "kit/pod_bootstrap.sh"))
    return pairs


def kit_status(project: str) -> tuple[bool, str]:
    """(needs a new upload, why)."""
    from huggingface_hub import HfApi

    info = HfApi(token=env_config.require("HF_TOKEN")).get_paths_info(env_config.hf_repo(project), ["pod_kit.tar.gz"], repo_type="dataset", expand=True)
    if not info or not getattr(info[0], "last_commit", None):
        return True, "there is no pod kit on the HF repo yet"
    uploaded = info[0].last_commit.date.timestamp()
    newer = [arc for full, arc in kit_sources(project) if os.path.getmtime(full) > uploaded]
    if newer:
        return True, f"{len(newer)} file(s) changed since the kit was uploaded, e.g. {', '.join(n[4:] for n in newer[:3])}"
    return False, "the pod kit on HF is up to date"


def upload_kit(project: str) -> str:
    from huggingface_hub import HfApi

    out = os.path.join(REPO_ROOT, "output", "projects", project, "pod", "pod_kit.tar.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tarfile.open(out + ".tmp", "w:gz") as tar:
        for full, arc in kit_sources(project):
            tar.add(full, arcname=arc)
    os.replace(out + ".tmp", out)
    repo = env_config.hf_repo(project)
    HfApi(token=env_config.require("HF_TOKEN")).upload_file(
        path_or_fileobj=out, path_in_repo="pod_kit.tar.gz", repo_id=repo, repo_type="dataset",
        commit_message="pod kit (rebuilt by the Pod queue)")
    return f"uploaded a new pod kit ({os.path.getsize(out) / 1e6:.0f} MB) to {repo}"


# ---------------------------------------------------------------------------------------------
# the pod's ComfyUI, over HTTP


def wrong_scheme(e: BaseException) -> bool:
    """True for the errors you get talking HTTP to an HTTPS port or HTTPS to an HTTP one. urllib
    wraps some of them in URLError, so look at .reason too. Connection refused is *not* one of them."""
    kinds = (ssl.SSLError, http.client.RemoteDisconnected, http.client.BadStatusLine, ConnectionResetError)
    return isinstance(e, kinds) or (isinstance(e, urllib.error.URLError) and isinstance(e.reason, kinds))


class Remote:
    """The pod's ComfyUI behind the image's Caddy proxy: HTTPS with a self-signed certificate (so it
    isn't verified -- the traffic is still encrypted) and the WEB_PASSWORD comfy_pod.py set at launch
    as a Bearer token. Starts on https and flips to http if the proxy turns out to speak plain HTTP."""

    def __init__(self, url: str, token: str | None = None):
        self.hostport = url.rstrip("/").split("://", 1)[-1]
        self.scheme = "https"
        self.token = token
        self.ssl = ssl._create_unverified_context()

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.hostport}"

    def switch_scheme(self) -> None:
        self.scheme = "http" if self.scheme == "https" else "https"

    def _req(self, path: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 60) -> bytes:
        headers = dict(headers or {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url + path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout, context=self.ssl) as resp:
            return resp.read()

    def get_json(self, path: str, timeout: int = 30):
        return json.loads(self._req(path, timeout=timeout) or b"{}")

    def post_json(self, path: str, obj) -> dict:
        body = self._req(path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
        return json.loads(body) if body.strip() else {}

    def upload(self, local_path: str, subfolder: str, name: str, dir_type: str) -> None:
        with open(local_path, "rb") as f:
            data = f.read()
        boundary = uuid.uuid4().hex

        def field(key: str, value: str) -> bytes:
            return f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()

        body = (field("subfolder", subfolder) + field("type", dir_type) + field("overwrite", "true")
                + f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                  f"Content-Type: application/octet-stream\r\n\r\n".encode()
                + data + f"\r\n--{boundary}--\r\n".encode())
        self._req("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=600)

    def queue(self, item: dict, client_id: str) -> str:
        payload = {"prompt": item["prompt"], "client_id": client_id}
        if item.get("workflow"):
            payload["extra_data"] = {"extra_pnginfo": {"workflow": item["workflow"]}}  # PNGs keep the workflow
        try:
            res = self.post_json("/prompt", payload)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"rejected ({e.code}): {e.read().decode(errors='replace')[:800]}") from None
        if res.get("node_errors"):
            raise RuntimeError(f"node errors: {json.dumps(res['node_errors'])[:800]}")
        return res["prompt_id"]

    def download(self, entry: dict, dest: str) -> None:
        q = urllib.parse.urlencode({"filename": entry["filename"], "subfolder": entry.get("subfolder", ""), "type": "output"})
        data = self._req(f"/view?{q}", timeout=600)
        if not data:
            raise RuntimeError(f"{entry['filename']} came back empty")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)

    def cancel_all(self) -> None:
        for path, obj in (("/queue", {"clear": True}), ("/interrupt", {})):
            try:
                self.post_json(path, obj)
            except Exception:
                pass


def input_specs(object_info: dict, class_type: str) -> dict:
    spec = object_info.get(class_type, {}).get("input", {})
    return {**spec.get("optional", {}), **spec.get("required", {})}


def combo_options(spec) -> list | None:
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


def character_lora(project: str) -> str | None:
    """The LoRA file "(from character.json)" resolves to (the pod gets the same character.json in its kit)."""
    try:
        with open(os.path.join(REPO_ROOT, "output", "projects", project, "character.json"), encoding="utf-8") as f:
            return json.load(f).get("local_lora") or None
    except (OSError, ValueError):
        return None


def local_media(value: str) -> tuple[str, str, str, str] | None:
    """(local path, subfolder, file name, input|output) for an input value that names a file on this PC."""
    if not value.lower().split(" [")[0].endswith(MEDIA_EXTS):
        return None
    try:
        path = folder_paths.get_annotated_filepath(value)
    except Exception:
        return None
    if not path or not os.path.isfile(path):
        return None
    dir_type = "output" if value.endswith("[output]") else "input"
    clean = value.rsplit(" [", 1)[0] if value.endswith("]") else value
    sub, name = os.path.split(clean.replace("\\", "/"))
    return path, sub, name, dir_type


# ---------------------------------------------------------------------------------------------
# the run


class Stopped(Exception):
    pass


class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.pod_id: int | None = None
        self._setup_checked = 0.0
        self.project = DEFAULT_PROJECT
        self.remote: Remote | None = None
        self.pending: dict[str, dict] = {}  # pod prompt_id -> held item, until its files are downloaded
        self.undownloaded: list[tuple[dict, dict, str]] = []  # (item, output entry, error) that failed every retry
        self.state = self._fresh()

    @staticmethod
    def _fresh() -> dict:
        return {"running": False, "phase": "", "log": [], "total": 0, "done": 0, "failed": 0, "url": ""}

    def log(self, msg: str) -> None:
        logging.info("[pod queue] %s", msg)
        with self.lock:
            self.state["log"] = (self.state["log"] + [f"{time.strftime('%H:%M:%S')}  {msg}"])[-300:]

    def set(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def start(self, settings: dict) -> str | None:
        if self.thread and self.thread.is_alive():
            return "A run is already going."
        if not load_items():
            return "Nothing is held yet. Turn on \"Hold for the pod\" and press Queue."
        self.stop_flag.clear()
        self.state = self._fresh()
        self.set(running=True, phase="starting")
        self.thread = threading.Thread(target=self._run, args=(settings,), daemon=True, name="pod-queue")
        self.thread.start()
        return None

    def stop(self) -> None:
        self.stop_flag.set()
        self.log("Stop requested.")

    def _sleep(self, seconds: float) -> None:
        if self.stop_flag.wait(seconds):
            raise Stopped()

    # -- the pod ------------------------------------------------------------------------------

    @staticmethod
    def _tools():
        if TOOLS_DIR not in sys.path:
            sys.path.insert(0, TOOLS_DIR)
        import comfy_pod
        import vast_pod

        return comfy_pod, vast_pod

    def _pod_url(self, settings: dict, flags: list[str]) -> str:
        comfy_pod, vast_pod = self._tools()
        iid = vast_pod.load_state(self.project).get(comfy_pod.STATE_KEY)
        info = vast_pod.instance_info(iid) if iid else None
        if info is None:
            self.set(phase="checking the pod kit on HF")
            stale, why = kit_status(self.project)
            self.log(why)
            if stale:
                self.set(phase="rebuilding and uploading the pod kit")
                self.log(upload_kit(self.project))
            hours = float(settings.get("hours") or 2.0)
            cmd = [sys.executable, os.path.join(TOOLS_DIR, "comfy_pod.py"), "launch", "--yes",
                   "--project", self.project, "--hours", str(hours), "--hf-repo", env_config.hf_repo(self.project), *flags]
            self.set(phase="renting a pod")
            self.log(f"Renting a pod for {self.project} (hard limit {hours} h){' with ' + ' '.join(flags) if flags else ''}")
            proc = subprocess_run(cmd)
            for line in (proc.stdout + proc.stderr).splitlines():
                if line.strip():
                    self.log("  " + line)
            iid = vast_pod.load_state(self.project).get(comfy_pod.STATE_KEY)
            if proc.returncode != 0 or not iid:
                raise RuntimeError("comfy_pod.py launch failed (see above).")
        else:
            self.log(f"Using the pod that is already running (instance {iid}). It has whatever kit it was "
                     "started with; anything it lacks shows up in the check below.")
        self.pod_id = iid

        self.set(phase="waiting for the pod's address")
        deadline = time.time() + 30 * 60
        while True:
            info = vast_pod.instance_info(iid)
            if info is None:
                raise RuntimeError(f"Instance {iid} is gone.")
            url = comfy_pod.comfy_url(info)
            if url:
                return url
            if time.time() > deadline:
                raise RuntimeError("The pod never got a public port 8188 in 30 minutes.")
            self.log(f"pod: {info.get('actual_status')}, no address yet")
            self._sleep(20)

    def destroy(self, iid: int | None = None) -> str:
        comfy_pod, vast_pod = self._tools()
        iid = iid or vast_pod.load_state(self.project).get(comfy_pod.STATE_KEY)
        if not iid:
            return "No pod recorded."
        try:
            msg = f"Destroyed pod {iid}: {vast_pod.destroy_instance(iid)}"
        except BaseException as e:  # the vast wrapper raises SystemExit
            msg = (f"!! Could not destroy pod {iid} ({e}). Do it now: python "
                   f"custom_nodes/ai_influencer_toolkit/tools/comfy_pod.py destroy --project {self.project}")
        self.log(msg)
        return msg

    def _wait_ready(self, remote: Remote, items: list[dict]) -> dict:
        """Ready = the node pack is loaded AND every model file the held jobs use is listed. The node
        pack is copied before the ~16 GB of models download, and ComfyUI can be up in between, so the
        node alone isn't enough. If the pod's setup log says setup is over and files are still missing,
        they are never coming: stop waiting, run what can run."""
        self.set(phase="waiting for ComfyUI on the pod to finish setting up")
        deadline, last, rejected_since = time.time() + 25 * 60, "", None  # setup is normally 5-10 min
        self._setup_checked = 0.0
        while True:
            try:
                info = remote.get_json("/object_info", timeout=60)
                if READY_NODE in info:
                    missing = self._missing(items, info)
                    if not missing:
                        self.log(f"ComfyUI on the pod is ready ({remote.scheme}), with every model the jobs use.")
                        return info
                    setup = self._setup_state()
                    if setup is not None:
                        self.log(f"The pod's setup has finished{' with errors' if setup['errors'] else ''}, but "
                                 f"{len(missing)} thing(s) are still missing: {', '.join(missing[:6])}")
                        for line in setup["errors"]:
                            self.log("   pod: " + line)
                        return info  # _run skips what can't run; if nothing can, it stops and the pod is destroyed
                    status = (f"ComfyUI is up; waiting for {len(missing)} model file(s) to finish downloading "
                              f"({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''})")
                else:
                    status = "ComfyUI is up; still downloading models / installing the node pack"
                rejected_since = None
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    rejected_since = rejected_since or time.time()
                    status = f"the pod's proxy rejects the password (HTTP {e.code})"
                    if time.time() - rejected_since > 6 * 60:  # it won't fix itself; don't bill 45 min for it
                        raise RuntimeError(f"{status} for 6 minutes. Open {remote.url} in the browser (user vastai, "
                                           "password from `comfy_pod.py url`) to check.") from None
                else:
                    status = f"the proxy answers but ComfyUI doesn't yet (HTTP {e.code}; still starting)"
            except Exception as e:
                if wrong_scheme(e):  # plain HTTP to an HTTPS port hangs up, and vice versa
                    status = f"not reachable yet over {remote.scheme} ({type(e).__name__}); trying the other scheme"
                    remote.switch_scheme()
                else:  # connection refused, timeout...: the pod is still starting
                    status = f"not reachable yet ({type(e).__name__}: {str(e)[:80]})"
            if status != last:
                self.log(status)
                last = status
            if time.time() > deadline:
                raise RuntimeError(f"ComfyUI on the pod never became ready: {status}")
            self._sleep(20)

    def _missing(self, items: list[dict], info: dict) -> list[str]:
        """Node types and model/combo values the held jobs use that the pod doesn't list yet."""
        out = set()
        for it in items:
            for node in it["prompt"].values():
                ct = node.get("class_type")
                if ct not in info:
                    out.add(ct)
                    continue
                specs = input_specs(info, ct)
                for name, value in node.get("inputs", {}).items():
                    opts = combo_options(specs.get(name))
                    if value == "(from character.json)" and opts is not None:
                        value = character_lora(node.get("inputs", {}).get("project") or self.project) or value
                    if opts is not None and isinstance(value, str) and value not in opts and not local_media(value):
                        out.add(value)
        return sorted(out)

    def _setup_state(self) -> dict | None:
        """None while the pod's onstart script is still running; once it printed SETUP_DONE, the
        WARN / FATAL / FAILED lines it logged. Reads `vastai logs` at most once a minute."""
        if self.pod_id is None or time.time() - self._setup_checked < 60:
            return None
        self._setup_checked = time.time()
        _, vast_pod = self._tools()
        try:
            text = str(vast_pod.vast("logs", str(self.pod_id), "--tail", "200", raw=False))
        except BaseException:  # the vast wrapper raises SystemExit
            return None
        if "SETUP_DONE" not in text:
            return None
        errors = [l.strip() for l in text.replace("\r", "\n").splitlines()
                  if re.search(r"WARN|FATAL|FAILED", l) and "hf_" not in l][-8:]
        return {"errors": errors}

    def _problems(self, item: dict, info: dict) -> list[str]:
        out = []
        for nid, node in item["prompt"].items():
            ct = node.get("class_type")
            if ct not in info:
                out.append(f"node {ct} doesn't exist on the pod")
                continue
            specs = input_specs(info, ct)
            for name, value in node.get("inputs", {}).items():
                opts = combo_options(specs.get(name))
                if opts is None or not isinstance(value, str) or value in opts or local_media(value):
                    continue
                out.append(f"{ct}.{name} = {value!r} is not on the pod")
        return out

    # -- the whole thing ----------------------------------------------------------------------

    def _run(self, settings: dict) -> None:
        remote = None
        self.pod_id, self.remote, self.pending, self.undownloaded = None, None, {}, []
        try:
            items = load_items()
            self.project = project_of(items)
            strings = [s.lower() for s in all_strings(items)]
            flags = []
            if any("seedvr2" in s for s in strings):
                flags.append("--with-upscale")
            if any("minimax_h3" in s for s in strings):
                flags.append("--with-minimax")

            url = self._pod_url(settings, flags)
            self.set(url=url)
            self.log(f"Pod ComfyUI: {url}")
            comfy_pod, vast_pod = self._tools()
            token = vast_pod.load_state(self.project).get("comfy_token")
            if not token:
                self.log("(this pod was started without a known password; if its proxy asks for one, the run "
                         "stops within a few minutes -- destroy it and let the Pod queue start a new one)")
            remote = self.remote = Remote(url, token)
            info = self._wait_ready(remote, items)
            self.set(url=remote.url)

            self.set(phase="checking the held prompts")
            runnable = []
            for it in items:
                notes = ensure_filters(it["prompt"])
                if any(n.startswith("added") for n in notes):
                    self.log(f"#{it['n']}: {'; '.join(notes)}")
                problems = [f"{v} is not on the pod" for v in self._missing([it], info)] + [n for n in notes if n.startswith("no Save node")]
                if problems:
                    self.log(f"!! skipped #{it['n']} ({it['name']}): {'; '.join(problems[:4])}")
                else:
                    runnable.append(it)
            if not runnable:
                raise RuntimeError("None of the held prompts can run on this pod (see above).")

            self.set(phase="uploading input images")
            uploaded = set()
            for it in runnable:
                for node in it["prompt"].values():
                    for v in node.get("inputs", {}).values():
                        media = isinstance(v, str) and local_media(v)
                        if media and media[0] not in uploaded:
                            if self.stop_flag.is_set():
                                raise Stopped()
                            remote.upload(*media)
                            uploaded.add(media[0])
            if uploaded:
                self.log(f"Uploaded {len(uploaded)} input file(s).")

            self.set(phase="queueing")
            client_id, pending = uuid.uuid4().hex, self.pending
            for it in runnable:
                try:
                    pending[remote.queue(it, client_id)] = it
                except RuntimeError as e:
                    self._failed(it, str(e))
            self.set(total=len(pending) + self.state["failed"], phase="generating")
            self.log(f"Queued {len(pending)} prompt(s) on the pod.")

            while pending:
                self._sleep(5)
                self._collect()
            self.log(f"Finished: {self.state['done']} done, {self.state['failed']} failed or skipped.")
            self.set(phase="finished")
        except Stopped:
            self.set(phase="stopped")
            if remote is not None:
                remote.cancel_all()
                self.log("Cleared the pod's queue.")
        except BaseException as e:  # SystemExit from the vast wrapper as well
            self.set(phase="failed")
            self.log(f"!! {e}")
            if not isinstance(e, (RuntimeError, SystemExit)):
                self.log(traceback.format_exc())
        finally:
            self._last_sweep()
            if self.pod_id is not None and self.undownloaded:
                self.log(f"!! {len(self.undownloaded)} file(s) could not be downloaded, so the pod is NOT destroyed. "
                         f"Open {self.state.get('url')} in the browser (user vastai, password from `comfy_pod.py url`) "
                         f"to save them (Queue -> history), then press "
                         f"\"Destroy pod\". Its hard time limit still ends it.")
                for it, e, err in self.undownloaded:
                    self.log(f"   #{it['n']}: {e.get('subfolder', '')}/{e['filename']} ({err})")
            elif self.pod_id is not None and settings.get("auto_destroy", True):
                self.destroy(self.pod_id)
            elif self.pod_id is not None:
                self.log("The pod is still running and billing. Press \"Destroy pod\" when you're done.")
            self.set(running=False)

    def _collect(self) -> None:
        """One pass over the queued prompts: download the files of every one that has finished."""
        remote, out_dir = self.remote, folder_paths.get_output_directory()
        for pid in list(self.pending):
            try:
                h = remote.get_json(f"/history/{pid}").get(pid)
            except Exception as e:
                self.log(f"(history check failed, retrying: {type(e).__name__})")
                return
            if not h:  # a prompt only appears in /history once it has finished (or failed)
                continue
            it = self.pending.pop(pid)
            status = h.get("status") or {}
            if status.get("status_str") == "error":
                msg = next((d.get("exception_message", "") for ev, d in status.get("messages", [])
                            if ev == "execution_error"), "unknown error")
                self._failed(it, msg.strip()[:500])
                continue
            saved, missing = 0, 0
            for node_out in (h.get("outputs") or {}).values():
                for entries in node_out.values():
                    for e in entries if isinstance(entries, list) else []:
                        if isinstance(e, dict) and e.get("filename") and e.get("type") == "output":
                            err = self._download(e, out_dir)
                            if err:
                                missing += 1
                                self.undownloaded.append((it, e, err))
                            else:
                                saved += 1
            with self.lock:
                self.state["done"] += 1
                done, total = self.state["done"], self.state["total"]
            if missing:
                self.log(f"!! [{done}/{total}] #{it['n']}: {saved} file(s) saved, {missing} could not be downloaded "
                         "(kept in the list)")
            elif saved == 0:
                self.log(f"!! [{done}/{total}] #{it['n']} finished but saved no image (kept in the list)")
            else:
                self._finished(it)
                self.log(f"[{done}/{total}] #{it['n']} done, {saved} file(s) saved to output/")

    def _download(self, entry: dict, out_dir: str) -> str | None:
        """Downloads one output file, retrying; returns None on success or the last error."""
        dest = unique_path(os.path.join(out_dir, entry.get("subfolder", ""), entry["filename"]))
        err = None
        for wait in (0, 5, 15, 45):
            time.sleep(wait)  # deliberately not interrupted by Stop: finished images always come home
            try:
                self.remote.download(entry, dest)
                return None
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        return err

    def _last_sweep(self) -> None:
        """Before the pod can be destroyed: fetch whatever finished but wasn't collected yet (after a
        Stop or an error), and give failed downloads one more round."""
        if self.remote is None:
            return
        if self.pending:
            self.set(phase="fetching finished images before shutting down")
            self._collect()
        if self.undownloaded:
            time.sleep(30)
            retry, self.undownloaded = self.undownloaded, []
            for it, e, _ in retry:
                err = self._download(e, folder_paths.get_output_directory())
                if err:
                    self.undownloaded.append((it, e, err))
                else:
                    self.log(f"#{it['n']}: {e['filename']} downloaded on retry")

    def _failed(self, it: dict, msg: str) -> None:
        with self.lock:
            self.state["failed"] += 1
        self.log(f"!! #{it['n']} failed: {msg}")

    def _finished(self, it: dict) -> None:
        with _store_lock:
            save_items([x for x in load_items() if x["id"] != it["id"]])


def subprocess_run(cmd: list[str]):
    import subprocess

    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900)


def unique_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(path):
        path = f"{base}_pod{n}{ext}"
        n += 1
    return path


RUNNER = Runner()


# ---------------------------------------------------------------------------------------------
# routes for web/pod_queue.js


routes = PromptServer.instance.routes


def _listing() -> dict:
    items = load_items()
    return {"items": [{**{k: it[k] for k in ("id", "n", "name", "summary", "added")}, "notes": it.get("notes", [])}
                      for it in items],
            "project": project_of(items), "run": RUNNER.snapshot()}


@routes.get("/ai_influencer/pod_queue")
async def pod_queue_list(request):
    return web.json_response(_listing())


@routes.post("/ai_influencer/pod_queue/add")
async def pod_queue_add(request):
    body = await request.json()
    prompt = body.get("prompt")
    if not isinstance(prompt, dict) or not prompt:
        return web.json_response({"error": "empty prompt"}, status=400)
    with _store_lock:
        items = load_items()
        n = max((it.get("n", 0) for it in items), default=0) + 1
        notes = ensure_filters(prompt)
        items.append({"id": uuid.uuid4().hex, "n": n, "name": str(body.get("name") or "workflow")[:80],
                      "summary": summarize(prompt), "added": time.time(), "prompt": prompt,
                      "workflow": body.get("workflow"), "notes": notes})
        save_items(items)
    return web.json_response({"ok": True, "count": len(items), "n": n, "notes": notes})


@routes.get("/ai_influencer/pod_queue/item")
async def pod_queue_item(request):
    it = next((x for x in load_items() if x["id"] == request.query.get("id")), None)
    if it is None:
        return web.json_response({"error": "That job is no longer in the list."}, status=404)
    return web.json_response(it)


@routes.post("/ai_influencer/pod_queue/update")
async def pod_queue_update(request):
    """Replaces one held job with the edited canvas; it keeps its number and its place in the list."""
    if RUNNER.state.get("running"):
        return web.json_response({"error": "A run is going. Edit after it has finished."}, status=409)
    body = await request.json()
    prompt = body.get("prompt")
    if not isinstance(prompt, dict) or not prompt:
        return web.json_response({"error": "empty prompt"}, status=400)
    with _store_lock:
        items = load_items()
        it = next((x for x in items if x["id"] == body.get("id")), None)
        if it is None:
            return web.json_response({"error": "That job is no longer in the list (did it run?)."}, status=404)
        notes = ensure_filters(prompt)
        it.update(prompt=prompt, workflow=body.get("workflow") or it.get("workflow"),
                  summary=summarize(prompt), notes=notes, edited=time.time())
        save_items(items)
    return web.json_response({**_listing(), "notes": notes})


SEED_INPUTS = ("seed", "noise_seed")


@routes.post("/ai_influencer/pod_queue/reseed")
async def pod_queue_reseed(request):
    """Copies of one held job, identical except that every seed gets a new random value."""
    body = await request.json()
    count = max(1, min(64, int(body.get("count") or 1)))
    with _store_lock:
        items = load_items()
        src = next((it for it in items if it["id"] == body.get("id")), None)
        if src is None:
            return web.json_response({"error": "That job is no longer in the list."}, status=404)
        n = max((it.get("n", 0) for it in items), default=0)
        used = {v for node in src["prompt"].values() for k, v in node.get("inputs", {}).items() if k in SEED_INPUTS}
        copies = []
        for _ in range(count):
            prompt = json.loads(json.dumps(src["prompt"]))
            changed = False
            for node in prompt.values():
                for k in SEED_INPUTS:
                    if isinstance(node.get("inputs", {}).get(k), int):  # linked seeds ([node, slot]) follow their source
                        seed = random.randrange(2**31)  # < 2^31 fits every seed input, Save & Log's included
                        while seed in used:
                            seed = random.randrange(2**31)
                        used.add(seed)
                        node["inputs"][k] = seed
                        changed = True
            if not changed:
                return web.json_response({"error": "This job has no seed to change."}, status=400)
            n += 1
            copies.append({**src, "id": uuid.uuid4().hex, "n": n, "summary": summarize(prompt),
                           "added": time.time(), "prompt": prompt})
        i = items.index(src) + 1
        items[i:i] = copies  # right after the original, so they run together
        save_items(items)
    return web.json_response({**_listing(), "added": count})


@routes.post("/ai_influencer/pod_queue/remove")
async def pod_queue_remove(request):
    body = await request.json()
    ids = set(body.get("ids") or [])
    with _store_lock:
        save_items([] if body.get("all") else [it for it in load_items() if it["id"] not in ids])
    return web.json_response(_listing())


@routes.post("/ai_influencer/pod_queue/run")
async def pod_queue_run(request):
    err = RUNNER.start(await request.json())
    return web.json_response({"error": err} if err else {"ok": True}, status=409 if err else 200)


@routes.post("/ai_influencer/pod_queue/stop")
async def pod_queue_stop(request):
    RUNNER.stop()
    return web.json_response({"ok": True})


@routes.post("/ai_influencer/pod_queue/destroy")
async def pod_queue_destroy(request):
    RUNNER.project = project_of(load_items())
    msg = await PromptServer.instance.loop.run_in_executor(None, RUNNER.destroy)
    return web.json_response({"message": msg})

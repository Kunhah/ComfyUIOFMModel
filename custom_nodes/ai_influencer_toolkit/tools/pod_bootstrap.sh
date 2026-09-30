#!/usr/bin/env bash
# Run this INSIDE a ComfyUI GPU pod (vast.ai / RunPod), from the unpacked pod kit folder:
#     bash pod_bootstrap.sh
# It reproduces this local ComfyUI setup there: the AI Influencer node pack, workflows 01-11,
# the character's project files and dataset, and the Krea 2 + SeedVR2 models (~20 GB).
# Re-running is safe: existing model files are not downloaded again.
set -euo pipefail

COMFY="${COMFY:-/workspace/ComfyUI}"
PROJECT="${PROJECT:-MyCharacter}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -d "$COMFY" ] || { echo "No ComfyUI at $COMFY. Set COMFY=/path/to/ComfyUI and re-run."; exit 1; }

echo "== node pack =="
mkdir -p "$COMFY/custom_nodes"
rm -rf "$COMFY/custom_nodes/ai_influencer_toolkit"
cp -r "$KIT/ai_influencer_toolkit" "$COMFY/custom_nodes/"
# the vastai/comfy image has no bare `python`; its ComfyUI runs from /venv/main
PY=$([ -x /venv/main/bin/python ] && echo /venv/main/bin/python || command -v python3 || echo python)
"$PY" -m pip install -q opencv-python-headless python-dotenv || echo "(pip install failed; face scoring in Workflow 07 needs opencv-python-headless)"

echo "== workflows =="
mkdir -p "$COMFY/user/default/workflows"
cp -r "$KIT/workflows/ai_influencer" "$COMFY/user/default/workflows/"

echo "== project files =="
mkdir -p "$COMFY/output/projects/$PROJECT" "$COMFY/input/ai_influencer/$PROJECT" "$COMFY/models/loras/$PROJECT"
cp -r "$KIT/project/." "$COMFY/output/projects/$PROJECT/"
cp -r "$KIT/dataset" "$COMFY/input/ai_influencer/$PROJECT/"
[ -d "$KIT/poses" ] && cp -r "$KIT/poses" "$COMFY/input/ai_influencer/$PROJECT/"

echo "== models (~20 GB, skipped if already present) =="
# Hugging Face serves big files from its Xet storage, and one long curl stream of a 13 GB file keeps
# dying mid-transfer ("curl: (18) Transferred a partial file") while small files come through. Its own
# downloader fetches Xet files in chunks with retries, so huggingface.co files go through that; curl is
# the fallback. It is installed with --target into its own folder so ComfyUI's packages aren't touched.
HFDL=/opt/hfdl
"$PY" -m pip install -q --target "$HFDL" "huggingface_hub[hf_xet]" >/dev/null 2>&1 \
  && echo "  hf downloader ready" || echo "  WARN: could not install huggingface_hub (falling back to curl)"
hf_get() {  # hf_get <url> <dest-file>: 0 on success
  local rest repo path
  rest="${1#https://huggingface.co/}"
  [ "$rest" = "$1" ] && return 1                    # not a huggingface.co URL
  repo="${rest%%/resolve/*}"; path="${rest#*/resolve/main/}"
  [ -d "$HFDL/huggingface_hub" ] || return 1
  PYTHONPATH="$HFDL" HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" - "$repo" "$path" "$2" <<'PYEOF'
import os, shutil, sys
from huggingface_hub import hf_hub_download
repo, path, dest = sys.argv[1:4]
tmp = "/tmp/hfdl_" + os.path.basename(dest)   # outside models/, so ComfyUI never lists it half-done
got = hf_hub_download(repo_id=repo, filename=path, local_dir=tmp)
shutil.move(got, dest + ".part")
os.replace(dest + ".part", dest)
shutil.rmtree(tmp, ignore_errors=True)
PYEOF
}

DL_FAILED=""
dl() {  # dl <models-subdir> <url>
  # Downloads break mid-transfer on some hosts, so: retry and resume (-C -), into <file>.part that is
  # renamed only when complete -- ComfyUI never lists a half-written .safetensors. A file that still
  # fails is recorded and the rest carry on; the script exits non-zero at the end.
  local dir="$COMFY/models/$1" file attempt
  file="$(basename "$2")"
  mkdir -p "$dir"
  if [ -s "$dir/$file" ]; then
    echo "  have $file"
    return 0
  fi
  echo "  downloading $file"
  for attempt in 1 2 3; do
    if hf_get "$2" "$dir/$file"; then
      echo "  done $file ($(du -h "$dir/$file" | cut -f1), hf)"
      return 0
    fi
    echo "  hf download of $file failed (attempt $attempt); trying curl"
    if curl -fLsS --retry 8 --retry-delay 5 --retry-all-errors -C - -o "$dir/$file.part" "$2"; then
      mv "$dir/$file.part" "$dir/$file"
      echo "  done $file ($(du -h "$dir/$file" | cut -f1))"
      return 0
    fi
    echo "  retrying $file (attempt $attempt failed)"
    sleep 10
  done
  echo "  FAILED $file"
  DL_FAILED="$DL_FAILED $file"
}
dl diffusion_models https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors
dl text_encoders   https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
dl vae             https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors
dl clip_vision     https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors
# Background blur (Workflows 08/09/09b) needs the BiRefNet subject mask:
dl background_removal https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors
# Workflow 09b (multi-reference / pose transfer) needs these two as well:
dl loras           https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_style_reference.safetensors
dl checkpoints     https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/sdpose_wholebody_fp16.safetensors
if [ "${WITH_UPSCALE:-1}" = "1" ]; then
  dl diffusion_models https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_3b_int8_convrot.safetensors
  dl vae               https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/vae/seedvr2_ema_vae_fp16.safetensors
fi
# Workflow 11 (MiniMax H3 image-to-video with native audio). ~45 GB, so it is opt-in:
#     WITH_MINIMAX=1 bash pod_bootstrap.sh
if [ "${WITH_MINIMAX:-0}" = "1" ]; then
  echo "== MiniMax H3 (~45 GB) =="
  # H3 landed in ComfyUI after some v0.35.0 images were built; make sure the nodes exist.
  if [ ! -f "$COMFY/comfy_extras/nodes_minimax_h3.py" ]; then
    echo "  ComfyUI in this image has no MiniMax H3 nodes; updating it"
    git -C "$COMFY" pull --ff-only || echo "  WARN: git pull failed -- Workflow 11 will not load"
  fi
  dl diffusion_models https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
  dl text_encoders    https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
  dl vae              https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors
  dl vae              https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors
  dl loras            https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
fi

if [ -n "$DL_FAILED" ]; then
  echo "DOWNLOADS_FAILED:$DL_FAILED"
  exit 1
fi

cat <<EOF

Done. Next:
  1. Put your trained checkpoints in $COMFY/models/loras/$PROJECT/
  2. Restart ComfyUI, open Workflow 07 (checkpoint tester) or 08 (generate).
  3. Results land in $COMFY/output/projects/$PROJECT/ -- download them before destroying the pod.
EOF

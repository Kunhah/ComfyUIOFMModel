# AI Influencer Workstation (ComfyUI)

This turns this ComfyUI checkout into a workstation for one consistent AI influencer, with two
pipelines that share the same project folders, character.json and log:

- **Local Krea 2 + character LoRA (recommended for identity).** Krea 2 is open weights, so you
  can train a real character LoRA on it (Ostris AI Toolkit on a rented RunPod GPU), **test every
  saved checkpoint side by side** to find the best step count, and generate with Krea 2 Turbo
  plus a skin-polish pass and a SeedVR2 upscale. Workflows 07-09, see
  [Local Krea 2 pipeline](#local-krea-2-pipeline-character-lora). Needs an RTX 5090 / RTX 6000
  Pro class GPU, so in practice ComfyUI on RunPod, not the local GTX 1060.
- **Cloud API pipeline.** Generation through ComfyUI's official partner API nodes for Krea,
  OpenAI GPT Image, and ByteDance Seedance, with no local model at all. Workflows 01 and 04-06;
  Workflow 01 is where the character is created (GPT Image 2.5), which gives the local pipeline
  its training pictures.

ComfyUI is the interface, prompt/identity manager, and local cost log for both.

## What's actually installed vs. what you asked for

Before building anything, the current ComfyUI (v0.35.0) was inspected end to end. The good news:
**Krea 2, GPT Image 2.5, Seedance 2.5, and OpenRouter all already exist as official, first-party
ComfyUI nodes** (`comfy_api_nodes/`) — no third-party custom nodes were installed, so there is
nothing here to audit for maintenance/security risk, and `git pull`-ing ComfyUI updates these
nodes for you. The one exception is optional: choosing the Low memory (GGUF) Krea 2 version for
8 GB cards installs city96's ComfyUI-GGUF, pinned to one commit (see [Models](#models)). Two things from the brief work differently than assumed, and are documented in
full below rather than silently reinterpreted:

1. **Auth is one Comfy account key, not four provider keys.** See [Credentials](#credentials).
2. **No character LoRA in the Krea/GPT-Image/Seedance cloud APIs.** (The local, open-weights Krea 2
   pipeline does support one; see [Local Krea 2 pipeline](#local-krea-2-pipeline-character-lora).)
   None of their cloud APIs expose
   trainable-LoRA upload (that's an open-weights/local-GPU concept, which conflicts with "no
   local models"). The default identity strategy is reference-image chaining, which all three
   APIs do support: Krea's `style_reference` (≤10 images), GPT Image 2.5's multi-image edit mode
   (≤16 images), and Seedance's `reference_images`. See
   [Character identity strategy](#character-identity-strategy). **If you want a real trained
   LoRA anyway**, Workflow 06 adds one via fal.ai — a separate provider/account, still no local
   GPU. See [Optional: real LoRA training (fal.ai)](#optional-real-lora-training-falai).

Everything else — aspect ratio presets, candidate batching, the GPT refinement instruction,
Seedance motion prompting, OpenRouter prompt-helper — matches the brief using the real node
schemas, verified by generating and structurally validating each workflow against a running
ComfyUI instance (see [Testing performed](#testing-performed)).

## Installation & dependencies

Almost nothing new to install. The toolkit is one custom node folder,
`custom_nodes/ai_influencer_toolkit/`, using only the stdlib plus packages ComfyUI already ships
(`torch`, `Pillow`).
It's picked up automatically the next time ComfyUI starts; nothing in ComfyUI core was modified.
The one real dependency, `fal-client`, is optional and only needed for Workflow 06 (real LoRA
training) — see [Optional: real LoRA training (fal.ai)](#optional-real-lora-training-falai).

```bash
cd ComfyUI
source venv/bin/activate        # or however you normally activate your ComfyUI environment
python main.py
```

Open http://127.0.0.1:8188, then **Workflow → Open** and pick one of the thirteen files under
`user/default/workflows/ai_influencer/` (or use the workflow browser's "Workflows" tab).

**On Windows, without a terminal:** double-click `START_COMFYUI.bat`. The first run finds or
installs Python (via winget), creates `venv\`, installs the PyTorch build that fits the PC
(`cu130`, `cu126` for GTX 10-series and older, or CPU with `--cpu` when there is no NVIDIA card)
plus `requirements.txt`, opens the key window once, and then starts ComfyUI with the browser.
Later runs start right away, and they reinstall only when `requirements.txt` changes.
`SETTINGS_API_KEYS.bat` opens that key window (`tools/configure_gui.py`, a window version of
`configure.py`). `START_HERE_WINDOWS.txt` has the same steps for people who don't code. The logic
is in `tools/launcher.py`, and `START_COMFYUI.bat --reinstall` redoes the installs.

**On Linux:** `./start_comfyui.sh` does the same thing (Python itself comes from your package
manager; the script says which package to install when it's missing), and `./settings_api_keys.sh`
opens the key window, or asks in the terminal when Tk isn't installed. The launcher installs
PyTorch only when the venv doesn't have it yet, and it adds `--cpu` when the installed torch can't
run on the card (as with this box's GTX 1060). Extra `main.py` arguments go after `--`, as in
`./start_comfyui.sh -- --port 8190`.

**Updating to the newest version of this fork:** `UPDATE_COMFYUI.bat` / `./update_comfyui.sh`
(`tools/update.py`, standard library only). In a Git clone it runs `git pull --ff-only` and stops
if tracked files were edited. In a folder downloaded as a ZIP it asks GitHub for the newest commit
of `AI_INFLUENCER_UPDATE_REPO` (default `Kunhah/ComfyUIOFMModel`) on `AI_INFLUENCER_UPDATE_BRANCH`
(default `main`), downloads that ZIP and overwrites only the files whose content changed; the
installed commit is kept in `.update_version`. Anything not in the repo (venv, models, input,
output, `.env`, your own workflows) is untouched, files deleted upstream are left behind, and a
shipped workflow edited locally is replaced. While the repo is private, ZIP users need a GitHub
token with read access in `GITHUB_TOKEN`. The next start installs any changed `requirements.txt`.

## Credentials (API keys)

Every key and per-user setting lives in one file, `.env` at the ComfyUI root. It is gitignored and
never shipped: each user sets their own. The easiest way to fill it in or change it later:

```bash
python custom_nodes/ai_influencer_toolkit/tools/configure.py          # walk through every setting
python custom_nodes/ai_influencer_toolkit/tools/configure.py --show   # what is set (keys masked)
python custom_nodes/ai_influencer_toolkit/tools/configure.py --set HF_TOKEN=hf_...
```

It creates `.env` from `.env.example` the first time, keeps any other lines you have in it, and
makes it readable by you only. The toolkit reads `.env` again whenever it changes, so no restart is
needed. Pressing Run on a workflow whose nodes need a key that isn't set (`FAL_KEY` for Workflow
06's fal.ai nodes) opens a window to paste it (`web/key_helper.js`, server side `key_routes.py`).
It is saved to `.env` the same way, and the run goes ahead. A variable exported in your shell
wins over `.env`. You only need the keys for the parts you use:

| Setting | Needed for | Get it at |
|---|---|---|
| `COMFY_API_KEY` | Workflows 01, 04, 05 (Krea / GPT Image / Seedance partner nodes) when queued by script; in the browser, signing in to your Comfy account is enough | <https://platform.comfy.org/login> |
| `FAL_KEY` | Workflow 06 and `train_lora.py` / `eval_lora.py` (fal.ai) | <https://fal.ai/dashboard/keys> |
| `VAST_API_KEY` | renting GPU pods: `vast_pod.py`, `comfy_pod.py`, the Pod queue | <https://cloud.vast.ai/manage-keys/> |
| `HF_TOKEN` | the pod workflows: gated Krea 2 download and the private transfer repo (write access) | <https://huggingface.co/settings/tokens> |
| `AI_INFLUENCER_HF_REPO` | optional: that private repo as `user/name`; blank = `<you>/<project>-lora-transfer` | |
| `AI_INFLUENCER_LORA_PATH` | the ComfyUI pod: which trained LoRA in that repo to load | printed by `vast_pod.py watch` |
| `AI_INFLUENCER_DEFAULT_PROJECT` | optional: character used when a node's project field is blank | |
| `FAL_BUDGET_USD`, `COMFY_SERVER_URL`, `COMFY_API_BASE_URL` | optional limits / addresses | |

Krea, GPT Image 2.5 and Seedance go through ComfyUI's own `/proxy/<provider>/...` backend with
**one Comfy account key** billed against your Comfy credit balance, so there is no separate
`KREA_API_KEY` / `OPENAI_API_KEY` / `SEEDANCE_API_KEY`. Each partner node shows a live price
badge before you click Queue.

The whole list, with what each one is for, is in
`custom_nodes/ai_influencer_toolkit/env_config.py`. It is the only place the toolkit reads them.

### Keeping keys out of Git

`.env` and `.env.*` are gitignored (except `.env.example`, which holds no values). As a second
net, the repo ships a pre-commit hook that refuses a commit that stages `.env`, contains any value
from your `.env`, or contains something shaped like a Hugging Face, vast.ai, fal, OpenAI or Comfy
key. Turn it on once per clone:

```bash
git config core.hooksPath .githooks
```

Your own characters stay out as well. The repo ships code and workflow JSON only: `input/`,
`output/`, `models/`, `logs/`, `custom_nodes/ai_influencer_toolkit/datasets/*.json` (except
`example.json`) and any image, video, model weight or archive inside the toolkit are ignored, and
the hook also refuses a newly added one anywhere. The two small face-scoring models
`face_score.py` needs are downloaded from the OpenCV model zoo on first use and checked by hash.

## Project structure

ComfyUI sandboxes node file I/O on purpose: `LoadImage` only browses `input/`, and
`folder_paths.get_save_image_path` (used by every Save node, including this toolkit's) refuses
to write outside `output/` — see `folder_paths.py`. Rather than working around that, character
projects are split across the two sandboxes it already provides:

```
input/ai_influencer/<project>/        things you feed INTO generation
    references/canonical/                approved identity photos
    references/candidate/                rejected/unreviewed candidates (optional convention —
                                          copy files here yourself; nothing writes here automatically)
    poses/                                pose reference images
    scenes/                               scene / style reference images
    dataset/                              LoRA training images (used to score checkpoints, Workflow 07)

output/projects/<project>/            things generation PRODUCES
    character.json                       non-secret character description (see below)
    prompts/                             optional saved reusable prompt text (manual)
    outputs/images/candidates/           candidates; Workflow 09b: one folder per queue, 9b_<seed>/
    outputs/images/final/                approved images; Workflow 13's winners; reference_set/ (Workflow 01 example photos)
    outputs/images/sheets/<sheet>/       Workflow 01's character sheets (stage=character_sheet)
    outputs/images/sheet_crops/          the sheets cut into one picture + caption per view

user/ai_influencer/sheet_templates/  your own character sheet templates (listed in every sheet dropdown)
    outputs/images/misc/                 anything else
    outputs/videos/                      Seedance output (Save & Log Video)
    lora_tests/<timestamp>/              checkpoint test grid.png + report.json (Workflow 07)

logs/generations.jsonl                One JSON line per API call (see Cost control below).
```

Both `AIInfluencerLoadCharacter` and `AIInfluencerSaveImage`/`AIInfluencerSaveVideo` create this
tree automatically the first time you use a given project name — you don't need to pre-create it,
though `tools/new_project.py <name>` does it upfront if you prefer:

```bash
python custom_nodes/ai_influencer_toolkit/tools/new_project.py jane_doe \
    --trigger "photo of sks woman" \
    --feature "olive skin, dark brown wavy hair, brown eyes" \
    --feature "small scar above left eyebrow"
```

### character.json

```json
{
  "name": "jane_doe",
  "trigger": "photo of sks woman",
  "permanent_features": ["olive skin, dark brown wavy hair, brown eyes"],
  "variable_features": [],
  "default_generation_preferences": {},
  "local_lora": "lina/l1na_000002500.safetensors",
  "local_lora_strength": 1.0
}
```

`local_lora` / `local_lora_strength` are only used by the local Krea 2 pipeline. They are set by
`tools/pick_checkpoint.py` (or by Workflow 07's `save_best_to_character` switch), and the
*Apply Character LoRA* node in Workflows 08/09 reads them. Pick a better checkpoint later, and
every local workflow follows without edits.

Non-secret only — no API keys, no personal data beyond what you choose to describe for a
*fictional* character. `AIInfluencerLoadCharacter` reads this and outputs `trigger +
permanent_features` joined as one `identity_prompt` string, meant to prepend to every generation
prompt so the wording of "who this person is" never drifts between the 500 Krea calls a month.

## Character identity strategy

There's no trainable LoRA slot in any of these cloud APIs, so consistency is engineered instead:

1. **Wording consistency** — `character.json`'s `trigger` + `permanent_features`, prepended to
   every prompt via `AIInfluencerPromptBuilder`.
2. **Krea `style_reference`** — chain up to 10 images (Workflow 05) to steer style/composition.
   This is a *style* reference per Krea's own API, not a guaranteed face-identity lock.
3. **GPT Image 2.5 multi-image edit** — the strongest identity tool available: every request in
   Workflow 01 carries the front portrait plus the character sheets made before it, so the sheets
   and example photos converge on one face instead of drifting request by request.
4. **Seedance `reference_images`** — carry canonical photos alongside the starting frame into
   video generation (Workflow 04) so identity/clothing/environment stay locked across motion.

## Optional: real LoRA training (fal.ai)

Reference-image conditioning (above) is the ceiling for identity consistency without training
anything. If you want a real trained LoRA instead — e.g. because you already have a solid set of
real photos of the character — Workflow 06 adds that via **fal.ai**, a different provider from
Krea/GPT-Image/Seedance/OpenRouter, with its **own separate account and bill** (not routed through
your Comfy credits).

Why fal.ai and not a custom node: there's no official ComfyUI node for LoRA training on any
provider, and fal's own upload API isn't documented precisely enough to safely hand-roll with your
money on the line — so this uses fal.ai's own official `fal-client` Python SDK directly inside
the toolkit's nodes, not a third-party ComfyUI node.

**Setup (one time):**
```bash
pip install -r custom_nodes/ai_influencer_toolkit/requirements.txt   # installs fal-client
```
Get a key at <https://fal.ai/dashboard/keys> and set it with `tools/configure.py` (or add `FAL_KEY=...` to `.env`).

**Verified pricing (fal.ai's own pricing pages, not estimated):**
- Training (`fal-ai/flux-lora-portrait-trainer`): **$0.0024/step**, 1000-step minimum billed
  (~$2.40), default 2500 steps (~$6) — a one-time cost per character.
- Generation (`fal-ai/flux-lora`): **$0.035/megapixel** per image.

**Workflow 06** (`06_train_and_generate_character_lora.json`) has two halves:
- **Train** (left, run once): load your canonical reference photos (from Workflow 01, or your
  own existing photos), batch them, and train. The resulting LoRA URL is written straight into
  `character.json` — you never copy/paste it.
- **Generate** (right, run anytime after): Load Character now also outputs `lora_url` read
  from `character.json`, wired straight into the "Generate with Character LoRA" node.

This is a genuinely separate image-generation backend from Krea/GPT-Image (FLUX.1 [dev], not
Krea 2), so treat it as an alternative for this one character's shoots, not a drop-in replacement
— quality/style will differ from Krea 2 Large.

## Krea 2 character LoRA on fal.ai (no GPU rental)

`fal-ai/krea-2-trainer` trains a Krea 2 LoRA in the cloud ($0.003/step; trains on RAW) and
`fal-ai/krea-2/turbo/lora` generates with it ($0.01/megapixel). The trainer exposes steps,
learning rate (Krea recommends 3e-4 to 7e-4) and resolution (768/1024), and returns **only the
final weights**, so comparing step counts means one short run per step count.

1. **Curate + caption** (the biggest quality lever). A manifest in
   `custom_nodes/ai_influencer_toolkit/datasets/<project>_v1.json` lists the chosen images and
   captions: trigger first, then everything that should stay changeable (clothing, pose,
   expression, background, light), never the fixed traits (hair, face). Build and check it:
   ```bash
   T=custom_nodes/ai_influencer_toolkit/tools
   python $T/build_dataset.py custom_nodes/ai_influencer_toolkit/datasets/MyCharacter_v1.json
   python $T/check_dataset.py input/ai_influencer/MyCharacter/dataset --trigger mychr1
   python $T/face_score.py input/ai_influencer/MyCharacter/dataset   # face consistency
   ```
   `face_score.py` uses OpenCV's SFace recognizer (CPU, `pip install opencv-python-headless`,
   models in `custom_nodes/ai_influencer_toolkit/models/`).
2. **Train a step sweep** (one upload, runs in parallel, asks before spending, never re-trains a
   tag that already exists):
   ```bash
   python $T/train_lora.py MyCharacter --trigger mychr1 --steps 600,1000 --dry-run
   python $T/train_lora.py MyCharacter --trigger mychr1 --steps 600,1000
   ```
   Weights land in `models/loras/<project>/`, records in `output/projects/<project>/lora_runs/`.
3. **Compare** with the same seeds on every LoRA, plus a no-LoRA baseline and a no-trigger "leak"
   row, face-scored against the dataset (cached, so re-runs are free):
   ```bash
   python $T/eval_lora.py MyCharacter --runs <tag>,<tag>
   ```
   Pick the column that keeps her face in every row *and* still follows each prompt. High `leak`
   or dataset outfits/backgrounds showing up in unrelated prompts = too many steps.
4. **Record the winner**, then generate with *Generate with Character LoRA* (base
   `krea-2-turbo`, lora_url from Load Character):
   ```bash
   python $T/train_lora.py MyCharacter --pick <tag> --scale 1.0
   ```

## Running this setup on a rented GPU (vast.ai / RunPod)

Krea 2 needs far more VRAM than a 6 GB card, so training and local generation happen on a rented
pod running the *same* ComfyUI setup: `tools/pod_bootstrap.sh` installs this node pack, workflows
01-09, the project's `character.json` and dataset, and downloads the Krea 2 + SeedVR2 models
(~20 GB). Build a kit for a character with:

```bash
OUT=output/projects/<project>/pod; mkdir -p $OUT/kit/{workflows,project}
cp -r custom_nodes/ai_influencer_toolkit $OUT/kit/
cp -r user/default/workflows/ai_influencer $OUT/kit/workflows/
cp -r input/ai_influencer/<project>/dataset $OUT/kit/
cp output/projects/<project>/character.json $OUT/kit/project/
cp custom_nodes/ai_influencer_toolkit/tools/pod_bootstrap.sh $OUT/kit/
tar -czf $OUT/pod_kit.tar.gz -C $OUT kit && rm -rf $OUT/kit
```

Workflows 01-06 keep running on your own machine (they only call cloud APIs).

`tools/comfy_pod.py launch` does the whole rental end to end (rent → bootstrap → print the ComfyUI
URL → auto-destroy). Add **`--with-minimax`** for Workflow 11: the bootstrap then also pulls the
five MiniMax H3 files (~45 GB).

**The GPU is chosen automatically for each job** (`tools/gpu_select.py`). Each job has hard
requirements, and every vast.ai offer that meets them is ranked by *estimated total cost*: $/hr ×
(time to download the models on that host's line + job time scaled by the card's speed), so a
fast card on a fast line often beats a cheaper slow one.

| Job | Command | Requirements |
|---|---|---|
| LoRA training (Krea 2 RAW, AI Toolkit) | `vast_pod.py launch` | 32 GB+ VRAM, Ampere or newer, 150 GB disk |
| Krea 2 generation (07-10, 09b, 12) | `comfy_pod.py launch` | 24 GB+ VRAM, Ampere or newer, 60 GB disk |
| MiniMax H3 video (11) + Krea 2 | `comfy_pod.py launch --with-minimax` | 32 GB+ VRAM, Blackwell (nvfp4), 200 GB disk |

See the ranking before renting anything with `vast_pod.py search`, `comfy_pod.py search` or
`comfy_pod.py search --with-minimax`; each line shows the GPU, $/hr, speed, line speed and the
estimated hours and dollars. Every limit can be overridden (`--vram`, `--disk`, `--max-price`,
`--min-inet`, `--min-cc`), and `--gpu RTX_4090` pins one model, which also lifts the Blackwell
requirement: `--with-minimax --gpu RTX_4090` is a valid, slower choice. The Pod queue uses the same
selection. Everything that runs on the pod
is local open weights, so nothing generated there is sent to a provider to be moderated.

## Local Krea 2 pipeline (character LoRA)

A LoRA is only as good as its dataset, and the last checkpoint is not automatically the best
one. This pipeline covers both.

### 1. Dataset

- **26-34 strong images** of the same face: different angles, expressions and lighting, mixing
  clean studio shots with blurry phone-style shots. Quality beats quantity: every image must be
  on point, because the LoRA learns exactly what you give it.
- One `.txt` caption per image (same file name), containing your trigger word.
- **Trigger word:** one made-up word the model doesn't already know, e.g. `l1na` (with a one
  instead of an i), so it only means your character.
- Check it before paying for GPU time:

  ```bash
  python custom_nodes/ai_influencer_toolkit/tools/check_dataset.py path/to/lina --trigger l1na
  ```

  It fails on missing/empty captions, unreadable files and exact duplicates, and warns on
  count, resolution, near-duplicates and captions without the trigger. It can't tell whether it's
  the same face everywhere, so check that by eye. Also copy the images to
  `input/ai_influencer/<project>/dataset/`; Workflow 07 uses them for scoring.

### 2. Training (Ostris AI Toolkit on RunPod, ~$3-5)

1. RunPod → GPU **RTX 6000 Pro** (or **RTX 5090**) → template **Ostris AI Toolkit** (official).
   Storage: **volume disk, not a network volume, 150 GB** (room for all the checkpoints) →
   *Deploy On-Demand*. Wait a few minutes for it to boot.
2. *Connect* → HTTP service link opens the AI Toolkit UI (the template's default password is
   `password`).
3. *Datasets* → new dataset (e.g. `lina`) → drag in all images **and** caption files.
4. *New Job*. The settings that matter:

   | Setting | Value | Why |
   |---|---|---|
   | Model architecture | **Krea 2 (raw)**, never Turbo | Training on Turbo lowers quality. Train on RAW, generate on Turbo. |
   | Trigger word | `l1na` | see above |
   | Steps | 3000 | |
   | Optimizer / LR | AdamW8bit / default (1e-4) | |
   | Batch size | 1 on an RTX 5090 (out-of-memory otherwise); default is fine on a 6000 Pro | |
   | Save every | 250 steps | 12 checkpoints to compare |
   | **Max step saves to keep** | **100** | The default (4) **deletes older checkpoints during training**. You want all 12. |
   | Everything else | default, correct dataset selected | |

   Or generate the whole job config with those values and paste/run it:

   ```bash
   python custom_nodes/ai_influencer_toolkit/tools/make_ai_toolkit_config.py lina \
       --trigger l1na --dataset-path /path/on/pod/to/lina -o lina_krea2.yaml
   # add --low-vram on an RTX 5090
   ```

   The script refuses combinations where `max saves x save every < steps`, i.e. where checkpoints
   would be deleted.
5. *Create job* → play. Krea 2 training takes longer than most models. When it's done, **stop
   the pod** (it bills per hour) after downloading the checkpoints.

### 3. Pick the best checkpoint (Workflow 07)

The sweet spot is usually somewhere between 2,000 and 3,000 steps, and past it the LoRA overfits:
the face stays, but it starts copying the dataset's clothes, poses and backgrounds. So:

1. Download the **last 4 checkpoints** (more if you want) into `models/loras/<project>/` on the
   ComfyUI pod. AI Toolkit names them `l1na_000002250.safetensors`, ..., and the final one
   `l1na.safetensors`.
2. Open **`07_lora_checkpoint_tester.json`**. Set `lora_filter` (`l1na`), `last_n` (4) and
   `final_step` (3000), then queue. The **LoRA Checkpoint Tester** node renders every test prompt with
   **the same seed on every checkpoint** and returns one labelled grid: columns = checkpoints (plus
   an optional no-LoRA baseline, and optionally several strengths, e.g. `0.8, 1.0`), rows = prompts.
   Only the checkpoint changes between cells, so the difference you see is the training.
3. Use test prompts with outfits and places that are **not** in the dataset. The winner keeps the
   face stable in every row **and** still follows each prompt.
4. **Optional scores:** connect *Load CLIP Vision* (`sigclip_vision_patch14_384`) to the tester.
   Each column then gets `id` (similarity to the dataset centroid, higher = more like the character)
   and `copy` (similarity to the single closest training image). When `copy` keeps rising while `id`
   stays flat, the later checkpoints are memorizing. CLIP is a general image model, not a face
   recognizer, so the scores are a tie-breaker next to your own eyes, not a verdict.
5. Every run is kept in `output/projects/<project>/lora_tests/<timestamp>/` (`grid.png` +
   `report.json` with settings, prompts and scores) and logged to `logs/generations.jsonl`, so you
   can compare runs later. Record the winner:

   ```bash
   python custom_nodes/ai_influencer_toolkit/tools/pick_checkpoint.py lina --filter l1na            # list checkpoints + past test scores
   python custom_nodes/ai_influencer_toolkit/tools/pick_checkpoint.py lina --filter l1na --step 2500
   python custom_nodes/ai_influencer_toolkit/tools/pick_checkpoint.py lina --filter l1na --final     # the unnumbered last one
   ```

   Sometimes the last checkpoint really is the best, but you only know that after testing.

### 4. Generate (Workflow 08)

**`08_krea2_local_master.json`**: Krea 2 **Turbo** (8 steps, cfg 1) + *Apply Character LoRA* (the
checkpoint from step 3) + the Prompt Builder with the trigger/features from character.json.

- **03 Base image** → `outputs/images/candidates/`. With groups 04-08 muted (right-click group →
  *Set Group Nodes to Never*), this is the simple "base" workflow.
- **04 Skin polish**: upscales 1.25x and runs a low-denoise (0.3) second pass with a
  skin/eye/hair detail prompt. Lower the denoise if the face drifts, raise it (max ~0.35) for more
  detail.
- **05 SeedVR2 upscale** (2x, the official SeedVR2 3B int8 template unpacked, LAB color match) →
  `outputs/images/final/`, with a before/after slider.

### 5. Background blur (always on)

Every influencer image should have an out-of-focus background, and prompting for "shallow depth
of field" only gets you there *sometimes*: the model decides, and the amount drifts between
images. So the blur is done after generation instead, by the **Blur Background** node
(`nodes_background.py`), which is wired into Workflow 08 (group 06) and Workflow 09 (group 05)
and enabled by default (and into Workflow 09b's output, group 08):

1. Core ComfyUI's *Load Background Removal Model* → *Remove Background* (BiRefNet,
   `models/background_removal/birefnet.safetensors`) produces a subject mask.
2. Everything outside that mask is blurred. Three details keep it looking like a lens rather than
   a smudge: the background is blurred with the subject excluded and renormalized, so the
   subject's colors don't bleed into a halo around them; highlights are blurred in linear light so
   bright spots bloom into bokeh (`bokeh_highlights`); and the mask edge is feathered so hair and
   shoulders fade into the blur instead of looking cut out.
3. It runs **last**, after the SeedVR2 upscale, so nothing sharpens the background again.

`blur_amount` and `feather` are percentages of the image's short side, so the same numbers give
the same look at any resolution: `blur_amount` ~0.8 is subtle, 1.5 (the default) is a portrait
lens, 3+ is extreme. `subject_edge_shift` moves the sharp/blurred boundary a little if the mask
cuts too tightly. Set `blur_amount` to 0, or mute the group, for a sharp background.

The prompt's camera field in Workflows 05, 08 and 09 also asks for a shallow depth of field,
which helps the model compose for it — but that's a hint, and the node is what makes it certain.
`tools/pod_bootstrap.sh` downloads the BiRefNet file along with the other models. The cloud workflows (04, 05) only have the prompt wording; to blur those too, add the same three
nodes after their Save node (BiRefNet runs locally and is small, but it is a local model, which is
the one thing the cloud pipeline otherwise avoids).

### 6. Camera imperfections (always on)

The other half of "doesn't look AI". A diffusion model outputs a picture with no sensor noise,
perfectly aligned color channels, even corner-to-corner exposure and textbook white balance —
which is exactly what nobody's camera produces. Two things fix that, and both are automatic:

**In the prompt.** `AIInfluencerPromptBuilder` always appends a fixed block
(`MANDATORY_IMPERFECTIONS` in `nodes_character.py`): candid mundane photography, slightly
off-center framing, a tilted horizon, uneven light with underexposed areas, skin texture and small
asymmetries, flyaway hairs, creased clothing, an in-between expression, lens softness, ISO grain,
chromatic aberration, off white balance — and no airbrushing, HDR glow or oversharpening. It is
**not a node input**, so it can't be edited away in one workflow and drift from the others; the
`realism_instructions` field is still there for per-shot additions, and starts empty.

**After generation.** Prompt wording is a hint the model half-follows, so the artifacts themselves
are applied by the **Camera Imperfections** node (`nodes_camera.py`), wired into every image
workflow (05, 06, 08, 09, 09b, 10) as the last step, after any upscale and blur so nothing
sharpens the grain away:

- **ISO grain** whose clump size is a percentage of the short side (so it survives an upscale),
  weighted towards the shadows and midtones the way sensor noise actually is, plus a separate,
  blotchier chroma component that doesn't shift overall brightness.
- **Chromatic aberration**: the red channel scaled up and the blue channel down around the center,
  so the corners fringe like a cheap lens.
- **Vignette**: an r⁴ (cos⁴) corner falloff, applied in linear light.
- **Color jitter**: a small per-image white balance and exposure drift off the seed, so a batch
  doesn't share one identical color grade.

`iso_grain` 0.015 is a clean daylight shot, 0.025 (the default) an ordinary phone photo, 0.05+
pushed high ISO. Set it to 0, or mute the node, for a clean image. It's pure torch — no model, no
download — which is why the cloud workflows get it too.

It is deliberately **not** in Workflow 01 (its front portrait, reference set and sheets are
identity references and LoRA training pictures: every later generation is conditioned on them, and
grain in a dataset gets learned) or in Workflow 07 (checkpoint comparison).

**Images you already have** — generated before this existed, or from somewhere else — get the same
pass two ways:

- **Workflow 12** (`12_camera_pass.json`): upload or drag in any picture, optional
  background blur (bypassed by default), camera pass, before/after slider, save and log. Re-queue
  for a different grain roll.
- **A whole folder at once**, without ComfyUI:

  ```bash
  python custom_nodes/ai_influencer_toolkit/tools/camera_pass.py output/projects/lina/outputs/images/final
  ```

  Same code as the node. Each file gets its own seed from its name, so a folder doesn't come out
  sharing one grain pattern, and results are written as `<name>_camera.<ext>` next to the inputs
  (`--in-place` overwrites, `--out DIR` collects them elsewhere, `--device cpu` if the GPU can't
  run torch). The knobs match the node's: `--iso-grain`, `--grain-size`, `--chroma-noise`,
  `--chromatic-aberration`, `--vignette`, `--color-jitter`.

### 7. Pose transfer (Workflow 09) and edits

**`09_krea2_pose_transfer.json`** takes a reference photo from `input/ai_influencer/<project>/poses/`
and generates your character in the same pose and framing. This ComfyUI build has **no Krea 2
ControlNet**, so it's image-to-image: `denoise` 0.8 keeps pose and framing, 0.7 also keeps
colors/lighting (and pulls the face toward the reference), 0.9 keeps only the rough layout. It
transfers composition, not an exact skeleton.

**Edits** of a finished image (hair color, outfit, background, one prompt) have no workflow of
their own any more: Workflow 03 (GPT Image 2.5 edit) was removed, since generation is local now.
Core ComfyUI has no Krea 2 edit model. (AI Toolkit can train Krea 2 edit LoRAs, but running them
needs a third-party node pack, which this setup doesn't install.)

### 8. Preview small, then 4K on confirm (Workflow 10)

Rendering everything at full size to find out you don't like the pose is slow. **Workflow 10**
splits it: cheap previews first, the expensive pass only on the one you approve.

1. **Preview** — queue it as it opens. Group 03 renders **4 candidates at ~1 MP** (8 steps, no
   polish, no upscale, no blur) into `outputs/images/candidates/`, shown side by side. Queue again
   for 4 more, or lower the `megapixels` on *Preview size* for faster, rougher looks.
2. **Confirm** — when one is right: mute group 03 (right-click → *Set Group Nodes to Never*) and
   enable groups 04 and 05 (*Always*); in *The candidate you picked* click *choose file to upload*
   and pick that file from `outputs/images/candidates/` (or drag it onto the node); queue again. SeedVR2
   upscales it **4x** — a ~900x1200 preview becomes ≈3600x4800, comfortably 4K — the background
   blur runs, and it lands in `outputs/images/final/`.

**Why the 4K stage upscales the approved image instead of re-generating it at 4K:** a diffusion
model given the same seed at a different resolution produces a *different* picture — different
pose, different framing, different hands. "The same image, bigger" is only possible by taking the
pixels you approved and adding detail to them, which is what SeedVR2 does. Re-rolling at 4K is
still an option, of course; it just isn't the same image.

ComfyUI has no pause-for-approval node, so the mute/unmute step is the approval. Workflow 08 is
the opposite trade: one queue, generate → polish → 2x upscale → blur, no preview stage. Both save
into the same project folders, and the preview PNGs carry the full workflow, so dragging one back
onto the canvas restores its exact seed and prompt.

### 9. Queue everything locally, run it all on a pod with one button (Pod queue)

The pod bills by the hour, so set up your jobs on your own PC and rent the GPU only for the part
that needs it. The node pack adds a **Pod queue** pill at the bottom right of the canvas, and a
*Pod queue* button in the top bar.

1. Open the pill and tick **Hold for the pod**. From now on, ComfyUI's normal **Queue** button
   stores the job in the pod queue instead of running it on this PC. The batch count and seed
   randomizing work as usual, so *Queue* with batch count 4 holds 4 jobs with 4 seeds.
2. Set up the workflow and press Queue. Change the prompt, the reference image, the LoRA
   strength, anything, and press Queue again. Repeat this across as many workflows as you like.
   The held list is saved in `user/ai_influencer/pod_queue.json`, so it survives a restart.
3. Press **Start pod & run all**. It:
   - checks the pod kit on your HF repo. A new pod installs *that* kit, not the files on this PC,
     so if the node pack, workflows, `pod_bootstrap.sh` or `character.json` changed since it was
     uploaded, it rebuilds `output/projects/<project>/pod/pod_kit.tar.gz` and uploads it first;
   - rents a pod with `comfy_pod.py launch`, or uses the one already running, adding
     `--with-upscale` / `--with-minimax` when a held job uses SeedVR2 / MiniMax H3;
   - waits until the pod has finished downloading models and loading the node pack;
   - checks every held job against the pod's nodes and model files. A job that uses something the
     pod doesn't have is skipped and left in the list, with the reason in the log;
   - uploads the input images the jobs use, under the same names, so the jobs run unchanged;
   - makes every image a Save node writes go through the **background blur and then the camera
     pass as the very last step**, adding whichever is missing (the panel shows it when that
     happens, and warns if one is turned down to 0);
   - queues everything and downloads each result as it finishes into **this PC's `output/`
     folder, at the same path it would have had here**. Downloads are retried, and whatever
     finished is always fetched before the pod is destroyed, also after Stop or an error. If a
     file still can't be downloaded, the pod is **kept** (until its hard time limit) and the log
     says which file, instead of destroying it with your image on it;
   - **destroys the pod** when everything is done. It also does this if the run fails or you
     press Stop, unless you untick *destroy the pod when done*. The pod's own `--hours` timer
     is still there as a backstop.
4. Finished jobs leave the list. Failed ones stay, so you can fix them and run again.

To run the held jobs on this PC instead, press **Run here**. It sends each one to this ComfyUI's
normal queue (Hold doesn't affect it), with no pod, kit upload or download involved. A job leaves
the list as soon as ComfyUI accepts it. After that, its progress and any error show in ComfyUI's
own queue. A job ComfyUI rejects, for example because a model file isn't on this PC, stays in the
list and the panel shows the reason.

Untick **Hold for the pod** to run things on this PC normally again. The setting is saved
separately for each address you open ComfyUI at, so it is off on the pod's own ComfyUI.

Needs `VAST_API_KEY` and `HF_TOKEN` in `.env`, and the pod kit on your HF repo, the same as
`comfy_pod.py launch`. The generation log (`logs/generations.jsonl`) is written on the pod, so
held jobs are not in this PC's log.

### Models

Nothing needs downloading until you run a local workflow, and opening one never reports a missing
model: the loaders' dropdowns list every file in `model_catalog.py` whether it is downloaded or not.
**Pressing Run** with a file missing opens the **Models** window
(`web/model_helper.js`, server side `model_routes.py`) and nothing is queued. The window says what
each file is for and how big it is, checks the graphics card ComfyUI is using (too weak, CPU only,
or which Krea 2 version fits), and downloads only what the switched-on nodes load into the right
`models/` folder, with progress. An interrupted download resumes. It's always available from
**Extensions → AI Influencer: Models**. While "Hold for the pod" is on, Run isn't checked, because
the pod has its own models. A workflow queued from a script (`queue_workflow.py`) with a file
missing fails on that loader with ComfyUI's "Model ... not found" error.

**Pictures a workflow names that aren't on this computer** (the example photos in 04, 06 and 11,
or another character's project files) don't show up as missing inputs either
(`web/input_helper.js`, server side `input_listing.py`). When a workflow opens, a Load Image
pointing at `ai_influencer/<other project>/...` is switched to the same file in your project
(`AI_INFLUENCER_DEFAULT_PROJECT`, or the only project there is) when you have it. Otherwise it gets
the grey **Choose a picture** placeholder (`input/ai_influencer_choose_a_picture.png`, made on
startup), and a notice lists those nodes. Pressing Run while a placeholder is still selected names
the node and queues nothing, so no credits go to a placeholder.

From a terminal (or on a pod), `tools/models.py` does the same:

```bash
python custom_nodes/ai_influencer_toolkit/tools/models.py                  # GPU advice + every model, downloaded or not
python custom_nodes/ai_influencer_toolkit/tools/models.py 09b --download   # what 09b needs, and get the missing ones
python custom_nodes/ai_influencer_toolkit/tools/models.py --version nvfp4  # choose the Krea 2 version
```

**Krea 2 versions.** The workflows are saved with the FP8 files. When a workflow opens, the Krea 2
loaders are switched to the version chosen in `AI_INFLUENCER_KREA_VERSION` (set in the Models
window) if that one is downloaded, or else to any downloaded version:

| Version | Files (model + text encoder) | For |
|---|---|---|
| `fp8`, Standard (default) | `krea2_turbo_fp8_scaled` + `qwen3vl_4b_fp8_scaled`, 18.4 GB | most cards with 12 GB+; under 16 GB part of it waits in RAM (slower) |
| `nvfp4`, Small | `krea2_turbo_nvfp4` + `qwen3vl_4b_fp8_scaled`, 12.9 GB | RTX 50-series with 12 GB+ (fast NVFP4 math); slightly lower quality |
| `gguf`, Low memory | `krea2_turbo-Q4_K_M.gguf` ([vantagewithai/Krea-2-Turbo-GGUF](https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF)) + `qwen3vl_4b_fp8_scaled`, 12.7 GB | 8-11 GB cards (RTX 3060 Ti, 3070, 4060, 5060); a 4-bit Krea 2 that fits in 8 GB, with a little less fine detail |
| `bf16`, Full quality | `krea2_turbo_bf16` + `qwen3vl_4b_bf16`, 35.2 GB | 40 GB+ (RTX 6000 Pro class) |

The GGUF file loads with ComfyUI-GGUF's `UnetLoaderGGUF` node instead of `UNETLoader`, so opening
a workflow with that version chosen swaps the loader node too (same MODEL output, links kept). The
text encoder stays the FP8 one: ComfyUI unloads it before the image model runs, so it doesn't
compete for the 8 GB. ComfyUI-GGUF comes from GitHub at the commit pinned in `model_catalog.py`
(`GGUF_COMMIT`) and goes into `custom_nodes/ComfyUI-GGUF/` (git-ignored), with its one requirement
(`gguf`). The Models window or `tools/models.py <workflow> --download` installs it when the GGUF
version is chosen. ComfyUI has to be restarted once afterwards, and until then opening a workflow
keeps the normal loader.

The pod (`pod_bootstrap.sh`) always installs FP8, so jobs held for the pod should use FP8. The whole
list, with sizes and URLs, is `model_catalog.py`:

| File | Folder |
|---|---|
| Krea 2 (one version, above) | `models/diffusion_models/` + `models/text_encoders/` |
| `qwen_image_vae.safetensors` | `models/vae/` |
| `krea2_style_reference.safetensors` | `models/loras/` (09b: follow the reference photos) |
| `sdpose_wholebody_fp16.safetensors` | `models/checkpoints/` (09b: poses) |
| `seedvr2_3b_int8_convrot.safetensors` | `models/diffusion_models/` (Workflow 08 upscale) |
| `seedvr2_ema_vae_fp16.safetensors` | `models/vae/` (Workflow 08 upscale) |
| `sigclip_vision_patch14_384.safetensors` | `models/clip_vision/` (Workflow 07 scoring) |
| `birefnet.safetensors` | `models/background_removal/` (background blur) |
| `minimax_h3_*` (4 files, [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)) + the turbo LoRA ([lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)), 44.4 GB | `models/diffusion_models/`, `text_encoders/`, `vae/`, `loras/` (Workflow 11; 32 GB+ card) |
| your checkpoints | `models/loras/<project>/` |

Workflows 07-09 are generated by `tools/build_local_workflows.py`. Edit that script and re-run it
rather than hand-editing the JSON if you want different defaults.

### Responsible use and account risk

Keep the character fictional. Don't train on or imitate a real person without their consent, and
follow each platform's rules on labelling AI-generated people and content.

**Where a ban can actually come from.** Mature content is not equally risky on every path here —
it matters which pipeline you use:

| Path | Who can act on it | Notes |
|---|---|---|
| Workflows 01, 04, 05 (GPT Image 2.5, Seedance, Krea) | the provider, and your **Comfy account** | requests are proxied through Comfy's backend and billed to one account; the providers moderate prompts *and* outputs under their own usage policies, which do not allow sexual content. This is the realistic way to get flagged. |
| Workflow 06 (fal.ai LoRA training) | fal.ai | separate account and policy; your dataset is uploaded to them. |
| The training pod transfer | Hugging Face | the dataset passes through a private HF repo, which is still subject to their content policy. |
| vast.ai / RunPod | the GPU host | they have their own acceptable-use policy for what you run on the rented machine. |
| Workflows 07-12, 09b, 11 (local Krea 2, MiniMax H3, this toolkit) | nobody | no partner node, no moderation endpoint scoring your prompt or frames, no account to flag. Nothing leaves the machine. |
| Wherever you post | the platform | rules on AI-generated people, nudity and AI labelling. In practice this is where accounts die, not in generation. |

So: **anything mature belongs on the local pipeline**, which is why it exists. If you send it
through Workflows 01, 04 or 05 you are handing it to a moderated API under someone else's terms.

**Not a loophole:** the camera-imperfections pass is aesthetic. It does not strip metadata or
provenance signals, and it is not a way around a platform's AI-labelling rules — label your
content where the platform requires it.

**What the assistant working on this repo will and won't do.** Recorded here at the user's request
so it doesn't have to be restated:

- **Will**: build and maintain the local, unfiltered pipeline; keep content filters out of core
  code; say plainly which paths are moderated and which are not; keep everything local-only, since
  core ComfyUI takes no outbound internet requests beyond user-initiated model downloads.
- **Won't**: write or generate sexual content itself; help evade a provider's moderation,
  detection or account rules; work with a real person's likeness without their consent; or touch
  anything involving minors. Where a request crosses that, it gets said out loud and stops there,
  rather than being quietly done halfway.

## The workflows

All under `user/default/workflows/ai_influencer/`. Each has an in-canvas note explaining its
controls; group frames follow the requested numbering (01 INPUT, 02 CHARACTER IDENTITY, 03
PROMPT, 04 POSE/REFERENCES, 05 KREA GENERATION, 06 CANDIDATE SELECTION, 07 GPT REFINEMENT, 08
VIDEO, 09 OUTPUT) wherever a workflow's stages match that shape.

### 01 — Character Creation (`01_character_creation.json`)

Makes the character sheets and a few example photos her LoRA is trained on, with GPT Image 2.5
only. Two queues:

1. **Queue.** The first node is the only place she is ever described in words. It makes **4 front
   portraits in one call** (`n` = 4, saved to `outputs/images/candidates/`). *Pick Candidate* is
   0, so nothing after it runs.
2. **Set *PICK* to 1-4 and queue again.** The candidates call is not repeated: its inputs didn't
   change, so ComfyUI reuses the result (its seed is `fixed` for that reason; GPT itself ignores the
   seed, so change it by hand for 4 new ones). Already have a front photo, e.g. from ChatGPT?
   Ctrl+B *OR your own front photo* and load it: it wins over the candidates.

Everything else then runs by itself: **8 character sheets** and 4 example photos, each request
carrying the pictures that pin her down. The face-turnaround sees the front portrait; full-body-360,
expression-grid, upper-body, lighting-grid and expression-grid-extra see the front + face-turnaround;
outfit-grid, pose-grid and the example photos also see full-body-360. A sheet shows her in 7-9
views, so it is a lot of identity for one (billed) context image.

| Sheet | Changes | From |
|---|---|---|
| face-turnaround, full-body-360, upper-body | the angle | Character Sheet Studio |
| expression-grid | the expression | Character Sheet Studio |
| outfit-grid | the clothes (6 everyday outfits) | this pack, `sheet_templates/` |
| pose-grid | the pose (8: walking, sitting, arms crossed, crouching, ...) | this pack |
| lighting-grid | the light (window, midday sun, golden hour, overcast, lamp, neon) | this pack |
| expression-grid-extra | 8 more expressions | this pack |

All eight are sent unchanged, 16:9 at 3840x2160, and follow the studio's rules: one thing changes
per sheet, everything else is locked, and she is never described in words (the pictures are the
only identity source). The studio's templates are read from
`character-sheet-studio/.claude/skills/character-sheet-studio/templates/`; only the templates are
used, not its wizard (which uploads to WaveSpeed and has an agent look at every photo). Saved to
`outputs/images/sheets/<sheet>/`.

**Slicing into training pictures.** A LoRA trained on whole sheets learns to draw sheets, so each
sheet goes through *Slice Character Sheet*: it finds each figure on the grey background (a hand
reaching over a panel line stays with its figure; a panel it can't separate falls back to its grid
position), crops it, and writes a caption: the trigger word from `character.json` (or
`[trigger]`) plus that panel's wording from the template, e.g. "l1na, … lit by a warm tungsten table
lamp at frame-right, plain grey studio background". The light, outfit or pose is in the caption, so
the LoRA learns it as a variable rather than as part of her. About 60 crops per character, in
`outputs/images/sheet_crops/<sheet>_<time>/` as `00.png` + `00.txt` (00 is the large view). Grid
panels come out around 640x1080, fine for a LoRA but smaller than the example photos. The captions
assume GPT kept the template's panel order, so look at the crops before copying them into
`input/ai_influencer/<project>/dataset/`. For sheets made earlier:

```bash
python custom_nodes/ai_influencer_toolkit/tools/slice_sheets.py \
    output/projects/lina/outputs/images/sheets/pose-grid --project lina
```

The 4 example photos change outfit, place and light, and are saved to
`outputs/images/final/reference_set/`; copy an *Example* node + its Save node for more.

**Make it yours.** Nothing in Workflow 01 is fixed to one character or one taste:

- **Which sheets:** each *Character Sheet Prompt* has a dropdown of every template; mute a sheet's
  GPT node (Ctrl+M) to skip it, its save and its slicer.
- **Your own sheets:** *Custom Character Sheet* (group 02b, muted until you switch it on) takes a
  title, the large view, and one panel per line: outfits, poses, lights, hairstyles, props,
  anything. Describe what changes, never the person. The grid is sized from the number of lines.
  Its *save_as* keeps it as a template in `user/ai_influencer/sheet_templates/<name>.json`, which
  every dropdown lists (press R). A file there with a built-in's name replaces the built-in, which
  is how to change e.g. the outfit list. Hand-written JSON in the same format works too.
- **Which pictures each request sees, quality, size, how many candidates:** rewire the GPT node's
  `image_1..` inputs (each is billed), or change its `quality` / `size` / `n`.
- **Captions and cutting:** each slicer has *caption_prefix* (default: the trigger word),
  *caption_suffix*, *threshold* and *padding*. The slicer reads the layout and captions from the
  template it is wired to, so custom sheets are cut and captioned correctly too. The CLI takes
  `--template my.json`, `--caption-prefix`, `--caption-suffix`, `--threshold`, `--padding`.
- **Wording:** all prompts refer to "the person", so the character can be anyone. Only the first
  node describes them.

No blur and no camera pass in this workflow, on purpose (see
[Camera imperfections](#6-camera-imperfections-always-on)). About **$0.85 per character** at medium
quality (4 candidates ≈ $0.06, 8 sheets ≈ $0.55, 4 examples ≈ $0.24); slicing is local and free. Each
node shows its price badge.

### 02 / 03 — removed

The Krea main generator (02) and GPT Image refinement (03) were removed: generation is local
(Workflows 08-10, 09b) and character creation is all in 01. Workflow 05 still chains the cloud
Krea → GPT → Seedance path if you want it. Both files are in git history.

### 04 — Seedance Image-to-Video (`04_seedance_image_to_video.json`)

Takes an approved final still (plus an optional extra identity reference) into Seedance 2.5's
Reference-to-Video mode. Default motion prompt:

> She looks toward the street, naturally brushes one strand of hair behind her ear, then gives a
> subtle smile. Slight handheld smartphone camera movement. Preserve identity, clothing,
> environment and facial structure.

Configurable against what the current API actually supports: resolution (up to 1080p on 2.5;
switch the model dropdown to Seedance 2.0 for 4K), duration (up to 30s on 2.5), aspect ratio
("adaptive" keeps the still's own framing). Switching the model dropdown also exposes Seedance
2.0 / 2.0 Fast / 2.0 Mini for other cost/quality tradeoffs.

### 05 — Automated Production (`05_automated_production.json`, optional)

The full chain: Prompt → (optional OpenRouter expansion) → Krea → save candidate → **[ Use GPT
refinement: OFF ]** → save final → **[ Generate video: OFF ]** → save video. The two expensive
stages are off by default using ComfyUI's own node **Mode** (Never/Always) — the same mechanism
official ComfyUI templates use for optional branches:

- GPT Image 2.5 node: Mode = Never by default. Off, its Krea input passes straight through as the
  final image (same IMAGE type in and out, so bypass is a clean no-op). Turn on with right-click →
  Mode → Always.
- Seedance node **and** its Save & Log Video node: **both** Mode = Never by default. Unlike the
  image stage, IMAGE→VIDEO has no type-matched passthrough, so enabling only one of the pair would
  error on a missing input — enable both together.

Krea generation and the candidate save always run (that's the cheap, ~$0.03–0.06/image stage you
want by default).

### 06 — Train & Generate with a Character LoRA (`06_train_and_generate_character_lora.json`, optional)

See [Optional: real LoRA training (fal.ai)](#optional-real-lora-training-falai) above — a
different provider/account from the other five workflows, used only if reference-image
conditioning isn't tight enough for you.

### 07 / 08 / 09 / 10 — Local Krea 2 (checkpoint tester, master, pose transfer, preview→4K)

See [Local Krea 2 pipeline](#local-krea-2-pipeline-character-lora).

### 09b — Fixed location (`09b_krea2_fixed_location.json`)

The workflow for "her, in *this* place". You give it a real photo of a location, paint where she
should stand, and only that painted area is generated: `ImageCompositeMasked` pastes the result
back inside the mask, so every pixel of your location photo outside it stays bit-identical. Her
face comes from the character LoRA plus reference photos, and her pose (optionally) from a
skeleton extracted from any third photo — nothing else of those photos leaks in.

**Set once, then never again:** group 01 (the three Krea 2 loaders, *Apply Character LoRA* with
your project name, the image-reference LoRA), group 04 (sampler: 8 steps, cfg 1, denoise 1.0),
group 05 (mask grow/feather, the composite), group 08 (the filters), and the project name on the
*Save & Log Image* node.

**Per picture, six things:**

1. **Group 02 — location.** Load the photo. Right-click it → **Open in MaskEditor**, paint where
   she goes (her whole body plus a little room), Save. The shape you paint is the size she comes
   out: paint her small in the frame and she is small in the frame.
2. **Group 06 — identity.** A clear photo of her in *Reference 1*. *Reference 2* is bypassed;
   Ctrl+B it to add an outfit shot or a second angle.
3. **Group 07 — pose (optional).** Any photo of anyone in the pose you want. It is reduced to an
   OpenPose skeleton before it reaches the model, so no face, clothes or background travel with
   it. Ctrl+B the *Skeleton* node to drop the pose entirely.
4. **Group 03 — prompt.** Fill `subject_action`, `clothing`, `environment`, `lighting`. Leave the
   pose out of `subject_action` (the skeleton owns it) and keep `environment` describing the real
   location so the lighting agrees. Keep the standing "Picture 1 is her, Picture 3 is the
   skeleton" line in `extra`. You never type anti-AI wording: the Prompt Builder appends the
   mandatory imperfection block itself (see
   [Camera imperfections](#6-camera-imperfections-always-on)).
5. **Queue.** Group 08 then runs on its own: background blur → camera pass (ISO grain, corner
   fringing, falloff, white balance drift) → saved to `outputs/images/final/` and logged.
6. **Want another grain roll on the same picture?** Set *Placement & Angle per Seed*'s control to
   `fixed` and queue again — the generation is cached, so only the two filter nodes re-run.

**Several candidates per queue, each in a new place and at a new angle.** *Placement & Angle per
Seed* (group 05c) turns one queue into `candidates` (default 4) separate generations. Each gets its
own seed (base seed + 0, 1, 2, …), its own spot in the location photo (five slots from `x_min` to
`x_max`, nearer/bigger or further/smaller between `size_min` and `size_max`, with her eyes on the
`horizon` line) and its own body angle (front, three-quarter, profile or back-over-the-shoulder, to
either side). Within one queue no two candidates share a slot (up to 5) or an angle (up to 8). The
angle is drawn into the skeleton Krea 2 is conditioned on *and* written into the prompt, and Auto
Mask outputs the skeleton wherever it actually put the mask, so the mask, the skeleton and the words
always agree. Everything is a function of the seed, so a saved PNG dragged back onto the canvas
rebuilds the same set.

- `pose` = *new angle per seed* ignores the pose photo; *reference pose* keeps its skeleton and
  only moves, rescales and mirrors it (a 2D skeleton can't be turned to a new angle honestly).
- `vary` = *off* is the old behaviour: same place and pose, only the seed changes.
- `angles` is the list candidates draw from, one per line as `degrees: words for the prompt`
  (0 = facing the camera, positive = turned toward frame-right, 180 = back to the camera). Delete
  the ones you don't want, add your own.
- A **painted mask** still wins: every candidate stands where you painted and only the angle
  changes. A person already in the plate is replaced in place, the same way.

The candidates of one queue are saved together in `outputs/images/candidates/9b_<seed>/`, each with
its label (seed, place, angle) in `logs/generations.jsonl`. Through the Pod queue with batch count
10, that's 10 folders. Pick the winners with **Workflow 13**. Each candidate is a full generation:
4 candidates take 4x the GPU time.

Opening it on this machine is fine for reading the graph, but queuing it is not: Krea 2 needs an
RTX 5090 / RTX 6000 Pro class GPU, and this box's GTX 1060 isn't supported by the installed torch
build at all. Run it on the vast.ai pod (see
[Running this setup on a rented GPU](#running-this-setup-on-a-rented-gpu-vastai--runpod)).

### 13 — Pick the winner (`13_pick_winner.json`)

One node, *Pick Winner*: choose a candidate folder (newest first; press R to refresh the list), queue
with `winner` 0 to see its candidates in order, then set `winner` and queue again. That file is
**copied**, not re-saved, to `outputs/images/final/9b_<seed>_winner<N>.png`, so its pixels and
embedded workflow stay the same. It works the same for folders a pod run downloaded. It needs no
GPU or models, and the Pod queue's "Hold for the pod" lets it run on this PC.

### 11 — MiniMax H3 Image-to-Video (`11_minimax_h3_image_to_video.json`)

The open-weights alternative to Workflow 04: takes one generated still and animates it into a ~5 s
clip **with native audio** — MiniMax H3 generates speech, SFX and room tone in the same forward
pass as the picture, so the clip arrives already sounded and lip-synced. No API cost, nothing
leaves the machine.

It is a heavy model (the H3 DiT plus a Qwen3-VL-32B text encoder), so it runs on a **vast.ai pod**,
never on the local GTX 1060:

```bash
python custom_nodes/ai_influencer_toolkit/tools/comfy_pod.py launch --with-minimax --hours 2
python custom_nodes/ai_influencer_toolkit/tools/comfy_pod.py url
python custom_nodes/ai_influencer_toolkit/tools/comfy_pod.py watch --idle-minutes 25
```

`--with-minimax` makes the GPU search consider only Blackwell cards with 32 GB+ (H3's text encoder
ships in nvfp4, which needs compute capability ≥ 10; it still *runs* on a 3090/4090 pinned with
`--gpu`, dequantising every layer, several times slower), asks for a 200 GB disk, and downloads the five H3 files during bootstrap. Clips stay on the pod —
pull them out of `output/projects/<project>/outputs/videos/` before `destroy`.

**Uncensored by construction.** Nothing in this workflow filters anything: every node is local
ComfyUI core plus this toolkit, so unlike Workflow 04 (Seedance) there is no partner API node, no
provider moderation endpoint scoring the prompt or the frames, and no account to get flagged. The
pod only *downloads* models; outputs are written to its disk and come straight to you (the
*training* pod is the one that pushes files through a private HF repo). The one limit left is the
weights themselves — H3's own training makes it shy at explicit content, which is not a switch but
a LoRA problem, so the loader column has a second, bypassed **Optional extra LoRA** slot: point it
at a community fine-tune or at your character LoRA, Ctrl+B to enable, strength ~0.6–1.0.

**The three controls that matter**

- **Load Image** — your approved still. It is scaled to ~0.98 MP in multiples of 32 (H3's canvas
  cap) and its real size is read back into the model, so the clip keeps the still's aspect ratio.
- **Video description** — the big text box wired into the model's `prompt`, i.e. *what happens in
  the clip*. H3 wants one prose block, not tags. The shape that works, and the default shipped in
  the workflow:
  1. look + subject continuity ("photoreal, same face/hair/outfit as the starting image, background
     stays far out of focus") — repeating identity and the background blur here is what stops them
     drifting mid-clip;
  2. a timestamped beat list (`[0.0s-1.2s] she blinks and takes a breath…`) covering what the body
     does and what the camera does;
  3. dialogue in quotes — H3 speaks and lip-syncs it; one or two short lines per 5 s;
  4. an `Audio:` line (voice character, then background; say "no music" if you don't want any —
     silence is not the default);
  5. a negative tail ("no cuts, no scene changes, no text overlays, no extra fingers").
- **Duration (seconds)** — a Math Expression snaps it to H3's 24 fps / 17k+5 frame grid (5 s → 124
  frames). Trained range is roughly 5–15 s.

Optionally connect a second Load Image to the node's `last_frame` input to pin the closing frame
(useful for loops).

**Speed vs. quality.** The chain ships with the 8-step Lightning turbo LoRA on and `steps = 8`.
For maximum quality select the **turbo LoRA** node, press **Ctrl+B** to bypass it, and raise
`steps` on the Basic Scheduler to 20.

**Models** (in-canvas note has the links; all from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) except the LoRA):

```
models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
models/vae/minimax_h3_video_vae_fp16.safetensors
models/vae/minimax_h3_audio_vae_fp32.safetensors
models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
```

Output goes to `output/projects/<project>/outputs/videos/` and is logged to
`logs/generations.jsonl` with provider `minimax_h3_local`, like every other stage.

### 12 — Camera pass on an existing image (`12_camera_pass.json`)

The Camera Imperfections node on its own, for images generated before it was wired in or made
somewhere else. See [Camera imperfections](#6-camera-imperfections-always-on);
`tools/camera_pass.py` does the same thing to a whole folder from the command line.

## Pose / composition / reference images

Loaded via standard `LoadImage` nodes, which browse `input/` (and its subfolders, including
`input/ai_influencer/<project>/poses/` and `.../scenes/` created for you). Upload through the
node's own upload button, or copy files into that folder between runs. There is no dedicated
ControlNet-style pose-transfer input in Krea 2 / GPT Image 2.5 / Seedance 2.5's official APIs —
Krea's `style_reference` and GPT Image's multi-image edit both *influence* composition/pose but
aren't a hard pose lock; if exact pose matching turns out to matter more than style influence for
your use case, that's a real capability gap in the current provider APIs, not something this
toolkit works around.

## Cost control

Budget assumption from the brief: ~500 Krea images/month, ~100 GPT Image edits/month, low
Seedance volume. Design choices that keep it that way:

- Character creation and Seedance video are **separate workflows** you open on purpose (01/04), and
  **off by default** inside the combined workflow (05) — never triggered automatically by a Krea
  generation.
- Every partner node shows ComfyUI's own live price badge before you queue.
- `AIInfluencerSaveImage` / `AIInfluencerSaveVideo` append one line per output to
  `logs/generations.jsonl`:

  ```json
  {"timestamp": "...", "provider": "krea", "model": "Krea 2 Large", "operation": "generate",
   "project": "jane_doe", "stage": "krea_candidate", "resolution_or_aspect": "4:5",
   "quality": null, "num_outputs": 1, "seed": 12345, "job_id": null,
   "estimated_cost_usd": 0.06, "actual_cost_usd": null, "prompt_id": "a1b2c3d4e5",
   "output_path": "/…/output/projects/jane_doe/outputs/images/candidates/…png", "notes": null}
  ```

  `estimated_cost_usd` is computed locally (`custom_nodes/ai_influencer_toolkit/pricing.py`) only
  for Krea and GPT Image 2.5, whose official price-badge formulas are simple, stable lookup
  tables mirrored from `comfy_api_nodes/nodes_krea.py` / `nodes_openai.py` — it's a snapshot, not
  invented, and it can drift if ComfyUI's rates change (re-check against those files' `price_badge`
  if numbers look stale). Seedance/OpenRouter pricing depends on too many runtime factors to
  mirror safely, so those log `estimated_cost_usd: null` — use the in-canvas price badge for
  those. `actual_cost_usd` is always `null`: ComfyUI's backend does receive the real billed amount
  per call (`X-Comfy-Credits-Used` response header, see `comfy_api_nodes/util/client.py`), but it
  isn't exposed to node code, so this toolkit doesn't invent a number for it — check your real
  balance at <https://platform.comfy.org>.
- Exposed per generation: model, quality, resolution/aspect, number of outputs, provider,
  operation, seed, an optional job ID field, and a `prompt_id` (a short hash of the prompt text,
  not the full text) for grouping without bloating the log.

## Testing performed

1. Started ComfyUI (`--cpu`, isolated `--database-url` so it didn't collide with your
   already-running instance on 8188) and confirmed `ai_influencer_toolkit` imports with no
   warnings or errors (`Import times for custom nodes` log line, 0.0s, no exception) — both with
   and without `fal-client` installed (the LoRA nodes degrade gracefully if it's missing).
2. Verified all six custom nodes register correctly via `/object_info`, and cross-checked every
   node's actual `required`/`optional` field order there against what each workflow's
   `widgets_values` array assumes — this caught one real ordering bug (Train Character LoRA's
   `steps`/`trigger_phrase` were swapped) before it could reach you; fixed and re-verified.
3. **Executed every custom node for real** through `/prompt`: `LoadImage → AIInfluencerSaveImage`,
   `AIInfluencerLoadCharacter → AIInfluencerPromptBuilder → PreviewAny`,
   `LoadImage → CreateVideo → AIInfluencerSaveVideo`, `LoadImage → BatchImagesNode →
   AIInfluencerTrainCharacterLoRA`, and `AIInfluencerFalLoraImage → PreviewImage`. Confirmed
   correct file placement under `output/projects/<project>/...`, correct `character.json`
   auto-creation/update, correct `logs/generations.jsonl` entries, and that the two fal.ai nodes
   fail with a clear, specific error ("FAL_KEY is not set...") rather than crashing when no key is
   configured — then deleted all smoke-test projects/log lines.
4. Verified `.env` credential detection reads only variable *names*, never values (no secrets were
   printed at any point).
5. **Did not** run a real Krea/GPT-Image/Seedance/OpenRouter/fal.ai API call: no `COMFY_API_KEY`
   or `FAL_KEY` is configured in this environment and no Comfy account is signed in, so there was
   nothing to authenticate a paid test with. Per your own instruction to do at most one inexpensive
   test per provider *if credentials are available* — they weren't, so none were run. Do this
   yourself once credentials are in place; each node's price badge (or, for fal.ai, the numbers in
   this README) shows the cost before you confirm.
6. For all six workflow JSON files (graph/canvas format, not the flat API-execution format — see
   below): every node type was confirmed to exist in a live `/object_info`, and every link was
   checked for valid node references, in-range slot indices, and matching types (script-based
   structural validation, all passing). The DynamicCombo `widgets_values` arrays for the partner
   nodes (Krea 2, GPT Image 2.5, Seedance 2.5, OpenRouter) were built by adapting verified,
   real examples from the `comfyui-workflow-templates-json` package (installed alongside ComfyUI)
   rather than guessed, since that serialization isn't documented.
7. **Not verified**: actually opening these six files in a browser and clicking through them —
   this environment has no browser automation available. Do this once yourself before relying on
   them for real work; if something looks visually off (a socket in the wrong place, a
   mis-sized node), it's almost certainly cosmetic (position/size), not a wiring error, since
   wiring was checked programmatically. The one behavior worth eyeballing specifically: Workflow
   05's GPT-refinement bypass-passthrough (see its in-canvas note) — multi-input bypass routing
   is a frontend behavior this environment couldn't exercise live.

8. **Local Krea 2 additions (07-09)**, tested on the CPU instance:
   - `AIInfluencerLoraCheckpointTester` and `AIInfluencerApplyCharacterLora` register and show
     the expected inputs/outputs in `/object_info`.
   - The tester was run end to end with a stand-in for the model, sampler and VAE (no Krea 2
     weights on this machine, and the GTX 1060 can't run it): 6 fake checkpoint files, including
     a final one duplicating step 3000, `last_n=4`, a baseline column, two strengths and two
     prompts with a `#` comment line. Results: correct checkpoint selection and order
     (2250/2500/2750/3000, numbered file kept over the duplicate final), 18 renders, the grid,
     `report.json`, CLIP-style scores from the dataset folder, the character.json update and the
     log line. *Apply Character LoRA* then loaded the recorded checkpoint. All test files were
     deleted afterwards.
   - `check_dataset.py` against a synthetic dataset flagged the missing caption, empty caption,
     exact duplicate, orphan caption, low resolution, missing trigger and count warning.
     `make_ai_toolkit_config.py` produced the expected config and refused `--max-saves 4`.
     `pick_checkpoint.py` listed checkpoints and recorded `--step` and `--final` choices.
   - The generated 07-09 JSON was checked against the live `/object_info`: every node type exists,
     every required socket is connected, link types match, and the widget-value counts match
     (apart from the dynamic-dropdown and image-upload entries, which the official templates
     store the same way).
   - **Background blur** (`AIInfluencerBlurBackground`): registers in `/object_info`, and was run
     both directly and as a real queued prompt on the CPU instance (`LoadImage → Blur Background →
     PreviewImage`, executed without errors). On a synthetic image (striped background, flat disc
     as the "subject"): the subject came out bit-identical inside the mask, background contrast
     fell monotonically with `blur_amount` (0.8 → 1.5 → 3 → 6), no subject color bled into the
     background, `blur_amount=0` returned the input unchanged, and batches of 2 plus
     `subject_edge_shift` worked. Not tested with a real BiRefNet mask (the model isn't downloaded
     here), so check the mask edge on your first real run.
   - **Workflow 10 (preview → 4K)**: validated against the live schemas like the others, and
     checked that the confirm stage depends on nothing in the preview stage except the optional
     prompt text used for the log line — so muting the preview group can't break it. The muted
     (confirm) nodes are saved with mode 2, i.e. they stay off until you enable them.
   - **Not verified:** a real Krea 2 render, real LoRA loading/quality, the SeedVR2 chain on real
     weights, and opening 07-09 in a browser. Do one run of each on the RunPod ComfyUI before
     relying on them.
9. **Camera imperfections**: `AIInfluencerCameraImperfections` registers in `/object_info` with
   the expected widget order, and was run as a real queued prompt on a CPU instance
   (`LoadImage → Camera Imperfections → Prompt Builder → Save & Log Image`, executed without
   errors, correct file and log line, then deleted). Directly on tensors: the same seed gives a
   bit-identical result and a different seed doesn't, all-zero settings return the input
   unchanged, the vignette darkens corners and not the center, and a 3600x2700 image takes ~2 s
   on CPU. Checked by eye on a real portrait at 1:1 that `iso_grain` 0.025 reads as a phone photo
   rather than as noise. `tools/camera_pass.py` was run over a folder and wrote the graded copies.
   The Prompt Builder's mandatory block was confirmed present in its output with every field left
   blank.

## Troubleshooting

- **Checkpoint Tester: "No LoRA in models/loras matches"**: the files aren't in `models/loras/`
  (or a subfolder), or `lora_filter` doesn't match part of their path. Refresh ComfyUI (press R)
  after copying files in.
- **Apply Character LoRA: "has no local_lora yet"**: run Workflow 07 and record a checkpoint with
  `tools/pick_checkpoint.py`, or pick a file in the node's `lora_name`.
- **LoRA has no visible effect**: the prompt doesn't contain the trigger word (check `trigger`
  in character.json), or the LoRA was trained on a different architecture than Krea 2.
- **Faces look copied from the dataset / prompts ignored**: overtrained. Pick an earlier
  checkpoint in Workflow 07, or lower `local_lora_strength` (e.g. 0.8).
- **Out of memory in Workflow 08**: mute group 05 (SeedVR2) or lower its upscale factor.
- **Background blur cuts into hair, or leaves a sharp halo**: raise `feather`, or use
  `subject_edge_shift` (negative blurs slightly into the outline, positive keeps a sharp rim).
  A very cluttered background can also confuse the subject mask — check the node's `mask_used`
  output.
- **Background isn't blurred at all**: `blur_amount` is 0, the group is muted, or
  `birefnet.safetensors` is missing from `models/background_removal/`.

- **"Node not found" / missing-node errors on Open**: your ComfyUI version predates these nodes.
  Update ComfyUI (`git pull` — these are core, not a custom node install) and check
  `comfy_api_nodes/nodes_krea.py` / `nodes_openai.py` / `nodes_bytedance.py` / `nodes_openrouter.py`
  exist.
- **Partner node fails with an auth/401-style error**: no Comfy account is signed in / no
  `COMFY_API_KEY` for headless use. See [Credentials](#credentials).
- **GPT Image edit ignores your canonical reference**: make sure you actually wired an image into
  `model.images.image_2` (or a further slot) — GPT Image only uses images you connect, and the
  three-image socket count in the shipped workflow is a starting point, not a hard cap (up to 16
  are supported).
- **Video toggle in Workflow 05 does nothing when you flip only one node's Mode**: you must set
  **both** the Seedance node and its Save & Log Video node to Always — see that workflow's note.
- **Where did my project go?** `output/projects/<name>/` and `input/ai_influencer/<name>/` —
  everything is namespaced by the `project` field you type into Load Character / the Save nodes
  (or `AI_INFLUENCER_DEFAULT_PROJECT` in `.env` if you leave it blank).
- **"No module named 'fal_client'"**: run
  `pip install -r custom_nodes/ai_influencer_toolkit/requirements.txt` and restart ComfyUI — this
  only affects the two Workflow 06 LoRA nodes, everything else works without it.
- **Train Character LoRA fails with "No trigger phrase"**: set one on the node, or put a
  `trigger` in `character.json` first (`tools/new_project.py --trigger "..."`).

## Updating ComfyUI without breaking this setup

Everything added lives in new files that upstream ComfyUI doesn't have:

- `custom_nodes/ai_influencer_toolkit/` and `user/default/workflows/ai_influencer/` — tracked in this
  repo through `!` exceptions in `.gitignore` (upstream ignores `custom_nodes/` and `user/`); the
  rest of those folders, including your other nodes and settings, stays ignored.
- `output/projects/`, `input/ai_influencer/`, `logs/`, `.env` — ignored, never committed.
- `models/loras/<project>/` — your checkpoints; `models/` content is gitignored as well.
- `README_AI_INFLUENCER.md`, `.env.example`, `.githooks/` — plain new files at the repo root; a
  ComfyUI update won't overwrite them, and Git will only conflict if you also edit ComfyUI's own
  files.

Pull upstream changes with `git pull https://github.com/comfyanonymous/ComfyUI master`.
`git pull` (or however you normally update ComfyUI) is therefore safe. The one thing to re-check
after a major ComfyUI version bump: whether Krea/GPT Image/Seedance/OpenRouter's model lists or
pricing changed (`comfy_api_nodes/nodes_*.py`) — update `pricing.py`'s tables if the numbers there
look stale, and re-open the workflows to make sure their `model` dropdowns still resolve to
an entry that exists (a removed/renamed model would surface as a normal ComfyUI combo-value
warning on load, not a crash). `fal-client` (Workflow 06 only) is a normal pip package pinned by
nothing but its name in `custom_nodes/ai_influencer_toolkit/requirements.txt` — `pip install -U
fal-client` if fal.ai's API changes underneath you.

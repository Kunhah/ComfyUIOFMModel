"""Local (open-weights) Krea 2 character-LoRA nodes.

The training side runs outside ComfyUI (Ostris AI Toolkit, usually on a rented RunPod GPU, see
tools/make_ai_toolkit_config.py). It saves one checkpoint every N steps, and the final checkpoint
is not always the best one: past a certain step count the LoRA overfits (the face stays the same,
but poses, clothes and backgrounds start copying the training photos). These nodes help you
pick the best checkpoint:

AIInfluencerLoraCheckpointTester renders the same prompts with the same seed once per checkpoint
(and optionally a no-LoRA baseline) and puts them side by side in one labelled grid:
columns = checkpoints (by training step), rows = prompts. If you also connect CLIP_VISION, it
scores each checkpoint against your dataset images (input/ai_influencer/<project>/dataset/ by
default):
  - identity: mean cosine similarity of each render to the dataset centroid (higher = more
    like the character),
  - copy: mean of each render's similarity to its single closest dataset image (very high =
    memorizing training photos, a sign of overtraining).
CLIP vision is a general image embedding, not a face-recognition model, so treat the scores as a
tie-breaker next to your own eyes, not as ground truth. The grid, per-cell scores and settings go
to output/projects/<project>/lora_tests/<timestamp>/.

AIInfluencerApplyCharacterLora applies the LoRA recorded in character.json ("local_lora" /
"local_lora_strength") to a MODEL, so production workflows follow whatever checkpoint you picked
without having to re-select it in every workflow.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import nodes
from comfy_api.latest import IO, ui

from . import logging_util

FROM_CHARACTER = "(from character.json)"
_STEP_RE = re.compile(r"_(\d{3,})$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _checkpoint_step(filename: str) -> int | None:
    """AI Toolkit names intermediate saves <name>_000002500.safetensors and the final one
    <name>.safetensors (no step suffix)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = _STEP_RE.search(stem)
    return int(m.group(1)) if m else None


def find_checkpoints(lora_filter: str, last_n: int, final_step: int) -> list[tuple[str, int]]:
    """Returns [(lora filename, step)] sorted by step. The final (unsuffixed) checkpoint gets
    final_step, or sorts after every numbered checkpoint if final_step is 0."""
    needle = lora_filter.strip().lower()
    if not needle:
        raise ValueError("lora_filter is empty: type part of the LoRA filename, e.g. 'l1na'.")
    found = []
    for name in folder_paths.get_filename_list("loras"):
        if needle not in name.replace("\\", "/").lower():
            continue
        step = _checkpoint_step(name)
        if step is None:
            step = final_step if final_step > 0 else 10**12
        found.append((name, step))
    # A final checkpoint with final_step set can duplicate the last numbered save; keep the numbered one.
    by_step: dict[int, str] = {}
    for name, step in sorted(found, key=lambda x: (x[1], _checkpoint_step(x[0]) is None)):
        by_step.setdefault(step, name)
    result = sorted(((n, s) for s, n in by_step.items()), key=lambda x: x[1])
    if not result:
        raise ValueError(f"No LoRA in models/loras matches '{lora_filter}'.")
    if last_n > 0:
        result = result[-last_n:]
    return result


def _step_label(step: int) -> str:
    return "final" if step >= 10**12 else f"{step} steps"


def _parse_prompts(prompts: str, trigger: str) -> list[str]:
    lines = [l.strip() for l in prompts.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        raise ValueError("prompts is empty: write one test prompt per line.")
    return [l.replace("[trigger]", trigger) for l in lines]


def _parse_strengths(strengths: str) -> list[float]:
    try:
        values = [float(s) for s in strengths.replace(";", ",").split(",") if s.strip()]
    except ValueError as e:
        raise ValueError(f"lora_strengths must be comma-separated numbers, got '{strengths}'.") from e
    return values or [1.0]


def _embed(clip_vision, images: torch.Tensor) -> torch.Tensor:
    out = []
    for i in range(images.shape[0]):
        e = clip_vision.encode_image(images[i:i + 1]).image_embeds
        out.append(torch.nn.functional.normalize(e.float().cpu(), dim=-1))
    return torch.cat(out, dim=0)


def _reference_dir(project: str, reference_folder: str) -> str:
    input_dir = os.path.abspath(folder_paths.get_input_directory())
    rel = reference_folder.strip() or os.path.join("ai_influencer", project, "dataset")
    path = os.path.abspath(os.path.join(input_dir, rel))
    if os.path.commonpath([input_dir, path]) != input_dir:
        raise ValueError(f"reference_folder must be inside ComfyUI's input folder, got '{reference_folder}'.")
    return path


def _embed_folder(clip_vision, folder: str) -> torch.Tensor:
    names = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in _IMAGE_EXTS) if os.path.isdir(folder) else []
    if not names:
        raise ValueError(f"Scoring needs your dataset images in {folder} (or disconnect clip_vision to skip scoring).")
    embeds = []
    for name in names:
        with Image.open(os.path.join(folder, name)) as img:
            arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        embeds.append(_embed(clip_vision, torch.from_numpy(arr).unsqueeze(0)))
    return torch.cat(embeds, dim=0)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def build_grid(cells: list[list[torch.Tensor]], col_labels: list[str], row_labels: list[str]) -> torch.Tensor:
    """cells[row][col] is an HxWxC tensor in 0..1. Returns a 1xHxWxC grid with a label header
    per column and a prompt number per row."""
    h, w = cells[0][0].shape[0], cells[0][0].shape[1]
    font_size = max(14, w // 28)
    header = font_size * 3 + 8
    side = font_size * 3
    rows, cols = len(cells), len(cells[0])
    canvas = Image.new("RGB", (side + cols * w, header + rows * h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = _font(font_size)
    for c, label in enumerate(col_labels):
        for li, line in enumerate(label.split("\n")[:3]):
            draw.text((side + c * w + 8, 4 + li * font_size), line, fill=(235, 235, 235), font=font)
    for r in range(rows):
        draw.text((6, header + r * h + 6), row_labels[r], fill=(235, 235, 235), font=font)
        for c in range(cols):
            arr = (cells[r][c].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            tile = Image.fromarray(arr)
            if tile.size != (w, h):
                tile = tile.resize((w, h))
            canvas.paste(tile, (side + c * w, header + r * h))
    return torch.from_numpy(np.asarray(canvas).astype(np.float32) / 255.0).unsqueeze(0)


class AIInfluencerLoraCheckpointTester(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerLoraCheckpointTester",
            display_name="LoRA Checkpoint Tester (AI Influencer)",
            category="ai_influencer",
            description=(
                "Renders the same prompts with the same seed once per LoRA checkpoint and shows "
                "them side by side in one grid (columns = training steps, rows = prompts), so you "
                "can pick the best checkpoint instead of trusting the last one. Optionally scores "
                "each checkpoint against your dataset with CLIP vision."
            ),
            inputs=[
                IO.Model.Input("model", tooltip="Base model to test on. For Krea 2, test on Turbo (what you will generate with), even though you trained on RAW."),
                IO.Clip.Input("clip"),
                IO.Vae.Input("vae"),
                IO.String.Input("project", default="", tooltip="Project/character name. Empty uses AI_INFLUENCER_DEFAULT_PROJECT or 'default'."),
                IO.String.Input("lora_filter", default="", tooltip="Part of the checkpoint filenames in models/loras, e.g. 'l1na' or 'lina/'. Every match is a checkpoint."),
                IO.Int.Input("last_n", default=4, min=0, max=100, tooltip="Test only the last N checkpoints by step. 0 = all of them."),
                IO.Int.Input("final_step", default=0, min=0, max=1000000, tooltip="Total training steps, used to label the final checkpoint (saved without a step number). 0 = label it 'final'."),
                IO.Boolean.Input("include_baseline", default=False, tooltip="Add a column rendered without any LoRA, for comparison."),
                IO.String.Input("trigger", default="", tooltip="Replaces [trigger] in the prompts. Empty uses character.json's trigger."),
                IO.String.Input(
                    "prompts",
                    multiline=True,
                    default=(
                        "photo of [trigger], close-up portrait, looking at the camera, soft window light\n"
                        "photo of [trigger] laughing at a cafe table, candid smartphone photo\n"
                        "full body photo of [trigger] walking down a city street at night, wearing a red coat\n"
                        "photo of [trigger] in profile, hiking on a mountain trail, overcast daylight"
                    ),
                    tooltip="One test prompt per line (one grid row each). Lines starting with # are skipped. Mix close-ups with scenes and outfits that are NOT in your dataset to catch overfitting.",
                ),
                IO.String.Input("lora_strengths", default="1.0", tooltip="Comma-separated, e.g. '0.8, 1.0'. Each strength adds a column per checkpoint."),
                IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=False, tooltip="Kept fixed across all cells so only the checkpoint changes."),
                IO.Int.Input("steps", default=8, min=1, max=150, tooltip="Krea 2 Turbo: 8."),
                IO.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1, tooltip="Krea 2 Turbo: 1.0."),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                IO.Int.Input("width", default=832, min=256, max=4096, step=16),
                IO.Int.Input("height", default=1024, min=256, max=4096, step=16),
                IO.ClipVision.Input("clip_vision", optional=True, tooltip="Optional: enables the identity/copy scores."),
                IO.String.Input("reference_folder", default="", optional=True, tooltip="Folder under input/ with your dataset images, for scoring. Empty = ai_influencer/<project>/dataset."),
                IO.Boolean.Input("save_best_to_character", default=False, optional=True, tooltip="Write the best-scoring checkpoint into character.json (needs scoring). Leave off if you'd rather pick by eye with tools/pick_checkpoint.py."),
            ],
            outputs=[
                IO.Image.Output(display_name="grid"),
                IO.Image.Output(display_name="images"),
                IO.String.Output(display_name="report"),
                IO.String.Output(display_name="best_lora"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        vae,
        project: str,
        lora_filter: str,
        last_n: int,
        final_step: int,
        include_baseline: bool,
        trigger: str,
        prompts: str,
        lora_strengths: str,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        width: int,
        height: int,
        clip_vision=None,
        reference_folder: str = "",
        save_best_to_character: bool = False,
    ) -> IO.NodeOutput:
        project = logging_util.sanitize_project_name(project)
        logging_util.ensure_project(project)
        char_path = logging_util.character_json_path(project)
        with open(char_path, encoding="utf-8") as f:
            character = json.load(f)
        trigger = trigger.strip() or character.get("trigger", "")
        prompt_list = _parse_prompts(prompts, trigger)
        strengths = _parse_strengths(lora_strengths)
        checkpoints = find_checkpoints(lora_filter, last_n, final_step)

        columns: list[dict] = []
        if include_baseline:
            columns.append({"lora": None, "step": 0, "strength": 0.0, "label": "no LoRA"})
        for name, step in checkpoints:
            for s in strengths:
                label = _step_label(step) if len(strengths) == 1 else f"{_step_label(step)} @ {s:g}"
                columns.append({"lora": name, "step": step, "strength": s, "label": label})

        scoring = clip_vision is not None
        ref_embeds = _embed_folder(clip_vision, _reference_dir(project, reference_folder)) if scoring else None
        centroid = torch.nn.functional.normalize(ref_embeds.mean(0, keepdim=True), dim=-1) if scoring else None

        # The text encoder isn't touched by a model-only LoRA, so each prompt is encoded once.
        conds = []
        for text in prompt_list:
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(text))
            negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
            conds.append((positive, negative))
        latent = nodes.EmptyLatentImage().generate(width, height, 1)[0]

        pbar = comfy.utils.ProgressBar(len(columns) * len(prompt_list))
        cells: list[list[torch.Tensor]] = [[None] * len(columns) for _ in prompt_list]
        for c, col in enumerate(columns):
            patched = model
            if col["lora"] is not None:
                lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", col["lora"]), safe_load=True)
                patched, _ = comfy.sd.load_lora_for_models(model, None, lora, col["strength"], 0)
                del lora
            for r, (positive, negative) in enumerate(conds):
                comfy.model_management.throw_exception_if_processing_interrupted()
                samples = nodes.common_ksampler(patched, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent)[0]
                image = nodes.VAEDecode().decode(vae, samples)[0]
                cells[r][c] = image[0]
                pbar.update(1)

            if scoring:
                embeds = _embed(clip_vision, torch.stack([cells[r][c] for r in range(len(prompt_list))]))
                col["identity_per_prompt"] = [round(v, 4) for v in (embeds @ centroid.T).squeeze(1).tolist()]
                col["identity"] = round(sum(col["identity_per_prompt"]) / len(prompt_list), 4)
                col["copy"] = round((embeds @ ref_embeds.T).max(dim=1).values.mean().item(), 4)

        best = None
        if scoring:
            candidates = [c for c in columns if c["lora"] is not None]
            best = max(candidates, key=lambda c: c["identity"])
            for col in columns:
                col["label"] += f"\nid {col['identity']:.3f}  copy {col['copy']:.3f}"
                if col is best:
                    col["label"] += "\n* best identity"

        grid = build_grid(cells, [c["label"] for c in columns], [f"P{i + 1}" for i in range(len(prompt_list))])
        all_images = torch.stack([cells[r][c] for r in range(len(prompt_list)) for c in range(len(columns))])

        stamp = time.strftime("%Y%m%d_%H%M%S")
        test_dir = os.path.join(logging_util.get_project_output_dir(project), "lora_tests", stamp)
        os.makedirs(test_dir, exist_ok=True)
        Image.fromarray((grid[0].numpy() * 255).astype(np.uint8)).save(os.path.join(test_dir, "grid.png"))
        report = {
            "project": project,
            "timestamp": stamp,
            "settings": {
                "lora_filter": lora_filter, "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler_name, "scheduler": scheduler, "width": width, "height": height,
                "strengths": strengths, "scored": scoring,
            },
            "prompts": {f"P{i + 1}": p for i, p in enumerate(prompt_list)},
            "columns": [{k: v for k, v in c.items() if k != "label"} | {"label": c["label"].split("\n")[0]} for c in columns],
            "best_by_identity": best["lora"] if best else None,
            "note": (
                "identity = similarity to the dataset centroid (higher is more like the character). "
                "copy = similarity to the closest single dataset image; if it keeps rising while identity "
                "is flat, later checkpoints are memorizing the dataset. CLIP is not a face recognizer: "
                "confirm with your own eyes."
            ),
        }
        report_json = json.dumps(report, indent=2)
        with open(os.path.join(test_dir, "report.json"), "w", encoding="utf-8") as f:
            f.write(report_json)

        if best and save_best_to_character:
            character["local_lora"] = best["lora"]
            character["local_lora_strength"] = best["strength"]
            with open(char_path, "w", encoding="utf-8") as f:
                json.dump(character, f, indent=2)
            logging.info("AI Influencer: set %s local_lora=%s", project, best["lora"])

        logging_util.append_generation_log({
            "provider": "local", "model": "lora_checkpoint_test", "operation": "lora_test",
            "project": project, "stage": "lora_test", "resolution_or_aspect": f"{width}x{height}",
            "quality": None, "num_outputs": int(all_images.shape[0]), "seed": seed, "job_id": None,
            "estimated_cost_usd": None, "actual_cost_usd": None,
            "prompt_id": logging_util.make_prompt_id("\n".join(prompt_list)),
            "output_path": test_dir, "notes": f"best_by_identity={report['best_by_identity']}",
        })

        return IO.NodeOutput(
            grid, all_images, report_json, best["lora"] if best else "",
            ui=ui.PreviewImage(grid, cls=cls),
        )


class AIInfluencerApplyCharacterLora(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerApplyCharacterLora",
            display_name="Apply Character LoRA (AI Influencer)",
            category="ai_influencer",
            description=(
                "Applies the character's chosen LoRA checkpoint (character.json 'local_lora') to a "
                "model, and outputs the trigger word. Pick a specific file in lora_name to override."
            ),
            inputs=[
                IO.Model.Input("model"),
                IO.String.Input("project", default=""),
                IO.Combo.Input("lora_name", options=[FROM_CHARACTER] + folder_paths.get_filename_list("loras"), default=FROM_CHARACTER),
                IO.Float.Input("strength", default=-1.0, min=-1.0, max=4.0, step=0.05, tooltip="-1 = use character.json's local_lora_strength (or 1.0)."),
            ],
            outputs=[
                IO.Model.Output(),
                IO.String.Output(display_name="trigger"),
                IO.String.Output(display_name="lora_name"),
            ],
        )

    @classmethod
    def execute(cls, model, project: str, lora_name: str, strength: float) -> IO.NodeOutput:
        project = logging_util.sanitize_project_name(project)
        logging_util.ensure_project(project)
        with open(logging_util.character_json_path(project), encoding="utf-8") as f:
            character = json.load(f)
        if lora_name == FROM_CHARACTER:
            lora_name = character.get("local_lora", "")
            if not lora_name:
                raise ValueError(
                    f"character.json for '{project}' has no local_lora yet. Pick a checkpoint with the "
                    "LoRA Checkpoint Tester (Workflow 07) or tools/pick_checkpoint.py, or choose a file here."
                )
        if strength < 0:
            strength = float(character.get("local_lora_strength", 1.0))
        lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
        patched, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        return IO.NodeOutput(patched, character.get("trigger", ""), lora_name)

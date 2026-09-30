"""Character sheets for Workflow 01: the prompt that asks for one, and the slicer that cuts it into
one picture per view for LoRA training.

Prompts. Character Sheet Studio (character-sheet-studio/ at the repo root, not part of this pack)
ships four JSON templates -- face-turnaround, expression-grid, full-body-360, upper-body -- written
for a reference-image edit model: identity comes only from the attached photos, the template only
says which views to draw and locks background, lighting, framing and scale. Its rules are that the
template is sent unchanged and the subject is never described in words (extra words compete with
the photos and cause drift). This pack adds four more in the same format and under the same rules
in sheet_templates/ -- outfit-grid, lighting-grid, pose-grid, expression-grid-extra -- each
changing one thing (clothes, light, pose, expression) and keeping everything else locked.

Only the studio's templates are used. Its own wizard uploads to WaveSpeed and has the agent look at
every reference photo; here ComfyUI sends the photos to GPT Image 2.5 and nobody looks at them.

Slicing. A LoRA trained on whole sheets learns to draw sheets (grey boards, grids of little
people); it should see one figure per picture. Every template asks for the same layout -- one
large view in the left half, a rows x cols grid in the right half, a flat grey background and no
borders -- so the slicer:
  1. splits the canvas into those cells;
  2. marks as "figure" every pixel that differs clearly from its cell's background colour (the
     median of the cell's outer ring, so a tinted background in the lighting sheet still works);
  3. takes the connected figure shapes, gives each to the cell its centre falls in, and crops the
     union of a cell's shapes with some padding -- a hand reaching over a cell line stays whole;
  4. falls back to the plain cell rectangle when a cell has no shape of its own.
Each crop gets a caption: the trigger word plus that panel's own wording from the template (e.g.
"lit by a warm tungsten table lamp at frame-right"), so the LoRA learns the light, the outfit or the
pose as a variable, not as part of her. The labels assume GPT kept the template's reading order,
which it usually does -- look at the crops before they go into the dataset.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import torch

import folder_paths
from comfy_api.latest import IO, ui

from . import logging_util
from .sheet_slicing import (DEFAULT_AVOID, DEFAULT_DIR, DEFAULT_LOCKS, USER_DIR, build_template, list_sheets,
                            load_template, save_slices, slice_sheet)


def project_trigger(project: str) -> str:
    try:
        with open(logging_util.character_json_path(project), encoding="utf-8") as f:
            return json.load(f).get("trigger", "") or ""
    except (OSError, ValueError):
        return ""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip().lower()).strip("-") or "sheet"


class AIInfluencerSheetPrompt(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerSheetPrompt",
            display_name="Character Sheet Prompt (AI Influencer)",
            category="ai_influencer",
            search_aliases=["character sheet", "turnaround", "reference sheet", "model sheet"],
            description="Outputs one character sheet template, unchanged, as the prompt for a reference-image edit.",
            inputs=[
                IO.Combo.Input("sheet", options=list_sheets(), default="face-turnaround",
                               tooltip="Character Sheet Studio's four, this pack's (sheet_templates/), and yours "
                                       "(user/ai_influencer/sheet_templates/, e.g. saved by Custom Character Sheet). Press R to refresh."),
                IO.String.Input("templates_dir", default=DEFAULT_DIR,
                                tooltip="Folder with the studio's <sheet>.json files, relative to the ComfyUI folder or absolute."),
            ],
            outputs=[IO.String.Output(display_name="template", tooltip="The template JSON: the prompt for GPT, and the layout + captions for the slicer.")],
        )

    @classmethod
    def execute(cls, sheet: str, templates_dir: str) -> IO.NodeOutput:
        return IO.NodeOutput(json.dumps(load_template(sheet, templates_dir), ensure_ascii=False))


class AIInfluencerCustomSheet(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerCustomSheet",
            display_name="Custom Character Sheet (AI Influencer)",
            category="ai_influencer",
            search_aliases=["character sheet", "custom sheet", "my sheet"],
            description=(
                "Your own character sheet: one large view plus one panel per line, laid out and locked the way the "
                "studio's templates are. Describe what changes in each panel (outfit, pose, light...), never the person."
            ),
            inputs=[
                IO.String.Input("title", default="Outfit Variations"),
                IO.String.Input("hero", multiline=True,
                                default="seen straight from the front, standing, the whole person visible from the top of the head to both feet, "
                                        "wearing the same clothes as in the reference images, neutral expression",
                                tooltip="The large view in the left half."),
                IO.String.Input("panels", multiline=True,
                                default="seen straight from the front, standing, the whole person visible, wearing a red raincoat and rubber boots\n"
                                        "seen straight from the front, standing, the whole person visible, wearing a ski jacket and ski pants\n"
                                        "seen straight from the front, standing, the whole person visible, wearing a white linen shirt and beige shorts\n"
                                        "seen straight from the front, standing, the whole person visible, wearing a black leather jacket and dark jeans",
                                tooltip="One panel per line, in reading order. The grid (rows x cols) follows from how many there are."),
                IO.String.Input("locks", multiline=True, default="\n".join(DEFAULT_LOCKS),
                                tooltip="What stays the same in every panel, one per line. Remove a line to let it vary."),
                IO.String.Input("avoid", multiline=True, default="\n".join(DEFAULT_AVOID), tooltip="One per line."),
                IO.String.Input("save_as", default="",
                                tooltip="A name to keep this as a template: saved to user/ai_influencer/sheet_templates/<name>.json and "
                                        "listed in every Character Sheet Prompt (press R). Empty = don't save."),
            ],
            outputs=[IO.String.Output(display_name="template")],
        )

    @classmethod
    def execute(cls, title, hero, panels, locks, avoid, save_as) -> IO.NodeOutput:
        template = build_template(title, hero, panels.splitlines(), locks.splitlines(), avoid=avoid.splitlines())
        if save_as.strip():
            os.makedirs(USER_DIR, exist_ok=True)
            with open(os.path.join(USER_DIR, _slug(save_as) + ".json"), "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
        return IO.NodeOutput(json.dumps(template, ensure_ascii=False))


class AIInfluencerSliceSheet(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerSliceSheet",
            display_name="Slice Character Sheet (AI Influencer)",
            category="ai_influencer",
            search_aliases=["split sheet", "crop panels", "character sheet"],
            description=(
                "Cuts a character sheet into one image per view, each with a caption (trigger word + that panel's "
                "wording from the template), saved to outputs/images/sheet_crops/<sheet>_<time>/ for LoRA training."
            ),
            inputs=[
                IO.Image.Input("sheet_image"),
                IO.String.Input("template", force_input=True,
                                tooltip="The same template that made the sheet (from Character Sheet Prompt or Custom Character Sheet): "
                                        "it gives the layout and the captions."),
                IO.String.Input("project", default="", tooltip="For the folder and the trigger word in character.json. Empty = default project."),
                IO.String.Input("caption_prefix", default="", tooltip="Start of every caption. Empty = the trigger word from character.json (or [trigger])."),
                IO.String.Input("caption_suffix", default="plain grey studio background", tooltip="End of every caption. Empty = nothing."),
                IO.Float.Input("threshold", default=0.09, min=0.01, max=0.5, step=0.01,
                               tooltip="How different from the background a pixel must be to count as the person. Raise it if "
                                       "shadows or background gradients get included; lower it for pale clothes on grey."),
                IO.Float.Input("padding", default=0.05, min=0.0, max=0.5, step=0.01, tooltip="Extra room around each figure, fraction of its size."),
            ],
            outputs=[
                IO.Image.Output(display_name="crops", is_output_list=True),
                IO.String.Output(display_name="captions", is_output_list=True),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, sheet_image, template: str, project: str, caption_prefix: str, caption_suffix: str,
                threshold: float, padding: float) -> IO.NodeOutput:
        project = logging_util.sanitize_project_name(project)
        tpl = json.loads(template)
        name = _slug(tpl.get("task", "sheet").split(",")[0])
        trigger = project_trigger(project)
        crops, captions, shown = [], [], []
        base = os.path.join(logging_util.get_project_output_dir(project), "outputs", "images", "sheet_crops")
        for b in range(sheet_image.shape[0]):
            img = sheet_image[b].float().cpu().numpy()
            slices = slice_sheet(img, tpl, trigger, threshold, padding, caption_prefix, caption_suffix)
            folder = os.path.join(base, f"{name}_{time.strftime('%Y%m%d-%H%M%S')}" + (f"_{b}" if sheet_image.shape[0] > 1 else ""))
            names = save_slices(slices, folder)
            sub = os.path.relpath(folder, folder_paths.get_output_directory())
            shown += [ui.SavedResult(n, sub, IO.FolderType.output) for n in names]
            crops += [torch.from_numpy(np.ascontiguousarray(c))[None] for c, _, _ in slices]
            captions += [cap for _, cap, _ in slices]
        return IO.NodeOutput(crops, captions, ui=ui.SavedImages(shown))

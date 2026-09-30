"""Choosing one image out of several, without a "pause and ask" node (ComfyUI has none).

AIInfluencerPickCandidate  -- Workflow 01: one GPT Image 2.5 call returns up to 8 portraits; `pick`
                              says which one the rest of the workflow builds on. 0 = not chosen
                              yet: everything after it is skipped silently, so the first queue only
                              shows the candidates. Set a number and queue again -- the candidates
                              node's inputs didn't change, so ComfyUI reuses them instead of paying
                              for a new call.
AIInfluencerPickWinner     -- Workflow 13: shows every candidate a 09b queue saved into its own
                              folder, and copies the one you name into outputs/images/final/. Works
                              the same for jobs that ran on the pod (their folders are downloaded
                              into this PC's output/) -- a batch of 10 queues is 10 folders to pick
                              from, one at a time. Needs no GPU and no models.
"""
from __future__ import annotations

import os
import shutil
import time

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
from comfy_api.latest import IO, ui
from comfy_execution.graph_utils import ExecutionBlocker

from . import logging_util

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
NONE_YET = "(no candidate folders yet)"


class AIInfluencerPickCandidate(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerPickCandidate",
            display_name="Pick Candidate (AI Influencer)",
            category="ai_influencer",
            search_aliases=["choose", "select image", "winner"],
            description=(
                "Passes on one image of a batch. pick = 0 stops here silently (first queue: just look at the "
                "candidates); set the number of the one you want and queue again. An image on `own_image` "
                "always wins, for starting from a picture you already have."
            ),
            inputs=[
                IO.Image.Input("candidates", optional=True, tooltip="The batch to choose from."),
                IO.Int.Input("pick", default=0, min=0, max=16, tooltip="1 = first candidate. 0 = not chosen yet: nothing after this node runs."),
                IO.Image.Input("own_image", optional=True, tooltip="Use this image instead of a candidate."),
            ],
            outputs=[IO.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, pick: int, candidates=None, own_image=None) -> IO.NodeOutput:
        if own_image is not None:
            return IO.NodeOutput(own_image[:1])
        if candidates is None or pick == 0:
            return IO.NodeOutput(ExecutionBlocker(None))
        if pick > candidates.shape[0]:
            raise ValueError(f"pick is {pick}, but there are only {candidates.shape[0]} candidates")
        return IO.NodeOutput(candidates[pick - 1:pick])


def _candidates_root(project: str) -> str:
    return os.path.join(folder_paths.get_output_directory(), "projects", project,
                        logging_util.OUTPUT_STAGE_SUBDIR["local_candidate"])


def candidate_folders() -> list[str]:
    """'<project>/<folder>' for every candidate folder, newest first. Names only; nothing is opened."""
    base = os.path.join(folder_paths.get_output_directory(), "projects")
    found = []
    if os.path.isdir(base):
        for project in os.listdir(base):
            root = _candidates_root(project)
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                path = os.path.join(root, name)
                if os.path.isdir(path):
                    found.append((os.path.getmtime(path), f"{project}/{name}"))
    return [name for _, name in sorted(found, reverse=True)]


def _images_in(path: str) -> list[str]:
    return sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS))


class AIInfluencerPickWinner(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerPickWinner",
            display_name="Pick Winner (AI Influencer)",
            category="ai_influencer",
            search_aliases=["choose", "select", "winner", "candidates"],
            description=(
                "Shows the candidates one queue saved into its own folder; with winner > 0, copies that one "
                "(file untouched, workflow metadata included) into outputs/images/final/."
            ),
            inputs=[
                IO.Combo.Input("folder", options=candidate_folders() or [NONE_YET],
                               tooltip="One folder per queue, newest first. Press R on the canvas to refresh the list."),
                IO.Int.Input("winner", default=0, min=0, max=16, tooltip="0 = just show them. 1 = the first one, and so on."),
            ],
            outputs=[IO.Image.Output(display_name="winner")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, folder: str, winner: int):
        return time.time()  # the folder's contents can change between queues

    @classmethod
    def execute(cls, folder: str, winner: int) -> IO.NodeOutput:
        if folder == NONE_YET or "/" not in folder:
            raise ValueError("No candidate folders yet: queue Workflow 09b first (or press R to refresh the list).")
        project, name = folder.split("/", 1)
        project = logging_util.sanitize_project_name(project)
        root = _candidates_root(project)
        path = os.path.realpath(os.path.join(root, name))
        if os.path.dirname(path) != os.path.realpath(root) or not os.path.isdir(path):
            raise ValueError(f"No candidate folder '{folder}'")
        files = _images_in(path)
        if not files:
            raise ValueError(f"'{folder}' has no images")

        subfolder = os.path.relpath(path, folder_paths.get_output_directory())
        shown = [ui.SavedResult(f, subfolder, IO.FolderType.output) for f in files]
        if winner == 0:
            return IO.NodeOutput(ExecutionBlocker(None), ui=ui.SavedImages(shown))
        if winner > len(files):
            raise ValueError(f"winner is {winner}, but '{folder}' has {len(files)} candidates")

        src = os.path.join(path, files[winner - 1])
        final_dir = os.path.join(logging_util.get_project_output_dir(project), logging_util.OUTPUT_STAGE_SUBDIR["local_final"])
        dst = os.path.join(final_dir, f"{name}_winner{winner}{os.path.splitext(src)[1]}")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)  # a copy, not a re-save: pixels and embedded workflow stay exactly as they are
        with Image.open(dst) as im:
            arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float32) / 255.0
        final_sub = os.path.relpath(final_dir, folder_paths.get_output_directory())
        result = ui.SavedResult(os.path.basename(dst), final_sub, IO.FolderType.output)
        return IO.NodeOutput(torch.from_numpy(arr)[None], ui=ui.SavedImages([result]))

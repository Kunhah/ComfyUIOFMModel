"""Save+Log nodes: save generation output into the project's folder tree (see logging_util.py)
and append one line to logs/generations.jsonl, so every API call this toolkit makes has a
local, auditable record even though ComfyUI doesn't expose the actual billed credits to node
code (see pricing.py)."""

from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import IO, ComfyExtension, Input, Types, ui
from typing_extensions import override

from . import logging_util, pricing

_PROVIDERS = ["krea", "local_krea2", "openai_gpt_image", "bytedance_seedream", "bytedance_seedance", "openrouter", "other"]
_OPERATIONS = ["generate", "edit", "refine", "image_to_video", "other"]
_IMAGE_STAGES = ["character_candidate", "character_canonical", "krea_candidate", "gpt_refined", "local_candidate", "local_final", "misc_image"]
_VIDEO_STAGES = ["seedance_video", "minimax_h3_video", "misc_video"]


class AIInfluencerSaveImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerSaveImage",
            display_name="Save & Log Image (AI Influencer)",
            category="ai_influencer",
            description=(
                "Saves image(s) under output/projects/<project>/... and appends one record per "
                "output to logs/generations.jsonl (provider, model, operation, resolution, "
                "estimated cost where deterministically known, output path)."
            ),
            inputs=[
                IO.Image.Input("images", tooltip="Image(s) to save."),
                IO.String.Input("project", default="", tooltip="Project/character name. Empty uses AI_INFLUENCER_DEFAULT_PROJECT or 'default'."),
                IO.Combo.Input("stage", options=_IMAGE_STAGES, tooltip="Where in the project tree this belongs."),
                IO.Combo.Input("provider", options=_PROVIDERS),
                IO.String.Input("model", default="", tooltip="Exact model id/name used, e.g. 'Krea 2 Large' or 'gpt-image-2.5-flare'."),
                IO.Combo.Input("operation", options=_OPERATIONS),
                IO.String.Input("prompt_text", default="", multiline=True, optional=True, tooltip="The prompt sent, used only to derive a short prompt_id for the log (not stored in full)."),
                IO.String.Input("quality", default="", optional=True, tooltip="Provider quality tier, if applicable (e.g. 'high')."),
                IO.String.Input("size_or_aspect", default="", optional=True, tooltip="Size (e.g. '1024x1024') or aspect ratio (e.g. '4:5'), for the log and cost estimate."),
                IO.Int.Input("seed", default=-1, min=-1, max=2147483647, optional=True),
                IO.Int.Input("n_reference_images", default=0, min=0, max=20, optional=True, tooltip="Reference images sent with the request, if any (affects GPT Image cost estimate)."),
                IO.String.Input("job_id", default="", optional=True, tooltip="Provider job/request id if you have one on hand."),
                IO.String.Input("notes", default="", optional=True),
            ],
            outputs=[IO.Image.Output(), IO.String.Output(display_name="output_path")],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images: Input.Image,
        project: str,
        stage: str,
        provider: str,
        model: str,
        operation: str,
        prompt_text: str = "",
        quality: str = "",
        size_or_aspect: str = "",
        seed: int = -1,
        n_reference_images: int = 0,
        job_id: str = "",
        notes: str = "",
    ) -> IO.NodeOutput:
        project = logging_util.sanitize_project_name(project)
        basename = f"{stage}_{model.replace(' ', '_') or 'gen'}"
        filename_prefix = logging_util.output_filename_prefix(project, stage, basename)

        results: list[ui.SavedResult] = ui.ImageSaveHelper.save_images(
            images, filename_prefix=filename_prefix, folder_type=IO.FolderType.output, cls=cls
        )
        output_dir = folder_paths.get_output_directory()
        paths = [os.path.join(output_dir, r.subfolder, r.filename) for r in results]

        estimated_cost = pricing.estimate_cost_usd(
            provider,
            model,
            quality=quality,
            size=size_or_aspect,
            n=len(results),
            n_reference_images=n_reference_images,
        )

        for path, result in zip(paths, results):
            logging_util.append_generation_log(
                {
                    "provider": provider,
                    "model": model,
                    "operation": operation,
                    "project": project,
                    "stage": stage,
                    "resolution_or_aspect": size_or_aspect or None,
                    "quality": quality or None,
                    "num_outputs": len(results),
                    "seed": seed if seed >= 0 else None,
                    "job_id": job_id or None,
                    "estimated_cost_usd": estimated_cost,
                    "actual_cost_usd": None,
                    "prompt_id": logging_util.make_prompt_id(prompt_text) if prompt_text else None,
                    "output_path": path,
                    "notes": notes or None,
                }
            )

        return IO.NodeOutput(images, "; ".join(paths), ui=ui.SavedImages(results))


class AIInfluencerSaveVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerSaveVideo",
            display_name="Save & Log Video (AI Influencer)",
            category="ai_influencer",
            description=(
                "Saves a video under output/projects/<project>/outputs/videos/ and appends a "
                "record to logs/generations.jsonl."
            ),
            inputs=[
                IO.Video.Input("video"),
                IO.String.Input("project", default=""),
                IO.Combo.Input("stage", options=_VIDEO_STAGES),
                IO.Combo.Input("provider", options=["bytedance_seedance", "minimax_h3_local", "other"]),
                IO.String.Input("model", default="", tooltip="e.g. 'Seedance 2.5' or 'MiniMax H3'"),
                IO.String.Input("motion_prompt", default="", multiline=True, optional=True, tooltip="The motion prompt sent, used only to derive a short prompt_id for the log."),
                IO.String.Input("resolution", default="", optional=True, tooltip="e.g. '1080p'"),
                IO.String.Input("duration_seconds", default="", optional=True),
                IO.Int.Input("seed", default=-1, min=-1, max=2147483647, optional=True),
                IO.String.Input("job_id", default="", optional=True),
                IO.String.Input("notes", default="", optional=True),
            ],
            outputs=[IO.Video.Output(), IO.String.Output(display_name="output_path")],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        project: str,
        stage: str,
        provider: str,
        model: str,
        motion_prompt: str = "",
        resolution: str = "",
        duration_seconds: str = "",
        seed: int = -1,
        job_id: str = "",
        notes: str = "",
    ) -> IO.NodeOutput:
        project = logging_util.sanitize_project_name(project)
        basename = f"{stage}_{model.replace(' ', '_') or 'gen'}"
        filename_prefix = logging_util.output_filename_prefix(project, stage, basename)

        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height
        )
        file = f"{filename}_{counter:05}_.mp4"
        full_path = os.path.join(full_output_folder, file)
        video.save_to(full_path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)

        logging_util.append_generation_log(
            {
                "provider": provider,
                "model": model,
                "operation": "image_to_video",
                "project": project,
                "stage": stage,
                "resolution_or_aspect": resolution or None,
                "duration_seconds": duration_seconds or None,
                "num_outputs": 1,
                "seed": seed if seed >= 0 else None,
                "job_id": job_id or None,
                "estimated_cost_usd": None,
                "actual_cost_usd": None,
                "prompt_id": logging_util.make_prompt_id(motion_prompt) if motion_prompt else None,
                "output_path": full_path,
                "notes": notes or None,
            }
        )

        return IO.NodeOutput(video, full_path)


class AIInfluencerToolkitSaveExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [AIInfluencerSaveImage, AIInfluencerSaveVideo]

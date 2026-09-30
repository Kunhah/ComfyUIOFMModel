"""Optional real character-LoRA training/inference via fal.ai (a different provider/account from
Krea/GPT Image/Seedance, which all bill through ComfyUI's own Comfy-account proxy and have no
LoRA mechanism at all -- see README_AI_INFLUENCER.md, "Where's the LoRA?").

Uses fal.ai's own official `fal-client` SDK rather than hand-rolled HTTP calls: fal's upload
endpoint isn't documented precisely enough to reimplement safely, and this is fal's own
maintained client, not a third-party dependency.

Verified against fal.ai's own docs (Sept 2026):
  - training: fal-ai/flux-lora-portrait-trainer -- $0.0024/step, 1000-step minimum billed.
  - inference: fal-ai/flux-lora -- $0.035/megapixel.
Both numbers are provider-quoted, not invented; see pricing.py for how they're used in the log.
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile

import numpy as np
import torch
from comfy_api.latest import IO, ComfyExtension, Input
from PIL import Image
from typing_extensions import override

from . import env_config, logging_util, pricing

logger = logging.getLogger(__name__)

TRAIN_ENDPOINT = "fal-ai/flux-lora-portrait-trainer"
INFER_ENDPOINTS = {
    "krea-2-turbo": "fal-ai/krea-2/turbo/lora",  # $0.01/MP; LoRAs from train_lora.py (fal-ai/krea-2-trainer)
    "flux-1-dev": "fal-ai/flux-lora",  # $0.035/MP; LoRAs from the Train Character LoRA node above
}

_IMAGE_SIZES = ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"]


def _require_fal_key() -> None:
    # fal_client reads FAL_KEY from os.environ itself; env_config puts .env's value there.
    env_config.require("FAL_KEY")


def _tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    arr = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _images_to_zip_bytes(images: Input.Image) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, frame in enumerate(images):
            zf.writestr(f"image_{i:03d}.png", _tensor_to_png_bytes(frame))
    return buf.getvalue()


def _url_to_image_tensor(data: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class AIInfluencerTrainCharacterLoRA(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerTrainCharacterLoRA",
            display_name="Train Character LoRA (fal.ai)",
            category="ai_influencer",
            description=(
                "Trains a real FLUX LoRA on your canonical reference photos via fal.ai's "
                "flux-lora-portrait-trainer. Costs ~$0.0024/step (min 1000 steps, ~$2.40; "
                "default 2500 steps, ~$6) billed to your fal.ai account, separate from your "
                "Comfy account. Takes several minutes. Saves the result into character.json."
            ),
            inputs=[
                IO.Image.Input("images", tooltip="Canonical reference photos of the character (10+ recommended, varied angles/expressions)."),
                IO.String.Input("project", default="", tooltip="Project/character name."),
                IO.String.Input("trigger_phrase", default="", optional=True,
                                 tooltip="Word/phrase used to invoke this LoRA later. Leave blank to reuse character.json's existing 'trigger'."),
                IO.Int.Input("steps", default=2500, min=1000, max=6000, step=100,
                             tooltip="Training steps. Cost scales linearly: steps x $0.0024."),
                IO.Float.Input("learning_rate", default=0.00009, min=0.00001, max=0.001, step=0.00001, optional=True, advanced=True),
            ],
            outputs=[
                IO.String.Output(display_name="lora_url"),
                IO.String.Output(display_name="character_json"),
            ],
            hidden=[IO.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls,
        images: Input.Image,
        project: str,
        trigger_phrase: str = "",
        steps: int = 2500,
        learning_rate: float = 0.00009,
    ) -> IO.NodeOutput:
        _require_fal_key()
        import fal_client

        project = logging_util.sanitize_project_name(project)
        logging_util.ensure_project(project)
        char_path = logging_util.character_json_path(project)
        import json

        with open(char_path, encoding="utf-8") as f:
            data = json.load(f)

        trigger = (trigger_phrase or data.get("trigger") or "").strip()
        if not trigger:
            raise ValueError(
                "No trigger phrase: pass one in, or set 'trigger' in character.json first."
            )

        n_images = images.shape[0]
        if n_images < 4:
            logger.warning("Training with only %d image(s); fal.ai recommends at least 10.", n_images)

        zip_bytes = _images_to_zip_bytes(images)
        images_data_url = await fal_client.upload_async(
            zip_bytes, content_type="application/zip", file_name="training_images.zip"
        )

        def on_update(update):
            if hasattr(update, "logs"):
                for log in update.logs or []:
                    logger.info("[fal LoRA training] %s", log.get("message", ""))

        result = await fal_client.subscribe_async(
            TRAIN_ENDPOINT,
            arguments={
                "images_data_url": images_data_url,
                "trigger_phrase": trigger,
                "steps": steps,
                "learning_rate": learning_rate,
            },
            with_logs=True,
            on_queue_update=on_update,
        )
        lora_file = result.get("diffusers_lora_file") or {}
        lora_url = lora_file.get("url")
        if not lora_url:
            raise RuntimeError(f"Training finished without a lora file in the response: {result!r}")

        data["lora_url"] = lora_url
        data["lora_base"] = "flux-1-dev"
        data["lora_trigger_phrase"] = trigger
        data["lora_trained_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        data["lora_steps"] = steps
        with open(char_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logging_util.append_generation_log(
            {
                "provider": "fal_ai",
                "model": "flux-lora-portrait-trainer",
                "operation": "lora_training",
                "project": project,
                "stage": "lora_training",
                "resolution_or_aspect": None,
                "quality": None,
                "num_outputs": 1,
                "seed": None,
                "job_id": None,
                "estimated_cost_usd": pricing.estimate_fal_training_cost_usd(steps),
                "actual_cost_usd": None,
                "prompt_id": None,
                "output_path": lora_url,
                "notes": f"trigger_phrase={trigger!r}",
            }
        )

        return IO.NodeOutput(lora_url, json.dumps(data, indent=2))


class AIInfluencerFalLoraImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerFalLoraImage",
            display_name="Generate with Character LoRA (fal.ai)",
            category="ai_influencer",
            description=(
                "Generates images with a trained character LoRA on fal.ai. base must match the "
                "model the LoRA was trained on: krea-2-turbo ($0.01/megapixel) for tools/train_lora.py "
                "runs, flux-1-dev ($0.035/megapixel) for the Train Character LoRA node. Wire lora_url "
                "from Load Character (after train_lora.py --pick) or paste one in."
            ),
            inputs=[
                IO.String.Input("prompt", multiline=True, default=""),
                IO.String.Input("lora_url", default="", tooltip="From Train Character LoRA or Load Character's lora_url output."),
                IO.Float.Input("lora_scale", default=1.0, min=-2.0, max=2.0, step=0.05, optional=True),
                IO.Combo.Input("image_size", options=_IMAGE_SIZES, default="portrait_4_3", optional=True),
                IO.Int.Input("custom_width", default=1024, min=256, max=2048, step=8, optional=True,
                             tooltip="Used only when image_size is 'custom'."),
                IO.Int.Input("custom_height", default=1024, min=256, max=2048, step=8, optional=True,
                             tooltip="Used only when image_size is 'custom'."),
                IO.Int.Input("num_inference_steps", default=28, min=1, max=50, optional=True, advanced=True,
                             tooltip="flux-1-dev only (Krea 2 Turbo is fixed at 8 steps)."),
                IO.Float.Input("guidance_scale", default=3.5, min=0.0, max=10.0, step=0.1, optional=True, advanced=True,
                               tooltip="flux-1-dev only."),
                IO.Int.Input("num_images", default=1, min=1, max=4, optional=True),
                IO.Int.Input("seed", default=0, min=0, max=2147483647, optional=True, control_after_generate=True),
                IO.Combo.Input("base", options=list(INFER_ENDPOINTS), default="krea-2-turbo", optional=True,
                               tooltip="The model the LoRA was trained on."),
            ],
            outputs=[IO.Image.Output()],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        lora_url: str,
        lora_scale: float = 1.0,
        image_size: str = "portrait_4_3",
        custom_width: int = 1024,
        custom_height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images: int = 1,
        seed: int = 0,
        base: str = "krea-2-turbo",
    ) -> IO.NodeOutput:
        _require_fal_key()
        import fal_client

        if not lora_url.strip():
            raise ValueError("lora_url is empty -- train a LoRA first (Train Character LoRA node) or paste one in.")
        if not prompt.strip():
            raise ValueError("prompt is empty.")

        size_arg = {"width": custom_width, "height": custom_height} if image_size == "custom" else image_size
        if base == "krea-2-turbo" and not 0 <= lora_scale <= 4:
            raise ValueError("Krea 2 LoRA scale must be between 0 and 4.")
        arguments = {
            "prompt": prompt,
            "loras": [{"path": lora_url.strip(), "scale": lora_scale}],
            "image_size": size_arg,
            "num_images": num_images,
        }
        if base == "flux-1-dev":
            arguments.update(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        if seed > 0:
            arguments["seed"] = seed

        result = await fal_client.subscribe_async(INFER_ENDPOINTS[base], arguments=arguments, with_logs=True)
        urls = [img["url"] for img in result.get("images", [])]
        if not urls:
            raise RuntimeError(f"fal.ai returned no images: {result!r}")

        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            tensors = []
            for url in urls:
                resp = await client.get(url)
                resp.raise_for_status()
                tensors.append(_url_to_image_tensor(resp.content))
        return IO.NodeOutput(torch.cat(tensors, dim=0))


class AIInfluencerToolkitFalExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [AIInfluencerTrainCharacterLoRA, AIInfluencerFalLoraImage]

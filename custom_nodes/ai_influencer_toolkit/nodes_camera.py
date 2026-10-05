"""Camera and sensor imperfections, applied after generation.

A diffusion output is too clean to pass as a photo somebody took: no sensor noise, perfectly
aligned color channels, even corner-to-corner exposure, textbook white balance. The prompt asks
for grain and lens artifacts (see nodes_character.MANDATORY_IMPERFECTIONS), but the model only
delivers them when it feels like it, so this node applies them deterministically, as the last
step of every image workflow.

Core's *Add Noise to Image* is not enough here: it adds one flat gaussian value to all three
channels at every pixel, which reads as dithering. Real ISO grain has a cell size that scales
with resolution, is strongest in the shadows and midtones, and comes with a separate,
lower-frequency chroma component. The lens part (channel misregistration towards the corners,
cos^4 falloff) has no equivalent node at all.

Sizes are percentages of the image's short side, like the background blur node, so the same
settings look the same before and after an upscale.

apply_camera_imperfections() is the pass itself, on a (B, C, H, W) tensor; tools/camera_pass.py
runs it over images that already exist without going through ComfyUI.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.model_management
from comfy_api.latest import IO, Input

_GAMMA = 2.2
_CHUNK = 16  # images per pass on the GPU


def _grain(b: int, h: int, w: int, cell: float, generator: torch.Generator, device, dtype) -> torch.Tensor:
    """Unit-variance noise whose grain clumps are roughly `cell` pixels across."""
    gh = max(1, round(h / cell))
    gw = max(1, round(w / cell))
    noise = torch.randn((b, 1, gh, gw), generator=generator, device="cpu").to(device=device, dtype=dtype)
    if (gh, gw) != (h, w):
        noise = F.interpolate(noise, size=(h, w), mode="bicubic", align_corners=False)
        noise = noise / noise.std().clamp_min(1e-6)
    return noise


def _chromatic_aberration(img: torch.Tensor, k: float) -> torch.Tensor:
    """Scale the red channel up and the blue channel down around the center, like a cheap lens."""
    b, _, h, w = img.shape
    ys = torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype).view(1, h, 1, 1).expand(1, h, w, 1)
    xs = torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype).view(1, 1, w, 1).expand(1, h, w, 1)
    grid = torch.cat([xs, ys], dim=-1)
    out = img.clone()
    for channel, scale in ((0, 1.0 + k), (2, 1.0 - k)):
        sampled = F.grid_sample(
            img[:, channel:channel + 1], (grid / scale).expand(b, -1, -1, -1),
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        out[:, channel:channel + 1] = sampled
    return out


def apply_camera_imperfections(
    img: torch.Tensor,
    seed: int,
    iso_grain: float,
    grain_size: float,
    chroma_noise: float,
    chromatic_aberration: float,
    vignette: float,
    color_jitter: float,
    color_seed: int | None = None,
) -> torch.Tensor:
    """The pass itself, on a (B, C, H, W) tensor in 0..1. Returns the same shape and device.
    With `color_seed`, every image in the batch gets the same white balance drift, drawn from that
    seed (video frames: a drift that changes every frame is flicker, not a camera)."""
    b, _, h, w = img.shape
    device, dtype = img.device, img.dtype
    short_side = min(h, w)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if chromatic_aberration > 0:
        # chromatic_aberration is the corner displacement as % of the short side; the sampling grid
        # is in [-1, 1] coordinates, where the edge is half the image away from the center.
        img = _chromatic_aberration(img, chromatic_aberration / 50.0)

    if vignette > 0 or color_jitter > 0:
        # Exposure-like effects belong in linear light, where doubling the value is one stop.
        linear = img.clamp_min(0) ** _GAMMA
        if vignette > 0:
            ys = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(1, 1, h, 1)
            xs = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, 1, 1, w)
            r2 = (xs * xs + ys * ys) / 2.0  # 1.0 at the corners
            linear = linear * (1.0 - vignette * r2 * r2).clamp_min(0)  # r^4, the cos^4 falloff
        if color_jitter > 0:
            if color_seed is None:
                gains = torch.randn((b, 3, 1, 1), generator=generator, device="cpu")
            else:
                gains = torch.randn((1, 3, 1, 1), generator=torch.Generator(device="cpu").manual_seed(color_seed), device="cpu")
            gains = 1.0 + gains.to(linear) * 0.05 * color_jitter
            linear = linear * gains.clamp_min(0)
        img = linear.clamp_min(0) ** (1.0 / _GAMMA)

    if iso_grain > 0:
        cell = max(1.0, grain_size / 100.0 * short_side)
        luma = (img * torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
        weight = 1.0 - 0.7 * luma.clamp(0, 1)  # sensor noise is buried in the highlights
        img = img + _grain(b, h, w, cell, generator, device, dtype) * (iso_grain * weight)
        if chroma_noise > 0:
            # Chroma noise is blotchier than luma noise and must not shift overall brightness.
            speckle = _grain(b * 3, h, w, cell * 2.0, generator, device, dtype).view(b, 3, h, w)
            speckle = speckle - speckle.mean(1, keepdim=True)
            img = img + speckle * (iso_grain * chroma_noise * weight)

    return img.clamp(0, 1)


class AIInfluencerCameraImperfections(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerCameraImperfections",
            display_name="Camera Imperfections (AI Influencer)",
            category="ai_influencer",
            search_aliases=["iso grain", "film grain", "chromatic aberration", "vignette"],
            description=(
                "Adds the sensor and lens artifacts a real camera leaves behind: ISO grain "
                "(luma + chroma, strongest in the shadows), channel misregistration towards the "
                "corners, corner falloff and a slight per-image white balance drift. Run it last, "
                "after any upscale, so nothing sharpens the grain away."
            ),
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, tooltip="Grain pattern and white balance drift."),
                IO.Float.Input("iso_grain", default=0.025, min=0.0, max=0.3, step=0.005, tooltip="Grain strength. 0.015 = clean daylight shot, 0.025 = ordinary phone photo, 0.05+ = pushed high ISO."),
                IO.Float.Input("grain_size", default=0.1, min=0.02, max=1.0, step=0.01, tooltip="Size of one grain clump, as % of the short side, so it survives an upscale. 0.1 = fine, 0.3 = coarse film."),
                IO.Float.Input("chroma_noise", default=0.35, min=0.0, max=1.0, step=0.05, tooltip="Color speckle, relative to iso_grain. 0 = black and white grain only."),
                IO.Float.Input("chromatic_aberration", default=0.05, min=0.0, max=1.0, step=0.01, tooltip="Red/blue fringing at the corners, as % of the short side. 0.05 is visible only when zoomed in, which is the point."),
                IO.Float.Input("vignette", default=0.15, min=0.0, max=1.0, step=0.05, tooltip="How much darker the corners are than the center."),
                IO.Float.Input("color_jitter", default=0.3, min=0.0, max=1.0, step=0.05, tooltip="Per-image white balance and exposure drift, so a set of images doesn't share one identical color grade."),
                IO.Boolean.Input("video_frames", default=False, optional=True, advanced=True,
                                 tooltip="On when the images are the frames of one clip: the white balance drift is the same on every "
                                         "frame (otherwise the clip flickers) while the grain still changes frame to frame, like a real sensor."),
            ],
            outputs=[IO.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        image: Input.Image,
        seed: int,
        iso_grain: float,
        grain_size: float,
        chroma_noise: float,
        chromatic_aberration: float,
        vignette: float,
        color_jitter: float,
        video_frames: bool = False,
    ) -> IO.NodeOutput:
        device = comfy.model_management.get_torch_device()
        chunks = []
        for i in range(0, image.shape[0], _CHUNK):  # a 5 s clip is 124 frames; don't hold them all on the GPU at once
            img = image[i:i + _CHUNK].to(device).movedim(-1, 1).float()
            out = apply_camera_imperfections(img, seed + i, iso_grain, grain_size, chroma_noise, chromatic_aberration,
                                             vignette, color_jitter, color_seed=seed if video_frames else None)
            chunks.append(out.movedim(1, -1).to(comfy.model_management.intermediate_device()))
        return IO.NodeOutput(torch.cat(chunks))

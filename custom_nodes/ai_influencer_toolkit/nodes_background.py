"""Background blur (fake shallow depth of field).

Prompting for "shallow depth of field" helps but is not reliable: the model decides, and the
amount of blur drifts between images. This node makes it deterministic, after generation. It
takes a subject mask (core ComfyUI's Load Background Removal Model -> Remove Background, BiRefNet)
and blurs everything outside it, so every image of the character comes out with the background
out of focus.

Three details make it look like a lens rather than a smudge:

- The background is blurred with the subject *excluded and renormalized*
  (blur(bg * bg_mask) / blur(bg_mask)), so the subject's colors don't bleed outwards into a halo
  around them, which is the usual give-away of a masked blur.
- Highlights are blurred in a linear-light space, where bright spots spread into visible bokeh
  instead of being averaged away, and mixed back in (`bokeh_highlights`).
- The mask edge is feathered, so hair and shoulders fade into the blur instead of looking cut out.

`blur_amount` and `feather` are percentages of the image's short side, so the same settings give
the same look before and after an upscale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import comfy.model_management
from comfy_api.latest import IO, Input

_GAMMA = 2.2


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur on a (B, C, H, W) tensor. Reflect-padded, so edges don't darken."""
    if sigma <= 0:
        return image
    radius = max(1, int(math.ceil(sigma * 3)))
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(x * x) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    out = F.pad(image, (radius, radius, 0, 0), mode="reflect")
    out = F.conv2d(out, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    out = F.pad(out, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(out, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)


class AIInfluencerBlurBackground(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerBlurBackground",
            display_name="Blur Background (AI Influencer)",
            category="ai_influencer",
            description=(
                "Blurs everything outside the subject mask, for a consistent shallow depth of "
                "field on every image. Feed it a mask from Remove Background (BiRefNet). Sizes "
                "are percentages of the image's short side, so results match before/after upscaling."
            ),
            inputs=[
                IO.Image.Input("image"),
                IO.Mask.Input("subject_mask", tooltip="White = stays sharp. From Remove Background."),
                IO.Float.Input("blur_amount", default=1.5, min=0.0, max=10.0, step=0.1, tooltip="Blur radius as % of the short side. ~0.8 = subtle, 1.5 = portrait lens, 3+ = extreme."),
                IO.Float.Input("feather", default=0.4, min=0.0, max=5.0, step=0.1, tooltip="Softness of the mask edge, as % of the short side, so hair and shoulders blend into the blur."),
                IO.Float.Input("bokeh_highlights", default=0.3, min=0.0, max=1.0, step=0.05, tooltip="How much bright spots bloom into bokeh. 0 = plain blur."),
                IO.Float.Input("subject_edge_shift", default=0.0, min=-3.0, max=3.0, step=0.1, optional=True, tooltip="Move the sharp/blurred boundary, as % of the short side. Negative blurs slightly into the subject's outline, positive keeps a sharp rim around them."),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(display_name="mask_used", tooltip="The feathered mask, for checking the edge."),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: Input.Image,
        subject_mask: torch.Tensor,
        blur_amount: float,
        feather: float,
        bokeh_highlights: float,
        subject_edge_shift: float = 0.0,
    ) -> IO.NodeOutput:
        device = comfy.model_management.get_torch_device()
        img = image.to(device).movedim(-1, 1).float()  # B,C,H,W
        b, _, h, w = img.shape
        short_side = min(h, w)

        mask = subject_mask.to(device).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(1)  # B,1,H,W
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        if mask.shape[0] != b:
            mask = mask.expand(b, -1, -1, -1)
        mask = mask.clamp(0, 1)

        if subject_edge_shift:
            # Grow/shrink the mask by thresholding a softened copy: below 0.5 grows it, above shrinks.
            softened = _gaussian_blur(mask, abs(subject_edge_shift) / 100.0 * short_side / 2)
            mask = (softened > (0.15 if subject_edge_shift > 0 else 0.85)).float()

        sigma = blur_amount / 100.0 * short_side / 3.0  # blur_amount is a radius; sigma ~ radius/3
        if sigma <= 0:
            return IO.NodeOutput(image, mask.squeeze(1).cpu())

        bg_weight = 1.0 - mask
        eps = 1e-6

        def blur_background(x: torch.Tensor) -> torch.Tensor:
            """Blur only background pixels, renormalized so the subject doesn't bleed outwards."""
            weighted = _gaussian_blur(x * bg_weight, sigma)
            norm = _gaussian_blur(bg_weight, sigma)
            return weighted / norm.clamp_min(eps)

        blurred = blur_background(img)
        if bokeh_highlights > 0:
            linear = blur_background(img.clamp_min(0) ** _GAMMA).clamp_min(0) ** (1.0 / _GAMMA)
            blurred = torch.lerp(blurred, linear, bokeh_highlights)

        soft_mask = _gaussian_blur(mask, feather / 100.0 * short_side / 3.0) if feather > 0 else mask
        out = blurred * (1.0 - soft_mask) + img * soft_mask
        out = out.clamp(0, 1).movedim(1, -1).to(comfy.model_management.intermediate_device())
        return IO.NodeOutput(out, soft_mask.squeeze(1).to(comfy.model_management.intermediate_device()))

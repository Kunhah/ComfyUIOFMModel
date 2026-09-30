"""Best-effort local snapshot of published per-request rates for the two providers whose
official ComfyUI price-badge formula is a simple, stable lookup table: Krea 2 and OpenAI
GPT Image 2.5 (see comfy_api_nodes/nodes_krea.py and comfy_api_nodes/nodes_openai.py,
ComfyUI v0.35.0). These mirror the `price_badge` JSONata tables ComfyUI itself ships, not an
invented number. They are an *estimate only* for the generations.jsonl log: the ComfyUI node
canvas always shows the live, authoritative price badge before you click Queue, and your real
balance is at https://platform.comfy.org.

Seedance/Seedream video and OpenRouter pricing depend on too many runtime factors (duration,
reference-video counts, per-model token rates) to mirror safely here, so estimate_cost_usd()
returns None for them and the logger records technical metadata only.

If ComfyUI updates these rates, this file will drift — re-check against the price badge
`expr` in the node source above and update the tables here.
"""

from __future__ import annotations

KREA_RATES_USD: dict[str, dict[str, float]] = {
    "Krea 2 Medium Turbo": {"text": 0.015, "style": 0.0175, "moodboard": 0.02},
    "Krea 2 Medium": {"text": 0.03, "style": 0.035, "moodboard": 0.04},
    "Krea 2 Large": {"text": 0.06, "style": 0.065, "moodboard": 0.07},
}

GPT_IMAGE_PRESETS_USD: dict[str, dict[str, dict[str, float]]] = {
    "gpt-image-2": {
        "low": {"1024x1024": 0.0071, "1024x1536": 0.0057, "1536x1024": 0.0057, "2048x2048": 0.0143,
                "2048x1152": 0.0057, "1152x2048": 0.0057, "3840x2160": 0.0134, "2160x3840": 0.0134},
        "medium": {"1024x1024": 0.0632, "1024x1536": 0.0494, "1536x1024": 0.0494, "2048x2048": 0.1284,
                   "2048x1152": 0.0509, "1152x2048": 0.0509, "3840x2160": 0.1201, "2160x3840": 0.1201},
        "high": {"1024x1024": 0.2529, "1024x1536": 0.1976, "1536x1024": 0.1976, "2048x2048": 0.5138,
                 "2048x1152": 0.2034, "1152x2048": 0.2034, "3840x2160": 0.4803, "2160x3840": 0.4803},
    },
    "gpt-image-2.5": {
        "low": {"1024x1024": 0.0084, "1024x1536": 0.0068, "1536x1024": 0.0068, "2048x2048": 0.0170,
                "2048x1152": 0.0067, "1152x2048": 0.0067, "3840x2160": 0.0159, "2160x3840": 0.0159},
        "medium": {"1024x1024": 0.0188, "1024x1536": 0.0147, "1536x1024": 0.0147, "2048x2048": 0.0383,
                   "2048x1152": 0.0157, "1152x2048": 0.0157, "3840x2160": 0.0371, "2160x3840": 0.0371},
        "high": {"1024x1024": 0.0753, "1024x1536": 0.0589, "1536x1024": 0.0589, "2048x2048": 0.1531,
                 "2048x1152": 0.0606, "1152x2048": 0.0606, "3840x2160": 0.1431, "2160x3840": 0.1431},
        "xhigh": {"1024x1024": 0.1339, "1024x1536": 0.1055, "1536x1024": 0.1055, "2048x2048": 0.2721,
                  "2048x1152": 0.1077, "1152x2048": 0.1077, "3840x2160": 0.2544, "2160x3840": 0.2544},
        "max": {"1024x1024": 0.3013, "1024x1536": 0.2354, "1536x1024": 0.2354, "2048x2048": 0.6123,
                "2048x1152": 0.2424, "1152x2048": 0.2424, "3840x2160": 0.5724, "2160x3840": 0.5724},
    },
}

GPT_IMAGE_REF_IMAGE_SURCHARGE_USD: dict[str, float] = {
    "gpt-image-1": 0.0019,
    "gpt-image-1.5": 0.0016,
    "gpt-image-2": 0.0098,
    "gpt-image-2.5-flare": 0.0117,
    "gpt-image-2.5-sunburst": 0.0117,
}


def estimate_krea_cost_usd(model: str) -> float | None:
    """Baseline text-prompt rate for a Krea 2 model. Does not account for a moodboard or
    chained style_reference images, which raise the price — check the in-canvas badge for those."""
    rates = KREA_RATES_USD.get(model)
    return rates["text"] if rates else None


def estimate_gpt_image_cost_usd(model: str, quality: str, size: str, n: int = 1, n_reference_images: int = 0) -> float | None:
    """Per-request estimate for one OpenAI GPT Image 2.5/2/1.5/1 call at a known preset size.
    Returns None for a custom/unlisted size rather than guessing."""
    family = "gpt-image-2.5" if model in ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst") else model
    per_image = GPT_IMAGE_PRESETS_USD.get(family, {}).get(quality, {}).get(size)
    if per_image is None:
        return None
    surcharge = GPT_IMAGE_REF_IMAGE_SURCHARGE_USD.get(model, 0.0) * n_reference_images
    return round((per_image + surcharge) * n, 6)


FAL_TRAINING_USD_PER_STEP = 0.0024
"""fal-ai/flux-lora-portrait-trainer, per fal.ai's own pricing page (Sept 2026): $0.0024/step,
minimum 1000 steps billed."""
FAL_TRAINING_MIN_BILLED_STEPS = 1000

FAL_INFERENCE_USD_PER_MEGAPIXEL = 0.035
"""fal-ai/flux-lora, per fal.ai's own pricing page (Sept 2026): $0.035/megapixel, rounded up."""

_FAL_IMAGE_SIZE_MEGAPIXELS = {
    "square_hd": 1024 * 1024 / 1_000_000,
    "square": 512 * 512 / 1_000_000,
    "portrait_4_3": 864 * 1152 / 1_000_000,
    "portrait_16_9": 720 * 1280 / 1_000_000,
    "landscape_4_3": 1152 * 864 / 1_000_000,
    "landscape_16_9": 1280 * 720 / 1_000_000,
}


FAL_KREA2_TRAINING_USD_PER_STEP = 0.003
"""fal-ai/krea-2-trainer, per fal.ai's own pricing page (Sept 2026): $0.003/step, 100-step minimum."""
FAL_KREA2_TRAINING_MIN_BILLED_STEPS = 100

FAL_KREA2_LORA_USD_PER_MEGAPIXEL = 0.01
"""fal-ai/krea-2/turbo/lora, per fal.ai's own pricing page (Sept 2026): $0.01/megapixel."""


def estimate_fal_krea2_training_cost_usd(steps: int) -> float:
    return round(max(steps, FAL_KREA2_TRAINING_MIN_BILLED_STEPS) * FAL_KREA2_TRAINING_USD_PER_STEP, 4)


def estimate_fal_krea2_image_cost_usd(width: int, height: int, n: int = 1) -> float:
    """Assumes fractional-megapixel billing; if fal rounds up per image, 1024x1536 costs $0.02, not $0.0157."""
    return round(width * height / 1_000_000 * FAL_KREA2_LORA_USD_PER_MEGAPIXEL * n, 4)


def estimate_fal_training_cost_usd(steps: int) -> float:
    return round(max(steps, FAL_TRAINING_MIN_BILLED_STEPS) * FAL_TRAINING_USD_PER_STEP, 4)


def estimate_fal_inference_cost_usd(image_size: str, n: int = 1, custom_width: int = 0, custom_height: int = 0) -> float | None:
    if image_size == "custom":
        if not (custom_width and custom_height):
            return None
        mp = (custom_width * custom_height) / 1_000_000
    else:
        mp = _FAL_IMAGE_SIZE_MEGAPIXELS.get(image_size)
        if mp is None:
            return None
    import math

    return round(math.ceil(mp) * FAL_INFERENCE_USD_PER_MEGAPIXEL * n, 6)


def estimate_cost_usd(provider: str, model: str, **kwargs) -> float | None:
    if provider == "krea":
        return estimate_krea_cost_usd(model)
    if provider == "openai_gpt_image":
        quality = kwargs.get("quality", "")
        size = kwargs.get("size", "")
        n = kwargs.get("n", 1)
        n_reference_images = kwargs.get("n_reference_images", 0)
        return estimate_gpt_image_cost_usd(model, quality, size, n, n_reference_images)
    if provider == "fal_ai" and model == "flux-lora-portrait-trainer":
        return estimate_fal_training_cost_usd(kwargs.get("steps", 2500))
    if provider == "fal_ai" and model == "flux-lora":
        return estimate_fal_inference_cost_usd(
            kwargs.get("size", "portrait_4_3"), kwargs.get("n", 1),
            kwargs.get("custom_width", 0), kwargs.get("custom_height", 0),
        )
    return None

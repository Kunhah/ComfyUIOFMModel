#!/usr/bin/env python3
"""Generate the local (open-weights) Krea 2 workflows 07-09 into
user/default/workflows/ai_influencer/. Re-run after editing this file; it overwrites them.

    07_lora_checkpoint_tester.json   same prompts + seed on every checkpoint, one grid
    08_krea2_local_master.json       Krea 2 Turbo + character LoRA -> skin polish -> SeedVR2 upscale
    09_krea2_pose_transfer.json      copy the pose/composition of a reference photo
    10_preview_then_4k.json          cheap small previews, then upscale the one you picked to 4K
    12_camera_pass.json              the camera-imperfections pass on an image you already have
    09c_practice_no_gpu.json         09b's clicks without Krea 2, to rehearse before renting a GPU
"""
from __future__ import annotations

import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "user", "default", "workflows", "ai_influencer")

KREA_UNET = "krea2_turbo_fp8_scaled.safetensors"
KREA_TE = "qwen3vl_4b_fp8_scaled.safetensors"
KREA_VAE = "qwen_image_vae.safetensors"
SEEDVR_UNET = "seedvr2_3b_int8_convrot.safetensors"
SEEDVR_VAE = "seedvr2_ema_vae_fp16.safetensors"
SIGCLIP = "sigclip_vision_patch14_384.safetensors"
BIREFNET = "birefnet.safetensors"

NEVER = 2  # LiteGraph node mode: muted, not executed until you enable it
BYPASS = 4  # LiteGraph node mode: passes its input straight through

MODEL_URLS = {
    KREA_UNET: ("diffusion_models", "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors"),
    KREA_TE: ("text_encoders", "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"),
    KREA_VAE: ("vae", "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors"),
    SEEDVR_UNET: ("diffusion_models", "https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_3b_int8_convrot.safetensors"),
    SEEDVR_VAE: ("vae", "https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/vae/seedvr2_ema_vae_fp16.safetensors"),
    SIGCLIP: ("clip_vision", "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors"),
    BIREFNET: ("background_removal", "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors"),
}

MODELS_NOTE = """## Models (put them in ComfyUI/models/...)

| File | Folder |
|---|---|
""" + "\n".join(f"| [{name}]({url}) | `{folder}/` |" for name, (folder, url) in MODEL_URLS.items()) + """

Your character LoRA checkpoints go in `models/loras/` (a subfolder per character is fine, e.g. `models/loras/lina/`).

Krea 2 needs a big GPU (RTX 5090 / RTX 6000 Pro class). Run these on RunPod, not a 6 GB card."""


class Graph:
    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self.groups: list[dict] = []

    def node(self, type_: str, pos, widgets=None, size=(320, 120), inputs=(), outputs=(), title=None, mode=0) -> dict:
        n = {
            "id": len(self.nodes) + 1,
            "type": type_,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": len(self.nodes),
            "mode": mode,
            "inputs": [{"name": name, "type": t, "link": None} for name, t in inputs],
            "outputs": [{"name": name, "type": t, "links": [], "slot_index": i} for i, (name, t) in enumerate(outputs)],
            "properties": {"Node name for S&R": type_},
            "widgets_values": list(widgets or []),
        }
        if title:
            n["title"] = title
        models = [
            {"name": w, "url": MODEL_URLS[w][1], "directory": MODEL_URLS[w][0]}
            for w in n["widgets_values"] if isinstance(w, str) and w in MODEL_URLS
        ]
        if models:
            n["properties"]["models"] = models
        self.nodes.append(n)
        return n

    def link(self, src: dict, src_slot: int, dst: dict, dst_name: str, widget: bool = False):
        link_id = len(self.links) + 1
        t = src["outputs"][src_slot]["type"]
        inp = next((i for i in dst["inputs"] if i["name"] == dst_name), None)
        if inp is None:
            inp = {"name": dst_name, "type": t, "link": None}
            dst["inputs"].append(inp)
        if widget:
            inp["widget"] = {"name": dst_name}
        inp["link"] = link_id
        src["outputs"][src_slot]["links"].append(link_id)
        self.links.append([link_id, src["id"], src_slot, dst["id"], dst["inputs"].index(inp), t])

    def note(self, pos, text, size=(420, 300)):
        return self.node("MarkdownNote", pos, [text], size=size)

    def group(self, title, x, y, w, h, color="#3f789e"):
        self.groups.append({"id": len(self.groups) + 1, "title": title, "bounding": [x, y, w, h], "color": color, "font_size": 24, "flags": {}})

    def dump(self, filename: str):
        data = {
            "last_node_id": len(self.nodes),
            "last_link_id": len(self.links),
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {"ds": {"scale": 0.6, "offset": [0, 0]}},
            "version": 0.4,
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        print("wrote", filename)


def krea_loaders(g: Graph, x: int, y: int):
    unet = g.node("UNETLoader", (x, y), [KREA_UNET, "default"], inputs=[], outputs=[("MODEL", "MODEL")])
    clip = g.node("CLIPLoader", (x, y + 150), [KREA_TE, "krea2", "default"], outputs=[("CLIP", "CLIP")])
    vae = g.node("VAELoader", (x, y + 300), [KREA_VAE], outputs=[("VAE", "VAE")])
    return unet, clip, vae


def ksampler(g: Graph, pos, seed_mode="randomize", steps=8, cfg=1.0, denoise=1.0, title=None):
    return g.node(
        "KSampler", pos, [42, seed_mode, steps, cfg, "euler", "simple", denoise], size=(320, 262), title=title,
        inputs=[("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
        outputs=[("LATENT", "LATENT")],
    )


def encode_prompt(g: Graph, clip, text_src, pos):
    enc = g.node("CLIPTextEncode", pos, [""], size=(320, 100), inputs=[("clip", "CLIP")], outputs=[("CONDITIONING", "CONDITIONING")])
    g.link(clip, 0, enc, "clip")
    g.link(text_src, 0, enc, "text", widget=True)
    neg = g.node("ConditioningZeroOut", (pos[0], pos[1] + 140), size=(320, 30), inputs=[("conditioning", "CONDITIONING")], outputs=[("CONDITIONING", "CONDITIONING")])
    g.link(enc, 0, neg, "conditioning")
    return enc, neg


def character_prompt(g: Graph, x: int, y: int, project: str, scene: str, action: str, clothing: str, environment: str, camera: str, lighting: str):
    char = g.node("AIInfluencerLoadCharacter", (x, y), [project], size=(320, 150), outputs=[
        ("name", "STRING"), ("trigger", "STRING"), ("identity_prompt", "STRING"), ("lora_url", "STRING"), ("character_json", "STRING")])
    # identity_preservation_instructions is about reference-image edits; blank for LoRA text-to-image.
    # realism_instructions is left blank: the Prompt Builder always appends its own mandatory
    # imperfection block, which the Camera Imperfections node then delivers for real.
    builder = g.node(
        "AIInfluencerPromptBuilder", (x + 360, y),
        ["", scene, action, clothing, environment, camera, lighting, "", "", ""],
        size=(400, 620), inputs=[("identity", "STRING")], outputs=[("prompt", "STRING")],
    )
    g.link(char, 2, builder, "identity", widget=True)
    return char, builder


def build_07():
    g = Graph()
    g.group("01 MODELS (Krea 2 Turbo: test on the model you generate with)", 0, 0, 400, 520)
    unet, clip, vae = krea_loaders(g, 30, 60)

    g.group("02 CHECKPOINT TEST", 430, 0, 820, 1000, color="#8A8")
    tester = g.node(
        "AIInfluencerLoraCheckpointTester", (460, 60),
        ["lina", "l1na", 4, 3000, True, "",
         "photo of [trigger], close-up portrait, looking at the camera, soft window light\n"
         "photo of [trigger] laughing at a cafe table, candid smartphone photo\n"
         "full body photo of [trigger] walking down a city street at night, wearing a red coat\n"
         "photo of [trigger] in profile, hiking on a mountain trail, overcast daylight",
         "1.0", 42, 8, 1.0, "euler", "simple", 832, 1024, "", False],
        size=(420, 900),
        inputs=[("model", "MODEL"), ("clip", "CLIP"), ("vae", "VAE"), ("clip_vision", "CLIP_VISION")],
        outputs=[("grid", "IMAGE"), ("images", "IMAGE"), ("report", "STRING"), ("best_lora", "STRING")],
    )
    g.link(unet, 0, tester, "model")
    g.link(clip, 0, tester, "clip")
    g.link(vae, 0, tester, "vae")
    report = g.node("PreviewAny", (910, 60), size=(320, 400), inputs=[("source", "*")], outputs=[("STRING", "STRING")])
    g.link(tester, 2, report, "source")

    g.group("03 SCORING (optional, connect to enable)", 0, 540, 400, 200, color="#a1309b")
    g.node("CLIPVisionLoader", (30, 600), [SIGCLIP], size=(340, 60), outputs=[("CLIP_VISION", "CLIP_VISION")])

    g.note((1280, 0), f"""# 07 LoRA Checkpoint Tester

Don't just take the last checkpoint. Past a point the LoRA overtrains, and the best one is often somewhere between 2,000 and 3,000 steps. This workflow renders **the same prompts with the same seed on every checkpoint**, so only the checkpoint changes, and puts the results side by side in one grid.

1. Copy your checkpoints (usually the last 4) into `models/loras/lina/`. AI Toolkit names them `l1na_000002250.safetensors` ... and the final one `l1na.safetensors`.
2. `lora_filter` = part of the file name (`l1na`). `last_n` = how many of the newest to test (0 = all). `final_step` = your total steps (3000), so the final file is labelled by step.
3. Queue. Columns = checkpoints (plus a no-LoRA column if `include_baseline` is on), rows = prompts. Use prompts with outfits and places that are **not** in your dataset: an overtrained LoRA copies dataset backgrounds and clothes.
4. Pick the checkpoint that keeps the face the most stable **and** still follows the prompt. Record it:
   `python custom_nodes/ai_influencer_toolkit/tools/pick_checkpoint.py lina --filter l1na --step 2500`
   Workflows 08/09 then use it automatically.

**Optional scores:** connect *Load CLIP Vision* to `clip_vision` and put your dataset images in `input/ai_influencer/lina/dataset/`. Each column then shows
- `id`: similarity to the dataset (higher = more like the character)
- `copy`: similarity to the closest single training image. If `copy` keeps rising while `id` stays flat, the later checkpoints are memorizing photos.

CLIP isn't a face recognizer, so use the scores as a tie-breaker, not a verdict. With `save_best_to_character` on, the best `id` checkpoint is written to character.json.

Every run is saved to `output/projects/lina/lora_tests/<time>/` (grid.png + report.json).

{MODELS_NOTE}""", size=(560, 1000))
    g.dump("07_lora_checkpoint_tester.json")


def seedvr_upscale(g: Graph, image_src, x: int, y: int, scale: float = 2.0):
    """The official SeedVR2 3B image-upscale template, unpacked from its subgraph."""
    unet = g.node("UNETLoader", (x, y), [SEEDVR_UNET, "default"], outputs=[("MODEL", "MODEL")])
    vae = g.node("VAELoader", (x, y + 150), [SEEDVR_VAE], outputs=[("VAE", "VAE")])
    resize = g.node("ResizeImageMaskNode", (x, y + 280), ["scale by multiplier", scale, "lanczos"], size=(320, 110),
                    inputs=[("input", "*")], outputs=[("resized", "*")], title="Upscale factor")
    g.link(image_src, 0, resize, "input")
    resize["outputs"][0]["type"] = "IMAGE"
    pre = g.node("SeedVR2Preprocess", (x + 360, y), size=(280, 30), inputs=[("resized_images", "IMAGE")], outputs=[("IMAGE", "IMAGE")])
    g.link(resize, 0, pre, "resized_images")
    enc = g.node("VAEEncodeTiled", (x + 360, y + 80), [512, 128, 4096, 8], size=(280, 150),
                 inputs=[("pixels", "IMAGE"), ("vae", "VAE")], outputs=[("LATENT", "LATENT")])
    g.link(pre, 0, enc, "pixels")
    g.link(vae, 0, enc, "vae")
    cond = g.node("SeedVR2Conditioning", (x + 360, y + 270), size=(280, 50), inputs=[("model", "MODEL"), ("vae_conditioning", "LATENT")],
                  outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")])
    g.link(unet, 0, cond, "model")
    g.link(enc, 0, cond, "vae_conditioning")
    samp = ksampler(g, (x + 680, y), seed_mode="fixed", steps=1, cfg=1.0, title="SeedVR2 sampler")
    g.link(unet, 0, samp, "model")
    g.link(cond, 0, samp, "positive")
    g.link(cond, 1, samp, "negative")
    g.link(enc, 0, samp, "latent_image")
    dec = g.node("VAEDecodeTiled", (x + 680, y + 300), [512, 128, 4096, 8], size=(320, 150),
                 inputs=[("samples", "LATENT"), ("vae", "VAE")], outputs=[("IMAGE", "IMAGE")])
    g.link(samp, 0, dec, "samples")
    g.link(vae, 0, dec, "vae")
    post = g.node("SeedVR2PostProcessing", (x + 680, y + 490), ["lab"], size=(320, 80),
                  inputs=[("images", "IMAGE"), ("original_resized_images", "IMAGE")], outputs=[("images", "IMAGE")])
    g.link(dec, 0, post, "images")
    g.link(resize, 0, post, "original_resized_images")
    return post


def blur_background(g: Graph, image_src, x: int, y: int, blur_amount: float = 1.5):
    """Subject mask (BiRefNet) -> blur everything else, so the background is always out of focus."""
    bg_model = g.node("LoadBackgroundRemovalModel", (x, y), [BIREFNET], size=(340, 60), outputs=[("bg_model", "BACKGROUND_REMOVAL")])
    mask = g.node("RemoveBackground", (x, y + 120), size=(320, 50),
                  inputs=[("bg_removal_model", "BACKGROUND_REMOVAL"), ("image", "IMAGE")], outputs=[("mask", "MASK")])
    g.link(bg_model, 0, mask, "bg_removal_model")
    g.link(image_src, 0, mask, "image")
    blur = g.node("AIInfluencerBlurBackground", (x, y + 220), [blur_amount, 0.4, 0.3, 0.0], size=(340, 180),
                  inputs=[("image", "IMAGE"), ("subject_mask", "MASK")],
                  outputs=[("image", "IMAGE"), ("mask_used", "MASK")], title="Blur Background")
    g.link(image_src, 0, blur, "image")
    g.link(mask, 0, blur, "subject_mask")
    return blur


def camera_imperfections(g: Graph, image_src, x: int, y: int):
    """ISO grain, lens fringing and corner falloff: what a real camera leaves behind and a
    diffusion model doesn't. Always last, after the upscale and the blur."""
    cam = g.node("AIInfluencerCameraImperfections", (x, y), [42, "randomize", 0.025, 0.1, 0.35, 0.05, 0.15, 0.3],
                 size=(340, 280), inputs=[("image", "IMAGE")], outputs=[("image", "IMAGE")], title="Camera Imperfections")
    g.link(image_src, 0, cam, "image")
    return cam


def lora_model(g: Graph, unet, x: int, y: int, project: str):
    apply = g.node("AIInfluencerApplyCharacterLora", (x, y), [project, "(from character.json)", -1.0], size=(320, 130),
                   inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL"), ("trigger", "STRING"), ("lora_name", "STRING")])
    g.link(unet, 0, apply, "model")
    return apply


def save_node(g: Graph, image_src, pos, stage: str, model: str, prompt_src=None, operation="generate", project="lina", provider="local_krea2"):
    save = g.node("AIInfluencerSaveImage", pos, [project, stage, provider, model, operation, "", "", "", -1, 0, "", ""],
                  size=(360, 420), inputs=[("images", "IMAGE"), ("prompt_text", "STRING")], outputs=[("IMAGE", "IMAGE"), ("output_path", "STRING")])
    g.link(image_src, 0, save, "images")
    if prompt_src is not None:
        g.link(prompt_src, 0, save, "prompt_text", widget=True)
    return save


def build_08():
    g = Graph()
    g.group("01 MODELS + CHARACTER LORA", 0, 0, 400, 680)
    unet, clip, vae = krea_loaders(g, 30, 60)
    model = lora_model(g, unet, 30, 510, "lina")

    g.group("02 PROMPT (trigger + permanent features come from character.json)", 430, 0, 830, 700, color="#b58b2a")
    _, builder = character_prompt(
        g, 460, 60, "lina",
        scene="candid photo taken on an iPhone",
        action="sitting on a windowsill, holding a coffee cup, smiling softly",
        clothing="oversized cream knit sweater",
        environment="small bright apartment, plants in the background",
        camera="85mm f/1.8 portrait lens, shallow depth of field, background strongly out of focus, creamy bokeh, eye level",
        lighting="soft morning window light",
    )

    g.group("03 BASE IMAGE (Krea 2 Turbo, 8 steps, cfg 1)", 1290, 0, 760, 700, color="#8A8")
    pos, neg = encode_prompt(g, clip, builder, (1320, 60))
    res = g.node("ResolutionSelector", (1320, 300), ["3:4 (Portrait Standard)", 1, 8], size=(320, 130), outputs=[("width", "INT"), ("height", "INT")])
    latent = g.node("EmptyLatentImage", (1320, 470), [1024, 1024, 1], size=(320, 110),
                    inputs=[("width", "INT"), ("height", "INT")], outputs=[("LATENT", "LATENT")])
    g.link(res, 0, latent, "width", widget=True)
    g.link(res, 1, latent, "height", widget=True)
    base = ksampler(g, (1680, 60), title="Base sampler")
    g.link(model, 0, base, "model")
    g.link(pos, 0, base, "positive")
    g.link(neg, 0, base, "negative")
    g.link(latent, 0, base, "latent_image")
    base_img = g.node("VAEDecode", (1680, 360), size=(320, 50), inputs=[("samples", "LATENT"), ("vae", "VAE")], outputs=[("IMAGE", "IMAGE")])
    g.link(base, 0, base_img, "samples")
    g.link(vae, 0, base_img, "vae")
    save_node(g, base_img, (1680, 440), "local_candidate", "Krea 2 Turbo + LoRA", builder)

    g.group("04 SKIN POLISH (low-denoise second pass at 1.25x)", 0, 740, 1260, 520, color="#a1309b")
    polish_text = g.node("PrimitiveStringMultiline", (30, 800),
                         ["highly detailed natural skin texture, visible pores, fine peach fuzz, subtle skin imperfections, sharp eyes and eyelashes, individual hair strands"],
                         size=(340, 160), outputs=[("STRING", "STRING")], title="Polish details")
    concat = g.node("StringConcatenate", (30, 1000), ["", "", ", "], size=(340, 140),
                    inputs=[("string_a", "STRING"), ("string_b", "STRING")], outputs=[("STRING", "STRING")])
    g.link(builder, 0, concat, "string_a", widget=True)
    g.link(polish_text, 0, concat, "string_b", widget=True)
    ppos, pneg = encode_prompt(g, clip, concat, (400, 800))
    up = g.node("ImageScaleBy", (400, 1060), ["lanczos", 1.25], size=(320, 90), inputs=[("image", "IMAGE")], outputs=[("IMAGE", "IMAGE")])
    g.link(base_img, 0, up, "image")
    enc = g.node("VAEEncode", (400, 1180), size=(320, 50), inputs=[("pixels", "IMAGE"), ("vae", "VAE")], outputs=[("LATENT", "LATENT")])
    g.link(up, 0, enc, "pixels")
    g.link(vae, 0, enc, "vae")
    polish = ksampler(g, (760, 800), denoise=0.3, title="Polish sampler (denoise 0.2-0.35)")
    g.link(model, 0, polish, "model")
    g.link(ppos, 0, polish, "positive")
    g.link(pneg, 0, polish, "negative")
    g.link(enc, 0, polish, "latent_image")
    polished = g.node("VAEDecode", (760, 1100), size=(320, 50), inputs=[("samples", "LATENT"), ("vae", "VAE")], outputs=[("IMAGE", "IMAGE")])
    g.link(polish, 0, polished, "samples")
    g.link(vae, 0, polished, "vae")

    g.group("05 SEEDVR2 UPSCALE (2x)", 1290, 740, 1060, 640, color="#3f789e")
    final = seedvr_upscale(g, polished, 1320, 800, scale=2.0)

    g.group("06 BACKGROUND BLUR (always on)", 2380, 740, 420, 640, color="#a1309b")
    blurred = blur_background(g, final, 2410, 800)

    g.group("07 CAMERA IMPERFECTIONS (always on)", 2840, 740, 420, 640, color="#a1309b")
    shot = camera_imperfections(g, blurred, 2870, 800)

    g.group("08 OUTPUT", 3300, 740, 760, 640, color="#444")
    save_node(g, shot, (3330, 800), "local_final", "Krea 2 Turbo + LoRA + polish + SeedVR2 + bg blur + camera", builder, operation="refine")
    cmp_ = g.node("ImageCompare", (3720, 800), [], size=(320, 480), inputs=[("image_a", "IMAGE"), ("image_b", "IMAGE")], title="Before / after")
    g.link(base_img, 0, cmp_, "image_a")
    g.link(shot, 0, cmp_, "image_b")

    g.note((2080, 0), f"""# 08 Krea 2 Local Master (character LoRA)

Generate with **Krea 2 Turbo** plus your character LoRA (trained on RAW, see README).

- **01**: *Apply Character LoRA* uses the checkpoint you picked in Workflow 07 (`local_lora` in `output/projects/lina/character.json`). To try another file, choose it in `lora_name`.
- **02**: the trigger word and permanent features come from character.json. Only edit scene / action / clothing / environment / camera / lighting.
- **03**: base image, saved to `outputs/images/candidates/`.
- **04 + 05**: the "premium" chain: a gentle second pass for skin detail, then a SeedVR2 2x upscale.
- **06 Background blur**: a subject mask (BiRefNet) drives a real depth-of-field blur, so the background is out of focus on *every* image instead of only when the model feels like it. It runs last, after the upscale, so nothing sharpens it again. `blur_amount` and `feather` are percentages of the image's short side, so the look stays the same at any resolution: ~0.8 subtle, 1.5 portrait lens, 3+ extreme. For a sharp background set `blur_amount` to 0 or mute the group.
- **07 Camera imperfections**: ISO grain (heavier in the shadows, plus a little chroma speckle), red/blue fringing towards the corners, corner falloff and a per-image white balance drift. This is the pass that stops the output reading as AI: the prompt asks for these artifacts, the model half-ignores it, this node makes them certain. It runs after the upscale, so nothing sharpens the grain away. `iso_grain` 0.015 is a clean daylight shot, 0.025 an ordinary phone photo, 0.05+ pushed high ISO.
- **08**: saved to `outputs/images/final/`, with a before/after slider.

**Base image only:** right-click the groups 04-08 → *Set Group Nodes to Never* (or select them and press Ctrl+M).

**Polish too plastic or changing the face?** Lower the polish denoise (0.2). **Not enough detail?** Raise it (max ~0.35, beyond that the face drifts).

Change `lina` in the Load Character, Apply Character LoRA and Save nodes for another character.

{MODELS_NOTE}""", size=(560, 700))
    g.dump("08_krea2_local_master.json")


def build_09():
    g = Graph()
    g.group("01 MODELS + CHARACTER LORA", 0, 0, 400, 680)
    unet, clip, vae = krea_loaders(g, 30, 60)
    model = lora_model(g, unet, 30, 510, "lina")

    g.group("02 POSE REFERENCE (input/ai_influencer/lina/poses/)", 430, 0, 400, 700, color="#b58b2a")
    ref = g.node("LoadImage", (460, 60), ["example.png", "image"], size=(340, 380), outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    scaled = g.node("ImageScaleToTotalPixels", (460, 480), ["lanczos", 1.0, 16], size=(340, 130),
                    inputs=[("image", "IMAGE")], outputs=[("IMAGE", "IMAGE")])
    g.link(ref, 0, scaled, "image")

    g.group("03 PROMPT", 860, 0, 830, 700, color="#b58b2a")
    _, builder = character_prompt(
        g, 890, 60, "lina",
        scene="photo taken on an iPhone",
        action="same pose as the reference",
        clothing="black leather jacket, white t-shirt, blue jeans",
        environment="rooftop terrace at sunset, city skyline behind",
        camera="85mm f/1.8 portrait lens, shallow depth of field, background strongly out of focus, creamy bokeh, eye level",
        lighting="warm golden hour light",
    )

    g.group("04 GENERATE (img2img from the pose reference)", 1720, 0, 760, 700, color="#8A8")
    pos, neg = encode_prompt(g, clip, builder, (1750, 60))
    enc = g.node("VAEEncode", (1750, 300), size=(320, 50), inputs=[("pixels", "IMAGE"), ("vae", "VAE")], outputs=[("LATENT", "LATENT")])
    g.link(scaled, 0, enc, "pixels")
    g.link(vae, 0, enc, "vae")
    samp = ksampler(g, (2110, 60), denoise=0.8, title="Pose sampler (denoise 0.7-0.9)")
    g.link(model, 0, samp, "model")
    g.link(pos, 0, samp, "positive")
    g.link(neg, 0, samp, "negative")
    g.link(enc, 0, samp, "latent_image")
    img = g.node("VAEDecode", (2110, 360), size=(320, 50), inputs=[("samples", "LATENT"), ("vae", "VAE")], outputs=[("IMAGE", "IMAGE")])
    g.link(samp, 0, img, "samples")
    g.link(vae, 0, img, "vae")
    g.group("05 BACKGROUND BLUR (always on)", 2510, 0, 420, 700, color="#a1309b")
    blurred = blur_background(g, img, 2540, 60)

    g.group("06 CAMERA IMPERFECTIONS (always on)", 2950, 0, 420, 700, color="#a1309b")
    shot = camera_imperfections(g, blurred, 2980, 60)

    save_node(g, shot, (1750, 400), "local_candidate", "Krea 2 Turbo + LoRA (pose img2img) + bg blur + camera", builder)
    cmp_ = g.node("ImageCompare", (2110, 440), [], size=(320, 240), inputs=[("image_a", "IMAGE"), ("image_b", "IMAGE")], title="Reference / result")
    g.link(scaled, 0, cmp_, "image_a")
    g.link(shot, 0, cmp_, "image_b")

    g.note((3410, 0), f"""# 09 Krea 2 Pose Transfer

Put any photo with the pose you want in `input/ai_influencer/lina/poses/` and load it in 02. Your character is generated in the same pose and framing.

**How it works, honestly:** this ComfyUI build has **no ControlNet for Krea 2**, so this is image-to-image. The reference is encoded and re-noised, and `denoise` sets how much of it survives:
- **0.7**: pose, framing *and* colors/lighting of the reference stick (face may pull toward the reference person)
- **0.8**: good default, pose and framing kept, your character's face
- **0.9**: only the rough layout is kept

It copies composition, not an exact skeleton. If the pose must match exactly, a Krea 2 ControlNet (not available here yet) or a pose-capable edit model is needed.

**05 Background blur** always puts the background out of focus (BiRefNet subject mask + a real lens-style blur, same node as Workflow 08). Set `blur_amount` to 0 or mute the group to keep it sharp.

**06 Camera imperfections** adds the ISO grain, lens fringing, corner falloff and white balance drift a real camera leaves behind, which is what keeps the result from reading as AI. Same node and same defaults as Workflow 08.

For **edits of an existing image** (hair color, outfit, background) use Workflow 03 (GPT Image 2.5 edit), with the image as the candidate and a canonical photo as the reference.

{MODELS_NOTE}""", size=(560, 700))
    g.dump("09_krea2_pose_transfer.json")


def build_10():
    g = Graph()
    g.group("01 MODELS + CHARACTER LORA", 0, 0, 400, 680)
    unet, clip, vae = krea_loaders(g, 30, 60)
    model = lora_model(g, unet, 30, 510, "lina")

    g.group("02 PROMPT", 430, 0, 830, 700, color="#b58b2a")
    _, builder = character_prompt(
        g, 460, 60, "lina",
        scene="candid photo taken on an iPhone",
        action="sitting on a windowsill, holding a coffee cup, smiling softly",
        clothing="oversized cream knit sweater",
        environment="small bright apartment, plants in the background",
        camera="85mm f/1.8 portrait lens, shallow depth of field, background strongly out of focus, creamy bokeh, eye level",
        lighting="soft morning window light",
    )

    g.group("03 PREVIEW — 4 small candidates (queue this first)", 1290, 0, 1120, 760, color="#8A8")
    pos, neg = encode_prompt(g, clip, builder, (1320, 60))
    res = g.node("ResolutionSelector", (1320, 300), ["3:4 (Portrait Standard)", 1, 8], size=(320, 130),
                 outputs=[("width", "INT"), ("height", "INT")], title="Preview size (lower megapixels = faster)")
    latent = g.node("EmptyLatentImage", (1320, 470), [1024, 1024, 4], size=(320, 110),
                    inputs=[("width", "INT"), ("height", "INT")], outputs=[("LATENT", "LATENT")], title="4 candidates per queue")
    g.link(res, 0, latent, "width", widget=True)
    g.link(res, 1, latent, "height", widget=True)
    preview_sampler = ksampler(g, (1680, 60), title="Preview sampler")
    g.link(model, 0, preview_sampler, "model")
    g.link(pos, 0, preview_sampler, "positive")
    g.link(neg, 0, preview_sampler, "negative")
    g.link(latent, 0, preview_sampler, "latent_image")
    preview_img = g.node("VAEDecode", (1680, 360), size=(320, 50), inputs=[("samples", "LATENT"), ("vae", "VAE")], outputs=[("IMAGE", "IMAGE")])
    g.link(preview_sampler, 0, preview_img, "samples")
    g.link(vae, 0, preview_img, "vae")
    save_node(g, preview_img, (1680, 440), "local_candidate", "Krea 2 Turbo + LoRA (preview)", builder)
    shown = g.node("PreviewImage", (2050, 60), size=(340, 400), inputs=[("images", "IMAGE")], title="Pick one of these")
    g.link(preview_img, 0, shown, "images")

    # Everything below is muted until you confirm a candidate.
    confirm_start = len(g.nodes)
    g.group("04 CONFIRM → 4K (muted: unmute after picking)", 0, 800, 1260, 700, color="#a1309b")
    chosen = g.node("LoadImageOutput", (30, 860), ["", "image"], size=(340, 400),
                    outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], title="The candidate you picked (refresh, then choose)")
    upscaled = seedvr_upscale(g, chosen, 420, 860, scale=4.0)

    g.group("05 BACKGROUND BLUR + CAMERA + SAVE", 1290, 800, 800, 940, color="#444")
    blurred = blur_background(g, upscaled, 1320, 860)
    shot = camera_imperfections(g, blurred, 1320, 1320)
    save_node(g, shot, (1680, 860), "local_final", "Krea 2 Turbo + LoRA + SeedVR2 4x + bg blur + camera", builder, operation="refine")
    final_preview = g.node("PreviewImage", (1680, 1320), size=(340, 160), inputs=[("images", "IMAGE")], title="4K result")
    g.link(shot, 0, final_preview, "images")
    for n in g.nodes[confirm_start:]:
        n["mode"] = NEVER

    g.note((2440, 0), f"""# 10 Preview first, 4K after

ComfyUI has no "pause and ask" node, so this is two queues on one canvas.

**1. Preview.** Queue as is: group 03 renders **4 small candidates** at ~1 MP (8 steps, no polish, no upscale, no blur — the cheap part). They are saved to `outputs/images/candidates/` and shown in *Pick one of these*. Queue again for 4 more. Lower the `megapixels` on *Preview size* for faster, rougher previews.

**2. Confirm.** When you like one:
- right-click group **03** → *Set Group Nodes to Never* (so it doesn't re-roll), and groups **04** and **05** → *Set Group Nodes to Always*;
- in *The candidate you picked*, press the refresh button and select that candidate's file (it lists the output folder, newest first);
- Queue. SeedVR2 upscales it **4x** — a ~1 MP preview (≈900x1200) becomes ≈3600x4800, i.e. comfortably 4K — then the background blur and the camera imperfections pass (ISO grain, lens fringing, corner falloff) run and it is saved to `outputs/images/final/`.

**Why the 4K pass upscales the picture instead of re-generating it at 4K:** a diffusion model with the same seed at a different resolution gives a *different* picture — different pose, different framing. The only way "the same image, bigger" works is to take the pixels you approved and add detail to them. That is what SeedVR2 does here.

Want the skin-polish pass as well? Workflow 08 has it (generate → polish → 2x upscale → blur) in one queue; this workflow trades that for the preview/confirm split.

The preview PNGs carry the full workflow inside them, so you can drag one back onto the canvas to recover its exact seed and prompt.

{{MODELS_NOTE}}""".replace("{{MODELS_NOTE}}", MODELS_NOTE), size=(560, 900))
    g.dump("10_preview_then_4k.json")


def build_12():
    g = Graph()
    g.group("01 THE IMAGE YOU ALREADY HAVE", 0, 0, 400, 520, color="#b58b2a")
    chosen = g.node("LoadImageOutput", (30, 60), ["", "image"], size=(340, 400),
                    outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], title="Pick one (refresh, newest first)")

    g.group("02 BACKGROUND BLUR (bypassed: the image usually has it already)", 430, 0, 420, 520, color="#a1309b")
    blur_start = len(g.nodes)
    blurred = blur_background(g, chosen, 460, 60)
    for n in g.nodes[blur_start:]:
        n["mode"] = BYPASS

    g.group("03 CAMERA PASS", 880, 0, 420, 520, color="#8A8")
    shot = camera_imperfections(g, blurred, 910, 60)

    g.group("04 OUTPUT", 1330, 0, 800, 520, color="#444")
    save_node(g, shot, (1360, 60), "local_final", "camera pass", operation="refine")
    cmp_ = g.node("ImageCompare", (1750, 60), [], size=(340, 400), inputs=[("image_a", "IMAGE"), ("image_b", "IMAGE")], title="Before / after")
    g.link(chosen, 0, cmp_, "image_a")
    g.link(shot, 0, cmp_, "image_b")

    g.note((2160, 0), """# 12 Camera pass on an existing image

The same *Camera Imperfections* node that Workflows 02-10 run at the end, on its own, for pictures
that were generated before it existed or that came from somewhere else. ISO grain, red/blue
fringing towards the corners, corner falloff, and a white balance drift that changes with the seed.

1. **01**: press refresh on the loader and pick any file in `output/`. For a picture that isn't in
   the output folder, drop it in `input/` and swap this node for a normal *Load Image*.
2. **02** is bypassed (Ctrl+B) because a final image from Workflow 08/09/10 already has its
   background blurred. Un-bypass it for an image that doesn't.
3. **03**: `iso_grain` 0.015 is a clean daylight shot, 0.025 an ordinary phone photo, 0.05+ pushed
   high ISO. The seed changes the grain pattern and the white balance drift, so re-queueing gives a
   different roll without touching anything else.
4. **04** saves to `outputs/images/final/` and logs the pass, with a before/after slider.

For a whole folder at once, skip ComfyUI:

    python custom_nodes/ai_influencer_toolkit/tools/camera_pass.py output/projects/lina/outputs/images/final

It runs the same code, gives each file its own grain from its name, and writes `<name>_camera.<ext>`
next to each input (`--in-place` to overwrite, `--out DIR` to collect them elsewhere).""", size=(560, 520))
    g.dump("12_camera_pass.json")


def build_09c():
    """09b with the diffusion step taken out: same mask, same prompt fields, same filters, same
    save, but her photo is pasted into the mask instead of generated. No model files, no GPU, so
    it runs on any machine and you can rehearse the clicks before the pod meter starts."""
    g = Graph()
    g.group("01 LOCATION PHOTO (paint the mask here)", 0, 0, 400, 700, color="#b58b2a")
    plate = g.node("LoadImage", (30, 60), ["ai_influencer/MyCharacter/scenes/practice_plate.png", "image"], size=(340, 380),
                   outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], title="LOCATION plate (mask this one)")
    scaled = g.node("ImageScaleToTotalPixels", (30, 480), ["lanczos", 1.0, 16], size=(340, 130),
                    inputs=[("image", "IMAGE")], outputs=[("IMAGE", "IMAGE")])
    g.link(plate, 0, scaled, "image")

    g.group("02 THE MASK", 430, 0, 400, 220, color="#b58b2a")
    grow = g.node("GrowMask", (460, 60), [12, True], size=(340, 110),
                  inputs=[("mask", "MASK")], outputs=[("MASK", "MASK")], title="Grow mask (feather the seam)")
    g.link(plate, 1, grow, "mask")

    g.group("03 HER (stands in for the Krea 2 generation)", 430, 260, 400, 500, color="#8A8")
    her = g.node("LoadImage", (460, 320), ["ai_influencer/MyCharacter/dataset/29.png", "image"], size=(340, 380),
                 outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], title="Any photo of her")

    g.group("04 PROMPT (the real thing, same fields as 09b)", 860, 0, 830, 760, color="#b58b2a")
    _, builder = character_prompt(
        g, 890, 60, "",
        scene="photo taken on an iPhone",
        action="relaxed expression, looking toward the camera",
        clothing="black leather jacket, white t-shirt, blue jeans",
        environment="rooftop terrace at sunset, city skyline behind",
        camera="85mm f/1.8 portrait lens, shallow depth of field, background strongly out of focus, eye level",
        lighting="warm golden hour light",
    )
    prompt_preview = g.node("PreviewAny", (890, 260), size=(320, 420), inputs=[("source", "*")], outputs=[("STRING", "STRING")], title="The prompt Krea 2 would get")
    g.link(builder, 0, prompt_preview, "source")

    g.group("05 FIXED BACKGROUND", 1720, 0, 400, 300, color="#a1309b")
    composite = g.node("ImageCompositeMasked", (1750, 60), [0, 0, True], size=(340, 150),
                       inputs=[("destination", "IMAGE"), ("source", "IMAGE"), ("mask", "MASK")],
                       outputs=[("IMAGE", "IMAGE")], title="Keep background pixel-identical")
    g.link(scaled, 0, composite, "destination")
    g.link(her, 0, composite, "source")
    g.link(grow, 0, composite, "mask")

    g.group("06 FILTERS (the same two nodes as 09b group 08)", 1720, 340, 400, 560, color="#a1309b")
    blur = g.node("AIInfluencerBlurBackground", (1750, 400), [1.5, 0.4, 0.3, 0.0], size=(340, 180),
                  inputs=[("image", "IMAGE"), ("subject_mask", "MASK")],
                  outputs=[("image", "IMAGE"), ("mask_used", "MASK")], title="Blur Background")
    g.link(composite, 0, blur, "image")
    g.link(grow, 0, blur, "subject_mask")  # on the pod this mask comes from BiRefNet instead
    shot = camera_imperfections(g, blur, 1750, 620)

    g.group("07 OUTPUT", 2150, 0, 800, 560, color="#444")
    save_node(g, shot, (2180, 60), "misc_image", "practice run (no GPU)", builder,
              operation="other", project="", provider="other")
    cmp_ = g.node("ImageCompare", (2570, 60), [], size=(340, 480), inputs=[("image_a", "IMAGE"), ("image_b", "IMAGE")], title="Plate / result")
    g.link(scaled, 0, cmp_, "image_a")
    g.link(shot, 0, cmp_, "image_b")

    g.note((2980, 0), """# 09c Practice run (no GPU, no models)

Workflow 09b's clicks, minus Krea 2. Everything here runs on CPU in a few seconds, so you can
learn the moves before the vast.ai meter starts. Queue it as is first, then change things.

## The loop you are rehearsing
1. **01**: right-click the plate -> **Open in MaskEditor**, paint where she should be, Save. The
   shape you paint is where she lands and how big she is. `practice_plate.png` already has one
   painted, so you can see the effect before doing your own.
2. **03**: any photo of her. On the pod this loader doesn't exist -- Krea 2 *generates* what goes
   inside the mask, using the character LoRA and the reference photos. Here the photo is just
   pasted in so you can see where the mask puts it.
3. **04**: the real Prompt Builder. Type in the fields and watch *The prompt Krea 2 would get*:
   the mandatory "don't look AI" block is appended for you, every time, whatever you type.
4. **05**: `ImageCompositeMasked` -- outside the painted mask, the plate is bit-identical. That is
   what "fixed background" means.
5. **06**: the two filter nodes, exactly as 09b runs them: background blur, then ISO grain, corner
   fringing, falloff and a white balance drift. Queue again for a different grain roll.
6. **07**: saved to `output/projects/MyCharacter/outputs/images/misc/` and logged, with a
   plate/result slider.

## What is missing on purpose
Krea 2 + your character LoRA, the identity reference LoRA, and the SDPose skeleton -- all of them
need model files and a big GPU. In 09b they sit between groups 04 and 05. Nothing else differs:
same mask, same prompt node, same filters, same save.

One honest difference: here the blur's subject mask is the mask you painted; on the pod BiRefNet
cuts her out properly, so hair and shoulders blend instead of following your brush.""", size=(560, 780))
    g.dump("09c_practice_no_gpu.json")


if __name__ == "__main__":
    build_07()
    build_08()
    build_09()
    build_10()
    build_12()
    build_09c()

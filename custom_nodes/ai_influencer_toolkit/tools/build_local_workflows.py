#!/usr/bin/env python3
"""Generate the local (open-weights) Krea 2 workflows 07-09 into
user/default/workflows/ai_influencer/. Re-run after editing this file; it overwrites them.

    07_lora_checkpoint_tester.json   same prompts + seed on every checkpoint, one grid
    08_krea2_local_master.json       Krea 2 Turbo + character LoRA -> skin polish -> SeedVR2 upscale
    09_krea2_pose_transfer.json      copy the pose/composition of a reference photo
    10_preview_then_4k.json          cheap small previews, then upscale the one you picked to 4K
    12_camera_pass.json              the camera-imperfections pass on an image you already have
    09c_practice_no_gpu.json         09b's clicks without Krea 2, to rehearse before renting a GPU
    01_character_creation.json       GPT Image 2.5: front portrait candidates -> character sheets -> example photos
    13_pick_winner.json              pick one of a 09b queue's candidates into final/
    14_wan_animate_motion_transfer.json  the person in a still copies the movement of a video (Wan Animate 2)
"""
from __future__ import annotations

import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_ANGLES = "\n".join(f"{yaw}: {words}" for yaw, words in [
        (0, "the body facing the camera"),
        (35, "the body turned slightly toward frame-right, face toward the camera"),
        (-35, "the body turned slightly toward frame-left, face toward the camera"),
        (65, "three-quarter view, the body turned toward frame-right, head turned back toward the camera"),
        (-65, "three-quarter view, the body turned toward frame-left, head turned back toward the camera"),
        (90, "seen in profile, facing frame-right"),
        (-90, "seen in profile, facing frame-left"),
        (160, "seen from behind, looking back over the shoulder toward the camera"),
    ])  # the node's own default (nodes_variation.ANGLES); kept here so this script needs no ComfyUI
OUT_DIR = os.path.join(REPO_ROOT, "user", "default", "workflows", "ai_influencer")

KREA_UNET = "krea2_turbo_fp8_scaled.safetensors"
KREA_TE = "qwen3vl_4b_fp8_scaled.safetensors"
KREA_VAE = "qwen_image_vae.safetensors"
SEEDVR_UNET = "seedvr2_3b_int8_convrot.safetensors"
SEEDVR_VAE = "seedvr2_ema_vae_fp16.safetensors"
SIGCLIP = "sigclip_vision_patch14_384.safetensors"
BIREFNET = "birefnet.safetensors"
CHOOSE_PICTURE = "ai_influencer_choose_a_picture.png"  # input_listing.PLACEHOLDER: made in input/ on startup; Run asks for a real one

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
    # Bypassed until you pick a file: any extra LoRA (NSFW / concept / style fine-tune) stacks on the character LoRA.
    extra = g.node("LoraLoaderModelOnly", (x, y + 160), ["", 1.0], size=(320, 130), inputs=[("model", "MODEL")],
                   outputs=[("MODEL", "MODEL")], title="Optional extra LoRA - NSFW / concept fine-tune (Ctrl+B to enable)", mode=BYPASS)
    g.link(apply, 0, extra, "model")
    return extra


def save_node(g: Graph, image_src, pos, stage: str, model: str, prompt_src=None, operation="generate", project="lina", provider="local_krea2",
              quality="", size="", n_refs=0, subfolder=""):
    # "fixed" is the frontend's control_after_generate value for the `seed` widget, which it adds to any seed input
    save = g.node("AIInfluencerSaveImage", pos, [project, stage, provider, model, operation, "", quality, size, -1, "fixed", n_refs, "", "", subfolder],
                  size=(360, 420), inputs=[("images", "IMAGE"), ("prompt_text", "STRING")], outputs=[("IMAGE", "IMAGE"), ("output_path", "STRING")])
    g.link(image_src, 0, save, "images")
    if prompt_src is not None:
        g.link(prompt_src, 0, save, "prompt_text", widget=True)
    return save


def character_views(g: Graph, builder, x=430, y=1440, seed=42, candidates=1, angles=DEFAULT_ANGLES):
    g.group("CHARACTER VIEWS — optional MATRIX expression / angle batches", x, y, 1040, 680, color="#b58b2a")
    views = g.node("AIInfluencerCharacterViews", (x + 30, y + 60),
                   ["manual", seed, "randomize", builder["widgets_values"][2], builder["widgets_values"][5], candidates, angles],
                   size=(440, 540), outputs=[("subject_action", "STRING"), ("camera_and_lens", "STRING"),
                                             ("seed", "INT"), ("folder", "STRING"), ("view", "STRING"),
                                             ("candidates", "INT"), ("angles", "STRING")])
    for output in views["outputs"]:
        output["shape"] = 6
    g.link(views, 0, builder, "subject_action", widget=True)
    g.link(views, 1, builder, "camera_and_lens", widget=True)
    g.note((x + 510, y + 60), """## Optional view batch (same Krea model and vast.ai Pod queue)

**manual**: edit action and camera here; other Prompt Builder fields still control identity, clothing, location and light.
Choose **head**, **head360**, **upper360**, **full360** for eight views, **all 32**, or one named view. Each gets its own seed and one candidate; manual mode keeps the configured candidate count. The view selector replaces action and framing only; no model or resolution changes.

Queue normally with **Hold for the pod**. The existing vast.ai path runs each view and saves filtered / clean copies as configured by this workflow.

08 uses the character LoRA to hold identity. In pose-based workflows, the photo and skeleton still constrain composition: a text request for a back/profile view cannot guarantee a real 3D turn. Use a matching pose photo or raise sampler denoise to allow larger changes. 09b gets the preset's matching skeleton angle; placement and reference-pose controls still apply. Its manual angle list is edited here.

Adapted from MATRIX LAB's 32-view prompts (Apache-2.0). Source and license: `view_presets/` in the toolkit. No Qwen models are needed.""", size=(470, 540))
    return views


def fuse_character_views(g: Graph):
    builder = next(n for n in g.nodes if n["type"] == "AIInfluencerPromptBuilder")
    mapper = next((n for n in g.nodes if n["type"] in ("AIInfluencerPlacementVariations", "AIInfluencerPoseAngle")), None)
    samplers = [n for n in g.nodes if n["type"] == "KSampler"]
    latent = next((n for n in g.nodes if n["type"] == "EmptyLatentImage"), None)
    seed = (mapper or samplers[0])["widgets_values"][0]
    candidates = mapper["widgets_values"][2] if mapper else latent["widgets_values"][2] if latent else 1
    y = max(group["bounding"][1] + group["bounding"][3] for group in g.groups) + 60
    angles = mapper["widgets_values"][-1] if mapper and mapper["type"] == "AIInfluencerPlacementVariations" else DEFAULT_ANGLES
    views = character_views(g, builder, y=y, seed=seed, candidates=candidates, angles=angles)
    if mapper:
        g.link(views, 2, mapper, "seed", widget=True)
        g.link(views, 5, mapper, "candidates", widget=True)
        if mapper["type"] == "AIInfluencerPlacementVariations":
            g.link(views, 6, mapper, "angles", widget=True)
    else:
        for sample in samplers:
            g.link(views, 2, sample, "seed", widget=True)
        if latent:
            g.link(views, 5, latent, "batch_size", widget=True)
        for save in [n for n in g.nodes if n["type"] == "AIInfluencerSaveImage"]:
            # The independent, uploaded-image confirmation pass in 10 must run only once.
            if save["mode"] == NEVER:
                continue
            g.link(views, 2, save, "seed", widget=True)
            g.link(views, 3, save, "subfolder", widget=True)
            g.link(views, 4, save, "notes", widget=True)


def build_08():
    g = Graph()
    g.group("01 MODELS + CHARACTER LORA", 0, 0, 400, 840)
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
    fuse_character_views(g)
    g.dump("08_krea2_local_master.json")


def build_09():
    g = Graph()
    g.group("01 MODELS + CHARACTER LORA", 0, 0, 400, 840)
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

    filtered_save = save_node(g, shot, (1750, 400), "local_candidate", "Krea 2 Turbo + LoRA (pose img2img) + bg blur + camera", builder)
    filtered_save["title"] = "Save filtered copy"
    raw_save = save_node(g, img, (3410, 60), "local_unfiltered", "Krea 2 Turbo + LoRA (pose img2img)", builder)
    raw_save["title"] = "Save unfiltered copy"
    cmp_ = g.node("ImageCompare", (2110, 440), [], size=(320, 240), inputs=[("image_a", "IMAGE"), ("image_b", "IMAGE")], title="Reference / result")
    g.link(scaled, 0, cmp_, "image_a")
    g.link(shot, 0, cmp_, "image_b")

    g.note((3800, 0), f"""# 09 Krea 2 Pose Transfer

Put any photo with the pose you want in `input/ai_influencer/lina/poses/` and load it in 02. Your character is generated in the same pose and framing.

**How it works, honestly:** this ComfyUI build has **no ControlNet for Krea 2**, so this is image-to-image. The reference is encoded and re-noised, and `denoise` sets how much of it survives:
- **0.7**: pose, framing *and* colors/lighting of the reference stick (face may pull toward the reference person)
- **0.8**: good default, pose and framing kept, your character's face
- **0.9**: only the rough layout is kept

It copies composition, not an exact skeleton. If the pose must match exactly, a Krea 2 ControlNet (not available here yet) or a pose-capable edit model is needed.

**05 Background blur** always puts the background out of focus (BiRefNet subject mask + a real lens-style blur, same node as Workflow 08). Set `blur_amount` to 0 or mute the group to keep it sharp.

**06 Camera imperfections** adds the ISO grain, lens fringing, corner falloff and white balance drift a real camera leaves behind, which is what keeps the result from reading as AI. Same node and same defaults as Workflow 08.

Every queue saves both versions: the normal filtered image in `outputs/images/candidates/` and the clean pre-filter image in `outputs/images/unfiltered/`, so you can apply a lighter pass later.

For **edits of an existing image** (hair color, outfit, background) there is no workflow any more: Workflow 03 (GPT Image 2.5 edit) was removed. Re-generate here with a changed prompt instead.

{MODELS_NOTE}""", size=(560, 700))
    fuse_character_views(g)
    g.dump("09_krea2_pose_transfer.json")


def build_10():
    g = Graph()
    g.group("01 MODELS + CHARACTER LORA", 0, 0, 400, 840)
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
    chosen = g.node("LoadImage", (30, 860), [CHOOSE_PICTURE, "image"], size=(340, 400),
                    outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], title="The candidate you picked (upload it here)")
    upscaled = seedvr_upscale(g, chosen, 420, 860, scale=4.0)

    g.group("05 BACKGROUND BLUR + CAMERA + SAVE", 1290, 800, 800, 940, color="#444")
    blurred = blur_background(g, upscaled, 1320, 860)
    shot = camera_imperfections(g, blurred, 1320, 1320)
    save_node(g, shot, (1680, 860), "local_final", "Krea 2 Turbo + LoRA + SeedVR2 4x + bg blur + camera", operation="refine")
    final_preview = g.node("PreviewImage", (1680, 1320), size=(340, 160), inputs=[("images", "IMAGE")], title="4K result")
    g.link(shot, 0, final_preview, "images")
    for n in g.nodes[confirm_start:]:
        n["mode"] = NEVER

    g.note((2440, 0), f"""# 10 Preview first, 4K after

ComfyUI has no "pause and ask" node, so this is two queues on one canvas.

**1. Preview.** Queue as is: group 03 renders **4 small candidates** at ~1 MP (8 steps, no polish, no upscale, no blur — the cheap part). They are saved to `outputs/images/candidates/` and shown in *Pick one of these*. Queue again for 4 more. Lower the `megapixels` on *Preview size* for faster, rougher previews.

**2. Confirm.** When you like one:
- right-click group **03** → *Set Group Nodes to Never* (so it doesn't re-roll), and groups **04** and **05** → *Set Group Nodes to Always*;
- in *The candidate you picked*, click *choose file to upload* and pick that candidate from `output/projects/<project>/outputs/images/candidates/` (or drag it onto the node);
- Queue. SeedVR2 upscales it **4x** — a ~1 MP preview (≈900x1200) becomes ≈3600x4800, i.e. comfortably 4K — then the background blur and the camera imperfections pass (ISO grain, lens fringing, corner falloff) run and it is saved to `outputs/images/final/`.

**Why the 4K pass upscales the picture instead of re-generating it at 4K:** a diffusion model with the same seed at a different resolution gives a *different* picture — different pose, different framing. The only way "the same image, bigger" works is to take the pixels you approved and add detail to them. That is what SeedVR2 does here.

Want the skin-polish pass as well? Workflow 08 has it (generate → polish → 2x upscale → blur) in one queue; this workflow trades that for the preview/confirm split.

The preview PNGs carry the full workflow inside them, so you can drag one back onto the canvas to recover its exact seed and prompt.

{{MODELS_NOTE}}""".replace("{{MODELS_NOTE}}", MODELS_NOTE), size=(560, 900))
    fuse_character_views(g)
    g.dump("10_preview_then_4k.json")


def build_12():
    g = Graph()
    g.group("01 THE IMAGE YOU ALREADY HAVE", 0, 0, 400, 520, color="#b58b2a")
    chosen = g.node("LoadImage", (30, 60), [CHOOSE_PICTURE, "image"], size=(340, 400),
                    outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], title="Your picture (upload or drag it here)")

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

1. **01**: click *choose file to upload* on the loader and pick the picture from any folder (a
   final image is in `output/projects/<project>/outputs/images/final/`), or drag it onto the node.
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
    """09b with the diffusion step taken out: same candidates-per-queue, same mask, same prompt
    fields, same filters, same save, but where 09b generates her, the skeleton Krea 2 would be
    given is pasted into the mask instead. No model files, no GPU, so it runs on any machine and
    you can rehearse the clicks before the pod meter starts."""
    g = Graph()
    g.group("01 LOCATION PHOTO (painting a mask is optional)", 0, 0, 400, 700, color="#b58b2a")
    plate = g.node("LoadImage", (30, 60), ["ai_influencer/MyCharacter/scenes/practice_plate.png", "image"], size=(340, 380),
                   outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], title="LOCATION plate")
    scaled = g.node("ImageScaleToTotalPixels", (30, 480), ["lanczos", 1.0, 16], size=(340, 130),
                    inputs=[("image", "IMAGE")], outputs=[("IMAGE", "IMAGE")])
    g.link(plate, 0, scaled, "image")

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

    g.group("02 CANDIDATES: a place and an angle per seed (same node as 09b)", 430, 0, 400, 1000, color="#8A8")
    vary = g.node("AIInfluencerPlacementVariations", (460, 60), [0, "randomize", 4, "placement + angle", "new angle per seed", 0.15, 0.85, 0.45, 0.7, 0.35, DEFAULT_ANGLES],
                  size=(340, 400), inputs=[("plate", "IMAGE"), ("prompt", "STRING"), ("pose_keypoints", "POSE_KEYPOINT")],
                  outputs=[("keypoints", "POSE_KEYPOINT"), ("prompt", "STRING"), ("seed", "INT"), ("folder", "STRING"), ("label", "STRING")],
                  title="Placement & Angle per Seed")
    for o in vary["outputs"]:
        o["shape"] = 6  # list output
    g.link(scaled, 0, vary, "plate")
    g.link(builder, 0, vary, "prompt")
    labels = g.node("PreviewAny", (460, 500), size=(340, 460), inputs=[("source", "*")], outputs=[("STRING", "STRING")], title="What each candidate got")
    g.link(vary, 4, labels, "source")

    g.group("03 WHERE SHE GOES (per candidate)", 0, 1040, 1500, 620, color="#8A8")
    auto = g.node("AIInfluencerAutoMask", (30, 1100), ["auto", 4.0, "center"], size=(340, 200),
                  inputs=[("plate", "IMAGE"), ("painted_mask", "MASK"), ("pose_keypoints", "POSE_KEYPOINT")],
                  outputs=[("mask", "MASK"), ("preview", "IMAGE"), ("method", "STRING"), ("keypoints", "POSE_KEYPOINT")], title="Auto Mask")
    g.link(scaled, 0, auto, "plate")
    g.link(plate, 1, auto, "painted_mask")
    g.link(vary, 0, auto, "pose_keypoints")
    grow = g.node("GrowMask", (30, 1340), [12, True], size=(340, 110),
                  inputs=[("mask", "MASK")], outputs=[("MASK", "MASK")], title="Grow mask (feather the seam)")
    g.link(auto, 0, grow, "mask")
    skeleton = g.node("SDPoseDrawKeypoints", (400, 1100), [True, True, True, False, 4, 2, 0.5, True], size=(320, 260),
                      inputs=[("keypoints", "POSE_KEYPOINT")], outputs=[("IMAGE", "IMAGE")], title="Skeleton (picture 3 in 09b)")
    g.link(auto, 3, skeleton, "keypoints")
    masks = g.node("PreviewImage", (750, 1100), size=(340, 500), inputs=[("images", "IMAGE")], title="Masks (red = where she goes)")
    g.link(auto, 1, masks, "images")
    skel_preview = g.node("PreviewImage", (1120, 1100), size=(340, 500), inputs=[("images", "IMAGE")], title="Skeletons")
    g.link(skeleton, 0, skel_preview, "images")

    g.group("05 FIXED BACKGROUND (the skeleton stands in for Krea 2)", 1720, 0, 400, 300, color="#a1309b")
    composite = g.node("ImageCompositeMasked", (1750, 60), [0, 0, False], size=(340, 150),
                       inputs=[("destination", "IMAGE"), ("source", "IMAGE"), ("mask", "MASK")],
                       outputs=[("IMAGE", "IMAGE")], title="Keep background pixel-identical")
    g.link(scaled, 0, composite, "destination")
    g.link(skeleton, 0, composite, "source")
    g.link(grow, 0, composite, "mask")

    g.group("06 FILTERS (the same two nodes as 09b group 08)", 1720, 340, 400, 560, color="#a1309b")
    blur = g.node("AIInfluencerBlurBackground", (1750, 400), [1.5, 0.4, 0.3, 0.0], size=(340, 180),
                  inputs=[("image", "IMAGE"), ("subject_mask", "MASK")],
                  outputs=[("image", "IMAGE"), ("mask_used", "MASK")], title="Blur Background")
    g.link(composite, 0, blur, "image")
    g.link(grow, 0, blur, "subject_mask")  # on the pod this mask comes from BiRefNet instead
    shot = camera_imperfections(g, blur, 1750, 620)

    g.group("07 OUTPUT: filtered + unfiltered copies", 2150, 0, 1170, 560, color="#444")
    save = save_node(g, shot, (2180, 60), "local_candidate", "practice run (no GPU)", None,
                     operation="other", project="", provider="other")
    save["title"] = "Save filtered copy"
    g.link(vary, 1, save, "prompt_text", widget=True)
    g.link(vary, 2, save, "seed", widget=True)
    g.link(vary, 4, save, "notes", widget=True)
    g.link(vary, 3, save, "subfolder", widget=True)
    raw_save = save_node(g, composite, (2570, 60), "local_unfiltered", "practice run (no GPU)", None,
                         operation="other", project="", provider="other")
    raw_save["title"] = "Save unfiltered copy"
    g.link(vary, 1, raw_save, "prompt_text", widget=True)
    g.link(vary, 2, raw_save, "seed", widget=True)
    g.link(vary, 4, raw_save, "notes", widget=True)
    g.link(vary, 3, raw_save, "subfolder", widget=True)
    cmp_ = g.node("ImageCompare", (2960, 60), [], size=(340, 480), inputs=[("image_a", "IMAGE"), ("image_b", "IMAGE")], title="Plate / result")
    g.link(scaled, 0, cmp_, "image_a")
    g.link(shot, 0, cmp_, "image_b")

    g.note((3350, 0), """# 09c Practice run (no GPU, no models)

Workflow 09b's clicks, minus Krea 2. Everything here runs on CPU in a few seconds, so you can
learn the moves before the vast.ai meter starts. Queue it as is first, then change things.

## The loop you are rehearsing
1. **01**: the location photo. Painting a mask is optional: with nothing painted, each candidate
   gets its own place. Paint one (right-click -> **Open in MaskEditor**) and every candidate stands
   where you painted -- only the angle still changes. `practice_plate.png` may already carry a
   painted mask; the *Masks* preview and *What each candidate got* show which case you are in.
2. **02**: *Placement & Angle per Seed* turns one queue into **4 candidates**, each with its own
   seed, its own spot in the photo (left to right, nearer or further back) and its own body angle
   (front, three-quarter, profile, back over the shoulder). The angle is drawn in the skeleton
   *and* written into the prompt, so they never disagree. Queue again: new seed, new set.
3. **03**: the masks and skeletons it made. In 09b the skeleton is picture 3 for Krea 2.
4. **04**: the real Prompt Builder. The mandatory "don't look AI" block is appended for you.
5. **05**: here the *skeleton* is pasted into the mask so you can see where she would stand. On
   the pod Krea 2 paints her there. Outside the mask the plate is bit-identical.
6. **06**: background blur, then ISO grain, fringing, falloff, white balance drift.
7. **07**: the 4 filtered candidates land together in `outputs/images/candidates/9b_<seed>/`, with
   clean pre-filter copies under `outputs/images/unfiltered/9b_<seed>/`. Each is logged with its
   label. Then open **Workflow 13** to pick the winner -- also a free, local step. (These practice
   files say `practice_run` in their names; delete their folders when you're done.)

## What is missing on purpose
Krea 2 + your character LoRA, the identity reference LoRA, SDPose's pose extraction from a photo,
and BiRefNet -- all need model files. Nothing else differs from 09b.""", size=(560, 820))
    g.dump("09c_practice_no_gpu.json")


# ---------------------------------------------------------------------------------------------
# Workflow 01: character creation on GPT Image 2.5 (partner API, moderated -- keep it SFW)

GPT_MODEL = "gpt-image-2.5-flare"

IDENTITY_FROM_PICTURES = (
    "Picture 1 is the person; every attached picture shows the same person. Keep their face, bone structure, hair, "
    "skin tone, marks and body exactly as in the pictures. Do not beautify or change the face."
)

# Example photos for LoRA training: (what the shot shows, clothing, setting, camera, light). They
# change expression, framing, outfit, place and light on purpose: a LoRA trained on one t-shirt
# learns the t-shirt.
EXAMPLE_SHOTS = [
    ("close-up of the face, facing the camera, natural smile showing the upper teeth",
     "the same clothes as in picture 1", "small bright kitchen", "eye level, phone camera", "morning daylight from a window"),
    ("from the waist up, facing the camera, arms relaxed",
     "grey hoodie", "city sidewalk, daytime", "eye level, phone camera", "overcast daylight"),
    ("full body from the top of the head to the feet, standing, facing the camera, both feet visible",
     "black t-shirt, blue jeans, white sneakers", "park path", "eye level, 28mm phone camera", "late afternoon sun"),
    ("sitting at a cafe table, looking out of the window, laughing",
     "cream knit sweater", "cafe interior", "slightly above eye level, phone camera", "warm indoor light"),
]
# (sheet, which earlier sheets it sees besides the front portrait). Studio's four, then this pack's.
SHEETS = [
    ("face-turnaround", []),
    ("full-body-360", ["face-turnaround"]),
    ("expression-grid", ["face-turnaround"]),
    ("upper-body", ["face-turnaround"]),
    ("outfit-grid", ["face-turnaround", "full-body-360"]),
    ("pose-grid", ["face-turnaround", "full-body-360"]),
    ("lighting-grid", ["face-turnaround"]),
    ("expression-grid-extra", ["face-turnaround"]),
]
TEMPLATES_DIR = "character-sheet-studio/.claude/skills/character-sheet-studio/templates"

CANDIDATE_PROMPT = (
    "Photo of a fictional adult: DESCRIBE THE CHARACTER HERE - gender, age, build, face shape, eyes, hair, skin, distinguishing marks. "
    "Front-facing head and shoulders portrait, looking straight into the camera, relaxed neutral expression, mouth closed. "
    "Plain white t-shirt. Plain light grey wall. Eye level, 50mm. Soft window light. "
    "Natural unretouched photo with real skin texture."
)


def shot_text(action, clothing, setting, camera, light):
    return (f"Photo of the person: {action}. Clothing: {clothing}. Setting: {setting}. Camera: {camera}. Light: {light}. "
            f"{IDENTITY_FROM_PICTURES} Natural unretouched photo with real skin texture.")


def gpt_image(g: Graph, pos, prompt, refs=(), n=1, size="1024x1536", quality="medium", title=None, prompt_src=None):
    """OpenAI GPT Image 2.5 partner node. refs become picture 1, 2, ... in order."""
    node = g.node("OpenAIGPTImageNodeV2", pos, [prompt, GPT_MODEL, size, 1024, 1024, "auto", quality, n, 0, "fixed"],
                  size=(420, 460), outputs=[("IMAGE", "IMAGE")], title=title)
    node["inputs"] = []
    for i, ref in enumerate(refs, 1):
        node["inputs"].append({"label": f"image_{i}", "name": f"model.images.image_{i}", "shape": 7, "type": "IMAGE", "link": None})
        g.link(ref, 0, node, f"model.images.image_{i}")
    k = len(refs) + 1
    node["inputs"].append({"label": f"image_{k}", "name": f"model.images.image_{k}", "shape": 7, "type": "IMAGE", "link": None})
    node["inputs"].append({"name": "model.mask", "shape": 7, "type": "MASK", "link": None})
    if prompt_src is not None:
        g.link(prompt_src, 0, node, "prompt", widget=True)
    node["color"], node["bgcolor"] = "#432", "#653"  # partner API: billed, moderated
    return node


def gpt_save(g: Graph, image_src, pos, stage, quality, size, n_refs, subfolder=""):
    return save_node(g, image_src, pos, stage, GPT_MODEL, None, operation="generate" if n_refs == 0 else "edit",
                     project="", provider="openai_gpt_image", quality=quality, size=size, n_refs=n_refs, subfolder=subfolder)


CUSTOM_LOCKS = [
    "Soft grey seamless studio background, the same grey behind every view.",
    "Flat even softbox lighting, no hard shadows.",
    "Eye level, identical framing and scale.",
    "Keep the face and identity exactly as in the reference images.",
    "No text, no labels, no watermark.",
]  # = sheet_slicing.DEFAULT_LOCKS / DEFAULT_AVOID
CUSTOM_AVOID = [
    "No borders and no drop shadows between the panels.",
    "Leave a clear strip of empty grey background between neighbouring panels; no figure overlaps another panel.",
    "Do not crop the person at a panel edge.",
]


def slicer(g: Graph, sheet_img, template_src, pos):
    cut = g.node("AIInfluencerSliceSheet", pos, ["", "", "plain grey studio background", 0.09, 0.05], size=(400, 380),
                 inputs=[("sheet_image", "IMAGE"), ("template", "STRING")], outputs=[("crops", "IMAGE"), ("captions", "STRING")],
                 title="Slice into one picture per view")
    for o in cut["outputs"]:
        o["shape"] = 6  # list output
    g.link(sheet_img, 0, cut, "sheet_image")
    g.link(template_src, 0, cut, "template")
    return cut


def build_01():
    g = Graph()
    g.group("01 FRONT PORTRAIT CANDIDATES — GPT Image 2.5 (queue this first)", 0, 0, 1320, 1000, color="#b58b2a")
    candidates = gpt_image(g, (30, 60), CANDIDATE_PROMPT, n=4, title="WHO SHE IS + 4 candidates per call (change the seed for 4 new ones)")
    gpt_save(g, candidates, (480, 60), "character_candidate", "medium", "1024x1536", 0)
    own = g.node("LoadImage", (30, 580), ["example.png", "image"], size=(400, 380), outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                 title="OR your own front photo (Ctrl+B to switch it on)", mode=BYPASS)
    pick = g.node("AIInfluencerPickCandidate", (480, 580), [0], size=(360, 110),
                  inputs=[("candidates", "IMAGE"), ("own_image", "IMAGE")], outputs=[("image", "IMAGE")],
                  title="PICK: 0 = just show them. Then 1-4 and queue again")
    g.link(candidates, 0, pick, "candidates")
    g.link(own, 0, pick, "own_image")
    front = g.node("PreviewImage", (880, 580), size=(400, 380), inputs=[("images", "IMAGE")], title="Picture 1 of every request below")
    g.link(pick, 0, front, "images")

    col_w = 880
    row_h = 900
    g.group("02 CHARACTER SHEETS (16:9, 4K) -> sliced into one picture per view, captioned, for the LoRA", 0, 1040, 4 * col_w + 20, 2 * row_h + 20, color="#a1309b")
    sheets = {}
    for i, (sheet, sees) in enumerate(SHEETS):
        x, y = 30 + (i % 4) * col_w, 1100 + (i // 4) * row_h
        sp = g.node("AIInfluencerSheetPrompt", (x, y), [sheet, TEMPLATES_DIR], size=(420, 110), outputs=[("prompt", "STRING")], title=sheet)
        refs = [pick, *(sheets[n] for n in sees)]
        img = gpt_image(g, (x, y + 140), "", refs=refs, size="3840x2160", title=f"{sheet} ({len(refs)} picture{'s' * (len(refs) > 1)} of her)", prompt_src=sp)
        sheets[sheet] = img
        gpt_save(g, img, (x + 450, y), "character_sheet", "medium", "3840x2160", len(refs), subfolder=sheet)
        slicer(g, img, sp, (x + 450, y + 460))

    # your own sheet: muted until you switch it on (Ctrl+M on the group's nodes), so it costs nothing by default
    x, y = 30, 1100 + 2 * row_h
    first_custom = len(g.nodes)
    g.group("02b YOUR OWN SHEET (muted: select its nodes, Ctrl+M to switch on)", 0, y - 20, col_w + 20, row_h - 40, color="#a1309b")
    custom = g.node("AIInfluencerCustomSheet", (x, y + 40), [
        "Outfit Variations",
        "seen straight from the front, standing, the whole person visible from the top of the head to both feet, "
        "wearing the same clothes as in the reference images, neutral expression",
        "\n".join(f"seen straight from the front, standing, the whole person visible, wearing {o}" for o in (
            "a red raincoat and rubber boots", "a ski jacket and ski pants",
            "a white linen shirt and beige shorts", "a black leather jacket and dark jeans")),
        "\n".join(CUSTOM_LOCKS), "\n".join(CUSTOM_AVOID), ""], size=(420, 560), outputs=[("template", "STRING")], title="Custom Character Sheet")
    img = gpt_image(g, (x + 450, y + 40), "", refs=[pick, sheets["face-turnaround"], sheets["full-body-360"]], size="3840x2160",
                    title="your sheet (3 pictures of the person)", prompt_src=custom)
    gpt_save(g, img, (x + 450, y + 520), "character_sheet", "medium", "3840x2160", 3, subfolder="custom")
    slicer(g, img, custom, (x + 450, y + 560))
    for n in g.nodes[first_custom:]:
        n["mode"] = NEVER

    g.group("03 EXAMPLE PHOTOS FOR THE LORA — each sees the front portrait + face and full-body sheets", 0, 1100 + 3 * row_h + 20, 4 * col_w + 20, 620, color="#3f789e")
    for i, shot in enumerate(EXAMPLE_SHOTS):
        x, y = 30 + i * col_w, 1100 + 3 * row_h + 80
        refs = [pick, sheets["face-turnaround"], sheets["full-body-360"]]
        img = gpt_image(g, (x, y), shot_text(*shot), refs=refs, title=f"Example {i + 1}")
        gpt_save(g, img, (x + 450, y), "character_canonical", "medium", "1024x1536", len(refs), subfolder="reference_set")

    g.note((1360, 0), """# 01 Character creation (GPT Image 2.5)

Makes character sheets and a few example photos to train the character's LoRA on. From then on the
**local** workflows (08-12) make everything else. Everything below can be changed on the canvas --
see **Make it yours** at the end.

**1. Queue.** Describe the character in the first node (the only place they are ever described in
words): it makes **4 front portraits in one call** (`n`). *PICK* is 0, so everything after it is
skipped and you only see the candidates (saved to `outputs/images/candidates/`).

**2. Pick.** Set *PICK* to the one you want and queue again. The candidates call is **not**
repeated: its inputs didn't change, so ComfyUI reuses it (that's why its seed is `fixed` -- GPT
ignores the seed, it only tells ComfyUI "new call"; change it by hand for new candidates). Already
have a front photo? Ctrl+B *OR your own front photo*, load it, and it wins over the candidates.

**3. The rest runs by itself**, each request carrying the pictures that pin the character down:

| Sheet | Shows | Pictures it sees |
|---|---|---|
| face-turnaround | 7 head angles | front portrait |
| full-body-360 | 7 body angles | front + face-turnaround |
| expression-grid | 9 expressions | front + face-turnaround |
| upper-body | 7 torso angles | front + face-turnaround |
| outfit-grid | 7 outfits | front + face + full-body sheets |
| pose-grid | 9 poses | front + face + full-body sheets |
| lighting-grid | 7 lights | front + face-turnaround |
| expression-grid-extra | 9 more expressions | front + face-turnaround |
| 4 example photos | outfit, place and light changes | front + face + full-body sheets |

The first four sheets are Character Sheet Studio's templates, the other four ship with the node
pack; all are sent unchanged. After the first node nothing describes the person in words -- the
studio's rule: extra words compete with the pictures and cause drift.

**Slicing.** A LoRA trained on whole sheets learns to draw sheets. *Slice into one picture per
view* cuts each sheet into its panels (it finds each figure on the grey background, so a hand over
a panel line stays whole) and writes one caption per crop: the trigger word from character.json
plus that panel's own wording ("lit by a warm tungsten table lamp", "wearing a grey sports
hoodie"...), so the LoRA learns the light, outfit or pose as a variable, not as the person. ~60
crops per character, in `outputs/images/sheet_crops/<sheet>_<time>/`. Grid panels are about
640x1080. Look at them before copying into `input/ai_influencer/<project>/dataset/` -- the captions
assume GPT kept the panel order. Sheets made earlier: `tools/slice_sheets.py`.

**No blur, no camera pass here, on purpose:** these are training and reference pictures. Grain or a
blurred background in the dataset gets learned by the LoRA.

**Cost at medium quality** (see each node's price badge): 4 candidates ≈ $0.06, 8 sheets ≈ $0.55,
4 examples ≈ $0.24 -- about **$0.85 per character**. Slicing is free (local, CPU).

## Make it yours
- **Which sheets:** any *Character Sheet Prompt* can be switched to another sheet in its dropdown.
  Don't want one? Select its GPT node and Ctrl+M (mute): its save and slicer are skipped too.
- **Your own sheet:** group 02b. Type a title, the large view, and **one panel per line** (outfits,
  poses, lights, props, hairstyles... describe what changes, not the person), then select the
  group's nodes and Ctrl+M to switch it on. Fill *save_as* to keep it as a template: it appears in
  every sheet dropdown (press R) and in `user/ai_influencer/sheet_templates/`. A file there with the
  same name as a built-in sheet replaces it -- that's how you edit the built-in outfit list.
- **Which pictures a sheet sees:** rewire its `image_1..` inputs (up to 16; each one is billed).
- **Quality, size, count:** on each GPT node (`quality`, `size`, `n`). Sheets at `2048x1152` cost
  ~$0.02 less each, but the sliced panels get half as big.
- **Example photos:** edit their prompts; copy an *Example* + its Save node for more.
- **Captions:** *caption_prefix* (default: the trigger word), *caption_suffix*, and *threshold* /
  *padding* for how tightly figures are cut, on each slicer.

**Account risk.** This is a partner API: proxied through Comfy, billed to your account, moderated
by OpenAI. Keep it to ordinary clothed pictures of a fictional adult. Anything else belongs on the
local pipeline (README, "Responsible use and account risk").""", size=(780, 1100))
    g.dump("01_character_creation.json")


def build_13():
    g = Graph()
    g.group("PICK THE WINNER (runs on this PC, no GPU, never held for the pod)", 0, 0, 900, 700, color="#8A8")
    g.node("AIInfluencerPickWinner", (30, 60), ["(no candidate folders yet)", 0], size=(820, 600),
           outputs=[("winner", "IMAGE")], title="Pick Winner")
    g.note((930, 0), """# 13 Pick the winner

Workflow 09b saves the candidates of each queue into their own folder,
`outputs/images/candidates/9b_<seed>/` -- also when the queue ran on the pod (the folders are
downloaded to this PC at the same path). A batch of 10 held jobs is 10 folders.

1. Press **R** on the canvas so the folder list is current, pick a **folder** (newest first).
2. Queue with **winner = 0**: all its candidates show in the node, in order (#1 first). The
   label of each -- seed, where she stands, which angle -- is in `logs/generations.jsonl`.
3. Set **winner** to the number you want, queue: that file is **copied** (not re-saved: same
   pixels, same embedded workflow) to `outputs/images/final/9b_<seed>_winner<N>.png`.
4. Next folder.

The *Pod queue*'s "Hold for the pod" never holds this workflow: it only reads files on this PC.
Dragging a candidate PNG onto the canvas restores its whole 09b setup, seed included.""", size=(520, 460))
    g.dump("13_pick_winner.json")


WAN_UNET = "wan_animate_2_distill_int8_convrot.safetensors"
WAN_TE = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_CLIP_VISION = "clip_vision_h.safetensors"
WAN_VAE = "Wan2_1_VAE_bf16.safetensors"
for _name, _folder in ((WAN_UNET, "diffusion_models"), (WAN_TE, "text_encoders"), (WAN_CLIP_VISION, "clip_vision"), (WAN_VAE, "vae")):
    MODEL_URLS[_name] = (_folder, f"https://huggingface.co/Comfy-Org/Wan-Animate-2/resolve/main/{_folder}/{_name}")
CHOOSE_VIDEO = "ai_influencer_choose_a_video.mp4"  # input_listing.VIDEO_PLACEHOLDER
# AIInfluencerMotionPrompt's own defaults (nodes_motion.py); kept here so this script needs no ComfyUI
MOTION_LOOK = ("Photoreal vertical social clip of the person from the reference image: same face, same hair, same outfit, "
               "same light and location. Keep their identity, skin texture and clothing identical to the reference image.")
MOTION_TEXT = ("The person moves exactly like the driving video, matching the timing of every step, turn, arm movement, "
               "hand gesture and facial expression.")
MOTION_AVOID = ("oversaturated, overexposed, static, blurry details, subtitles, text, watermark, painting, illustration, "
                "cartoon, airbrushed plastic skin, HDR, oversharpened, worst quality, low quality, JPEG artifacts, deformed, "
                "extra fingers, fused fingers, poorly drawn hands, poorly drawn face, extra limbs, warped face, flickering, "
                "scene cut")


def build_14():
    """Motion transfer: the person in a still copies the movement of a driving video (Wan Animate 2, distilled).
    Core ComfyUI's WanAnimate2ToVideo takes the raw video as pose_video, so the pose/face detectors, segmentation
    and ~8 custom node packs of the "Motion Control" workflow this is based on aren't needed."""
    g = Graph()
    g.group("01 WHO + WHAT MOVEMENT", 0, 0, 1180, 1100)
    still = g.node("LoadImage", (30, 60), [CHOOSE_PICTURE, "image"], size=(340, 420),
                   outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], title="The person (a final still of the character)")
    video = g.node("LoadVideo", (400, 60), [CHOOSE_VIDEO, "image"], size=(340, 420),
                   outputs=[("VIDEO", "VIDEO")], title="Driving video (the movement to copy)")
    drive = g.node("AIInfluencerDrivingVideo", (770, 60), [0.0, 5.0, 30.0, 832], size=(380, 420),
                   inputs=[("video", "VIDEO")],
                   outputs=[("frames", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT"), ("length", "INT"), ("width", "INT"),
                            ("height", "INT"), ("long_clip", "BOOLEAN"), ("duration", "STRING")],
                   title="Which part, how fast, how big")
    g.link(video, 0, drive, "video")
    words = g.node("AIInfluencerMotionPrompt", (30, 520), [MOTION_LOOK, MOTION_TEXT, MOTION_AVOID], size=(1120, 540),
                   outputs=[("positive", "STRING"), ("motion", "STRING"), ("negative", "STRING")],
                   title="Motion description")

    g.group("02 WAN ANIMATE 2 MODELS", 0, 1140, 620, 820, color="#88A")
    unet = g.node("UNETLoader", (30, 1200), [WAN_UNET, "default"], size=(560, 110), outputs=[("MODEL", "MODEL")],
                  title="Wan Animate 2 (distilled, 10 steps)")
    extra = g.node("LoraLoaderModelOnly", (30, 1360), ["", 1.0], size=(560, 130), inputs=[("model", "MODEL")],
                   outputs=[("MODEL", "MODEL")], title="Optional extra LoRA - Wan 2.x character / concept fine-tune (Ctrl+B to enable)",
                   mode=BYPASS)
    g.link(unet, 0, extra, "model")
    clip = g.node("CLIPLoader", (30, 1540), [WAN_TE, "wan", "default"], size=(560, 130), outputs=[("CLIP", "CLIP")],
                  title="Text encoder (umT5)")
    vision = g.node("CLIPVisionLoader", (30, 1720), [WAN_CLIP_VISION], size=(560, 80),
                    outputs=[("CLIP_VISION", "CLIP_VISION")], title="CLIP vision")
    vae = g.node("VAELoader", (30, 1850), [WAN_VAE], size=(560, 80), outputs=[("VAE", "VAE")], title="Wan VAE")

    g.group("03 CONDITIONING", 650, 1140, 940, 820, color="#3f789e")
    ref = g.node("ResizeImageMaskNode", (680, 1200), ["scale dimensions", 832, 480, "center", "area"], size=(320, 180),
                 inputs=[("input", "IMAGE,MASK")], outputs=[("resized", "IMAGE")], title="Still at the video's size")
    g.link(still, 0, ref, "input")
    g.link(drive, 4, ref, "resize_type.width", widget=True)
    g.link(drive, 5, ref, "resize_type.height", widget=True)
    first = g.node("ImageFromBatch", (680, 1430), [0, 1], size=(320, 100), inputs=[("image", "IMAGE")],
                   outputs=[("IMAGE", "IMAGE")], title="Video's first frame")
    g.link(drive, 0, first, "image")
    seen = []
    for img, y, title in ((ref, 1580, "CLIP vision: the still"), (first, 1730, "CLIP vision: the video")):
        enc = g.node("CLIPVisionEncode", (680, y), ["none"], size=(320, 100),
                     inputs=[("clip_vision", "CLIP_VISION"), ("image", "IMAGE")],
                     outputs=[("CLIP_VISION_OUTPUT", "CLIP_VISION_OUTPUT")], title=title)
        g.link(vision, 0, enc, "clip_vision")
        g.link(img, 0, enc, "image")
        seen.append(enc)
    conds = []
    for slot, y, title in ((0, 1200, "Positive (the person)"), (1, 1330, "Motion (pose branch)"), (2, 1460, "Negative")):
        enc = g.node("CLIPTextEncode", (1030, y), [""], size=(300, 100), inputs=[("clip", "CLIP")],
                     outputs=[("CONDITIONING", "CONDITIONING")], title=title)
        g.link(clip, 0, enc, "clip")
        g.link(words, slot, enc, "text", widget=True)
        conds.append(enc)
    animate = g.node("WanAnimate2ToVideo", (1030, 1600), [832, 480, 81, 1, 0, 1.0, 0.0, 1.0, 1.0], size=(520, 340),
                     inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("vae", "VAE"),
                             ("reference_image", "IMAGE"), ("pose_video", "IMAGE"), ("clip_vision_output", "CLIP_VISION_OUTPUT"),
                             ("positive_pose", "CONDITIONING"), ("clip_vision_output_pose", "CLIP_VISION_OUTPUT")],
                     outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT"),
                              ("trim_latent", "INT"), ("trim_image", "INT"), ("video_frame_offset", "INT")],
                     title="Wan Animate 2 (pose_strength / reference_image_strength live here)")
    g.link(conds[0], 0, animate, "positive")
    g.link(conds[2], 0, animate, "negative")
    g.link(vae, 0, animate, "vae")
    g.link(ref, 0, animate, "reference_image")
    g.link(drive, 0, animate, "pose_video")
    g.link(seen[0], 0, animate, "clip_vision_output")
    g.link(conds[1], 0, animate, "positive_pose")
    g.link(seen[1], 0, animate, "clip_vision_output_pose")
    for slot, name in ((4, "width"), (5, "height"), (3, "length")):
        g.link(drive, slot, animate, name, widget=True)

    g.group("04 SAMPLE", 1620, 1140, 760, 820, color="#8A8")
    windows = g.node("ContextWindowsManual", (1650, 1200), [21, 8, "standard_static", 1, False, "pyramid", 2, True, "0", False, "", True],
                     size=(340, 380), inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL")],
                     title="Context windows (used only for clips over 81 frames)")
    g.link(extra, 0, windows, "model")
    switch = g.node("ComfySwitchNode", (1650, 1630), [False], size=(340, 100),
                    inputs=[("on_false", "*"), ("on_true", "*")], outputs=[("output", "*")], title="Long clip? windows on")
    g.link(extra, 0, switch, "on_false")
    g.link(windows, 0, switch, "on_true")
    g.link(drive, 6, switch, "switch", widget=True)
    cache = g.node("WanAnimate2Cache", (1650, 1780), ["cpu", "int8"], size=(340, 110), inputs=[("model", "MODEL")],
                   outputs=[("MODEL", "MODEL")], title="Pose cache (about 2x faster, uses RAM)")
    g.link(switch, 0, cache, "model")
    shift = g.node("ModelSamplingSD3", (2020, 1200), [5.0], size=(330, 80), inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL")])
    g.link(cache, 0, shift, "model")
    sampler = g.node("KSamplerSelect", (2020, 1320), ["lcm"], size=(330, 80), outputs=[("SAMPLER", "SAMPLER")])
    sched = g.node("BasicScheduler", (2020, 1440), ["simple", 10, 1.0], size=(330, 130), inputs=[("model", "MODEL")],
                   outputs=[("SIGMAS", "SIGMAS")], title="Steps (10 for the distilled model)")
    g.link(cache, 0, sched, "model")
    sample = g.node("SamplerCustom", (2020, 1610), [True, 0, "randomize", 1.0], size=(330, 300),
                    inputs=[("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                            ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
                    outputs=[("output", "LATENT"), ("denoised_output", "LATENT")], title="Sample (seed here)")
    g.link(shift, 0, sample, "model")
    g.link(animate, 0, sample, "positive")
    g.link(animate, 1, sample, "negative")
    g.link(sampler, 0, sample, "sampler")
    g.link(sched, 0, sample, "sigmas")
    g.link(animate, 2, sample, "latent_image")

    g.group("05 DECODE + CAMERA PASS + SAVE", 2410, 1140, 1240, 820, color="#a1309b")
    trim = g.node("TrimVideoLatent", (2440, 1200), [0], size=(280, 80), inputs=[("samples", "LATENT")],
                  outputs=[("LATENT", "LATENT")], title="Drop the reference frame")
    g.link(sample, 0, trim, "samples")
    g.link(animate, 3, trim, "trim_amount", widget=True)
    decode = g.node("VAEDecode", (2440, 1320), [], size=(280, 70), inputs=[("samples", "LATENT"), ("vae", "VAE")],
                    outputs=[("IMAGE", "IMAGE")], title="Decode video")
    g.link(trim, 0, decode, "samples")
    g.link(vae, 0, decode, "vae")
    cam = g.node("AIInfluencerCameraImperfections", (2440, 1430), [0, "randomize", 0.025, 0.1, 0.35, 0.05, 0.15, 0.3, True],
                 size=(280, 300), inputs=[("image", "IMAGE")], outputs=[("image", "IMAGE")], title="Camera pass on every frame")
    g.link(decode, 0, cam, "image")
    mux = g.node("CreateVideo", (2750, 1200), [30.0, "auto", "sRGB", "none"], size=(280, 160),
                 inputs=[("images", "IMAGE"), ("audio", "AUDIO")], outputs=[("VIDEO", "VIDEO")],
                 title="Mux with the driving video's sound")
    g.link(cam, 0, mux, "images")
    g.link(drive, 1, mux, "audio")
    g.link(drive, 2, mux, "fps", widget=True)
    save = g.node("AIInfluencerSaveVideo", (3060, 1200), ["", "wan_animate_video", "wan_animate_local", "Wan Animate 2", "",
                                                         "480p", "5", -1, "fixed", "", ""],
                  size=(400, 420), inputs=[("video", "VIDEO")], outputs=[("VIDEO", "VIDEO"), ("output_path", "STRING")],
                  title="Save & log the clip")
    g.link(mux, 0, save, "video")
    g.link(words, 1, save, "motion_prompt", widget=True)
    g.link(drive, 7, save, "duration_seconds", widget=True)

    # side by side with the driving video, to judge how well the movement was followed (off until you need it)
    side = g.node("ImageStitch", (2750, 1430), ["left", True, 0, "white"], size=(280, 160),
                  inputs=[("image1", "IMAGE"), ("image2", "IMAGE")], outputs=[("IMAGE", "IMAGE")],
                  title="Side by side: driving | result (Ctrl+M to switch on)", mode=NEVER)
    g.link(decode, 0, side, "image1")
    g.link(drive, 0, side, "image2")
    side_mux = g.node("CreateVideo", (2750, 1630), [30.0, "auto", "sRGB", "none"], size=(280, 160),
                      inputs=[("images", "IMAGE"), ("audio", "AUDIO")], outputs=[("VIDEO", "VIDEO")], title="Comparison video", mode=NEVER)
    g.link(side, 0, side_mux, "images")
    g.link(drive, 1, side_mux, "audio")
    g.link(drive, 2, side_mux, "fps", widget=True)
    side_save = g.node("AIInfluencerSaveVideo", (3060, 1660), ["", "misc_video", "wan_animate_local", "Wan Animate 2 comparison", "",
                                                              "480p", "5", -1, "fixed", "", "side by side with the driving video"],
                       size=(400, 280), inputs=[("video", "VIDEO")], outputs=[("VIDEO", "VIDEO"), ("output_path", "STRING")],
                       title="Save the comparison", mode=NEVER)
    g.link(side_mux, 0, side_save, "video")

    g.note((1220, 0), """# 14 Motion transfer (Wan Animate 2, open weights)

The person in a **still** copies the **movement of a video**: a dance, a walk to the camera, a
gesture, a facial expression. The look, outfit and location come from the still; only the motion
comes from the video. Based on the "Motion Control" workflow (Wan 2.2 Animate through
WanVideoWrapper + 8 custom node packs), rebuilt on core ComfyUI's **Wan Animate 2**, which reads
the driving video directly: no pose detector, face crop or segmentation to set up.

**Runs on a pod**, not on the local GTX 1060: tick *Hold for the pod* and press Run; the Pod queue
adds `--with-animate` (24 GB card, ~25 GB of models) and uploads the still *and* the video.

1. **The person**: a final still (Workflow 08/09/09b/10 output). It is cropped to the video's
   shape, so pick a still framed like the video (full body for a dance, waist up for talking).
2. **Driving video**: click *upload*. One person, steady camera, the whole body in frame for
   body motion. Its **sound is kept** on the result (none if the video has none).
3. **Which part, how fast, how big**: `start_seconds` / `duration_seconds` pick the span (0 = to
   the end); `fps` 30 is what the model is trained on; `long_side` 832 is 480p class (fast), 1280
   is 720p class. The node prints the frame count it worked out. Up to 81 frames (2.7 s at 30 fps)
   is one pass; longer clips switch on the context windows automatically and take
   proportionally longer.
4. **Motion description**: `look` keeps the identity, `motion` names the movement for the pose
   branch. The blurred background and handheld-phone wording is always added (not editable here,
   same as Workflow 11).
5. Run. The clip gets the camera pass on every frame and is saved to
   `outputs/videos/` and logged.

**Fine-tuning** (on the *Wan Animate 2* node): `pose_strength` below 1 loosens the choreography,
`pose_end_percent` 0.7 keeps the movement but frees fine detail; `reference_image_strength` above
1 holds the face tighter if it drifts. Steps/sampler are set for the distilled model: 10 steps,
`lcm`, cfg 1. The seed is on *Sample*.

**Compare**: select the three muted nodes in 05 (Side by side, Comparison video, Save the
comparison) and press Ctrl+M to save a driving | result video next to the clip.

**Use** a fictional character as the still. The video only lends its movement, but its sound is
copied onto the result, so mind the music's and voice's rights where you post it.""", size=(900, 1100))
    g.dump("14_wan_animate_motion_transfer.json")


if __name__ == "__main__":
    build_07()
    build_08()
    build_09()
    build_10()
    build_12()
    build_09c()
    build_01()
    build_13()
    build_14()

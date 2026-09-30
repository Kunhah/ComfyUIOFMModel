"""Character identity + prompt-assembly nodes.

AIInfluencerLoadCharacter reads output/projects/<project>/character.json (created on first use)
and exposes its non-secret fields as strings so they can be prepended to every prompt, which is
the cheapest and most reliable way to keep one character's description consistent across
hundreds of generations without a trainable LoRA (none of Krea 2 / GPT Image 2.5 / Seedance 2.5's
official cloud APIs expose one).

AIInfluencerPromptBuilder assembles the labelled prompt sections from the architecture diagram
(scene / action / clothing / environment / camera / lighting / realism / identity) into one
string, so each concern stays editable on its own node input instead of one large prompt blob.
It always appends MANDATORY_IMPERFECTIONS: the wording that keeps the model from producing the
symmetrical, evenly lit, retouched look that reads as AI at a glance. It is not an input, so it
cannot be edited away in one workflow and drift from the others; the grain and lens artifacts it
asks for are then applied for real by the Camera Imperfections node (nodes_camera.py).
"""

from __future__ import annotations

import json

from comfy_api.latest import IO, ComfyExtension
from typing_extensions import override

from . import logging_util

MANDATORY_IMPERFECTIONS = (
    "mundane candid photography, unretouched amateur snapshot: slightly off-center framing, "
    "horizon a little tilted, uneven ambient light with some underexposed areas, natural skin "
    "texture with visible pores and small asymmetries, a few flyaway hairs, clothing creased and "
    "wrinkled, relaxed unposed body language, in-between facial expression, realistic hand "
    "anatomy with correct finger count, slight lens softness and a touch of motion blur, visible "
    "ISO grain, subtle chromatic aberration, slightly off white balance, ordinary phone camera "
    "look; avoid airbrushed skin, HDR glow, oversharpening and studio lighting in casual scenes"
)

DEFAULT_IDENTITY_INSTRUCTIONS = (
    "preserve this exact person's facial identity, bone structure, and recognizable features "
    "from the reference; do not beautify or change the face"
)


class AIInfluencerLoadCharacter(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerLoadCharacter",
            display_name="Load Character (AI Influencer)",
            category="ai_influencer",
            description=(
                "Reads output/projects/<project>/character.json (created automatically on first "
                "use) and outputs its trigger/description as strings to prepend to prompts, for "
                "consistent identity wording across generations."
            ),
            inputs=[
                IO.String.Input("project", default="", tooltip="Project/character name. Empty uses AI_INFLUENCER_DEFAULT_PROJECT or 'default'."),
            ],
            outputs=[
                IO.String.Output(display_name="name"),
                IO.String.Output(display_name="trigger"),
                IO.String.Output(display_name="identity_prompt"),
                IO.String.Output(display_name="lora_url"),
                IO.String.Output(display_name="character_json"),
            ],
        )

    @classmethod
    def execute(cls, project: str) -> IO.NodeOutput:
        project = logging_util.sanitize_project_name(project)
        logging_util.ensure_project(project)
        with open(logging_util.character_json_path(project), encoding="utf-8") as f:
            data = json.load(f)

        name = data.get("name", project)
        trigger = data.get("trigger", "")
        permanent_features = data.get("permanent_features") or []
        identity_parts = [p for p in [trigger, *permanent_features] if p]
        identity_prompt = ", ".join(identity_parts)
        lora_url = data.get("lora_url", "")

        return IO.NodeOutput(name, trigger, identity_prompt, lora_url, json.dumps(data, indent=2))


class AIInfluencerPromptBuilder(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerPromptBuilder",
            display_name="Prompt Builder (AI Influencer)",
            category="ai_influencer",
            description=(
                "Assembles a text-to-image prompt from labelled sections (identity, scene, "
                "action, clothing, environment, camera, lighting, realism) into one string. "
                "Leave any field blank to skip it. A fixed block asking for candid, unretouched, "
                "grainy photography is always appended."
            ),
            inputs=[
                IO.String.Input("identity", default="", multiline=True, optional=True, tooltip="Usually the identity_prompt output of Load Character."),
                IO.String.Input("scene", default="", multiline=True, optional=True),
                IO.String.Input("subject_action", default="", multiline=True, optional=True),
                IO.String.Input("clothing", default="", multiline=True, optional=True),
                IO.String.Input("environment", default="", multiline=True, optional=True),
                IO.String.Input("camera_and_lens", default="", multiline=True, optional=True),
                IO.String.Input("lighting", default="", multiline=True, optional=True),
                IO.String.Input("realism_instructions", default="", multiline=True, optional=True, tooltip="Extra realism wording for this shot. The mandatory imperfection block is appended after it either way."),
                IO.String.Input("identity_preservation_instructions", default=DEFAULT_IDENTITY_INSTRUCTIONS, multiline=True, optional=True),
                IO.String.Input("extra", default="", multiline=True, optional=True),
            ],
            outputs=[IO.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(
        cls,
        identity: str = "",
        scene: str = "",
        subject_action: str = "",
        clothing: str = "",
        environment: str = "",
        camera_and_lens: str = "",
        lighting: str = "",
        realism_instructions: str = "",
        identity_preservation_instructions: str = "",
        extra: str = "",
    ) -> IO.NodeOutput:
        parts = [
            identity,
            scene,
            subject_action,
            clothing,
            environment,
            camera_and_lens,
            lighting,
            realism_instructions,
            MANDATORY_IMPERFECTIONS,
            identity_preservation_instructions,
            extra,
        ]
        prompt = ", ".join(p.strip() for p in parts if p and p.strip())
        return IO.NodeOutput(prompt)


class AIInfluencerToolkitCharacterExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [AIInfluencerLoadCharacter, AIInfluencerPromptBuilder]

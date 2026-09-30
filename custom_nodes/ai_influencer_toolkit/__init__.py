"""AI Influencer Toolkit — small, dependency-free ComfyUI node pack that supports the
Krea 2 / GPT Image 2.5 / Seedance 2.5 / OpenRouter cloud-API pipeline and the local
open-weights Krea 2 + character LoRA pipeline documented in README_AI_INFLUENCER.md. It does
not call any cloud provider directly: it manages local project folders, prompt assembly, the
local generations.jsonl log, and LoRA checkpoint testing/selection.
"""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO

from .nodes_background import AIInfluencerBlurBackground
from .nodes_automask import AIInfluencerAutoMask
from .nodes_camera import AIInfluencerCameraImperfections
from .nodes_character import AIInfluencerLoadCharacter, AIInfluencerPromptBuilder
from .nodes_local_lora import AIInfluencerApplyCharacterLora, AIInfluencerLoraCheckpointTester
from .nodes_pick import AIInfluencerPickCandidate, AIInfluencerPickWinner
from .nodes_save import AIInfluencerSaveImage, AIInfluencerSaveVideo
from .nodes_sheet import AIInfluencerCustomSheet, AIInfluencerSheetPrompt, AIInfluencerSliceSheet
from .nodes_variation import AIInfluencerPlacementVariations

try:
    from .nodes_fal_lora import AIInfluencerFalLoraImage, AIInfluencerTrainCharacterLoRA
    _FAL_LORA_NODES = [AIInfluencerTrainCharacterLoRA, AIInfluencerFalLoraImage]
except ImportError:
    # fal-client isn't installed; the rest of the toolkit (Krea/GPT Image/Seedance/OpenRouter
    # pipeline, which needs no extra dependency) still loads fine without the optional LoRA nodes.
    _FAL_LORA_NODES = []


try:
    from . import pod_queue  # noqa: F401  (registers the /ai_influencer/pod_queue routes)
except Exception as e:  # never let the pod queue take the nodes down with it
    import logging

    logging.warning("AI Influencer pod queue disabled: %s", e)

try:
    from . import model_routes  # noqa: F401  (registers the /ai_influencer/models routes)
except Exception as e:
    import logging

    logging.warning("AI Influencer Models window disabled: %s", e)

try:
    from . import input_listing

    input_listing.install()  # LoadImage dropdown includes input/ai_influencer/** subfolders
    input_listing.make_placeholder()  # picked by web/input_helper.js for pictures this computer doesn't have
    input_listing.install_routes()
except Exception as e:
    import logging

    logging.warning("AI Influencer picture inputs helper disabled: %s", e)

WEB_DIRECTORY = "./web"  # web/pod_queue.js: the "Pod queue" panel; web/model_helper.js: the "Models" window;
                         # web/input_helper.js: pictures a workflow names that aren't on this computer


class AIInfluencerToolkitExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            AIInfluencerLoadCharacter,
            AIInfluencerPromptBuilder,
            AIInfluencerSaveImage,
            AIInfluencerSaveVideo,
            AIInfluencerLoraCheckpointTester,
            AIInfluencerApplyCharacterLora,
            AIInfluencerBlurBackground,
            AIInfluencerCameraImperfections,
            AIInfluencerAutoMask,
            AIInfluencerPlacementVariations,
            AIInfluencerPickCandidate,
            AIInfluencerPickWinner,
            AIInfluencerSheetPrompt,
            AIInfluencerSliceSheet,
            AIInfluencerCustomSheet,
            *_FAL_LORA_NODES,
        ]


async def comfy_entrypoint() -> AIInfluencerToolkitExtension:
    return AIInfluencerToolkitExtension()

"""Video description for MiniMax H3 (Workflow 11), assembled from short fields.

H3 reads one prose block: a look/continuity sentence, a timestamped beat list, the spoken lines, an
`Audio:` line and a negative tail (see the "How to write the description" note in Workflow 11).
Writing that by hand means re-typing every timestamp whenever the duration changes. This node takes
one beat per line and spreads them evenly over the clip, so the timeline always matches the frame
count it hands to the model.

It always adds MANDATORY_VIDEO_LOOK: the out-of-focus background and the handheld-phone
imperfections every clip in this project needs. Like MANDATORY_IMPERFECTIONS for stills
(nodes_character.py) it is not an input, so it cannot be edited away in one workflow; the grain is
then applied for real to every frame by the Camera Imperfections node.
"""

from __future__ import annotations

import re

from comfy_api.latest import IO, ui

FPS = 24  # H3's native rate; its frame counts sit on a 17k+5 grid (see comfy_extras/nodes_minimax_h3.py)
WORDS_PER_SECOND = 2.5  # unhurried speech; more than this and H3 rushes the lip sync

MANDATORY_VIDEO_LOOK = (
    "The background stays soft and far out of focus the whole time (shallow depth of field, creamy "
    "bokeh); only the person is sharp. Ordinary handheld phone footage, not a film: slight natural "
    "camera shake, auto-exposure and white balance that drift a little, visible sensor noise, a "
    "touch of motion blur on quick movements, natural skin texture with pores and small "
    "asymmetries, a few flyaway hairs, clothing that creases as the person moves, realistic hands "
    "with the correct number of fingers. Avoid airbrushed skin, HDR glow, oversharpening, "
    "cinematic color grading and studio lighting."
)

_TIMED = re.compile(r"^\s*\[\s*\d")


def frame_count(seconds: float) -> int:
    """Seconds -> frames at 24 fps, snapped up to H3's 17k+5 grid (5 s -> 124)."""
    n = max(5, round(seconds * FPS))
    return n + (5 - n % 17) % 17


def _lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def timeline(beats: list[str], seconds: float) -> list[str]:
    """Evenly timed beats. A line that already starts with "[1.5s" keeps its own timing."""
    step = seconds / max(1, len(beats))
    out = []
    for i, beat in enumerate(beats):
        out.append(beat if _TIMED.match(beat) else f"[{i * step:.1f}s-{(i + 1) * step:.1f}s] {beat}")
    return out


def build(look: str, beats: str, dialogue: str, audio: str, avoid: str, seconds: float) -> tuple[str, list[str]]:
    """(the description, warnings)."""
    warnings = []
    parts = [" ".join(look.split()), MANDATORY_VIDEO_LOOK]
    beat_lines = _lines(beats)
    if beat_lines:
        parts.append("Timeline:\n" + "\n".join(timeline(beat_lines, seconds)))
    else:
        warnings.append("no beats: describe at least one movement, or the clip barely moves")
    spoken = [line.strip('"“” ') for line in _lines(dialogue)]
    if spoken:
        parts.append("The person says, in this order, unhurried and with natural lip sync: "
                     + " ".join(f'"{line}"' for line in spoken))
        words = sum(len(line.split()) for line in spoken)
        if words > WORDS_PER_SECOND * seconds:
            warnings.append(f"{words} spoken words in {seconds:g} s will sound rushed; "
                            f"about {int(WORDS_PER_SECOND * seconds)} fit")
    if audio.strip():
        a = " ".join(audio.split())
        parts.append(a if a.lower().startswith("audio:") else "Audio: " + a)
    if avoid.strip():
        parts.append(" ".join(avoid.split()))
    return "\n\n".join(p for p in parts if p), warnings


class AIInfluencerVideoPrompt(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerVideoPrompt",
            display_name="Video Description (AI Influencer, MiniMax H3)",
            category="ai_influencer",
            search_aliases=["minimax", "h3", "video prompt", "motion prompt"],
            description=(
                "Builds the MiniMax H3 description from short fields: one beat per line is timed "
                "evenly over the duration, and the frame count is worked out for you. Always adds "
                "the blurred background and the handheld-phone look."
            ),
            inputs=[
                IO.Float.Input("duration_seconds", default=5.0, min=3.0, max=15.0, step=0.5,
                               tooltip="Clip length. H3 is trained on roughly 5-15 s; one continuous shot per 5 s works best."),
                IO.String.Input("look", multiline=True,
                                default="Photoreal vertical social clip of the person from the starting image: same face, "
                                        "same hair, same outfit, same light. Keep their identity, skin texture and "
                                        "clothing identical to the starting image.",
                                tooltip="Who and what must stay the same. Don't re-describe the scenery: the model sees the starting image."),
                IO.String.Input("beats", multiline=True,
                                default="The shot opens exactly on the starting image; the person is still for a beat, then blinks and takes a small breath.\n"
                                        "They turn their head a little toward the camera and a small, genuine smile grows.\n"
                                        "They say the line below, relaxed, with a slight shoulder movement.\n"
                                        "They laugh softly, glance off-camera and settle into a calm half-smile while the camera drifts a little closer.",
                                tooltip="One beat per line: what the body does and what the camera does. Timed evenly over the "
                                        "duration; start a line with [1.5s-3.0s] to time it yourself. Small motions read best."),
                IO.String.Input("dialogue", multiline=True, default="Okay, this light is unreal right now.",
                                tooltip="What the person says, one line each, no quotes needed. About 2.5 words per second "
                                        "fit. Leave empty for no speech."),
                IO.String.Input("audio", multiline=True,
                                default="the person's natural voice, close to the mic and conversational, no reverb; under it the "
                                        "quiet room tone of the location. No music, no second speaker.",
                                tooltip="Voice, then background sound. Say 'no music' if you don't want any: silence is not the default."),
                IO.String.Input("avoid", multiline=True,
                                default="No cuts, no scene changes, no text overlays, no extra hands or fingers, no warping of the face.",
                                tooltip="The negative tail."),
            ],
            outputs=[
                IO.String.Output(display_name="description"),
                IO.Int.Output(display_name="length", tooltip="Frame count for the H3 node's length input."),
                IO.String.Output(display_name="duration", tooltip="For the Save & Log node's duration_seconds."),
            ],
        )

    @classmethod
    def execute(cls, duration_seconds: float, look: str, beats: str, dialogue: str, audio: str, avoid: str) -> IO.NodeOutput:
        text, warnings = build(look, beats, dialogue, audio, avoid, duration_seconds)
        frames = frame_count(duration_seconds)
        shown = "".join(f"!! {w}\n" for w in warnings) + f"{frames} frames ({frames / FPS:.2f} s)\n\n{text}"
        return IO.NodeOutput(text, frames, f"{duration_seconds:g}", ui=ui.PreviewText(shown))

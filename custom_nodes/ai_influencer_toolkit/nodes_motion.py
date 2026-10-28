"""Motion transfer (Workflow 14): the character from a still copies the movement of a driving video.

Wan Animate 2 (core ComfyUI's WanAnimate2ToVideo) needs the driving video cut to the part you want,
at one frame rate, at a size in multiples of 16, and with a frame count on Wan's 4k+1 grid. Doing that
with stock nodes means a calculator node per number (the "Motion Control" workflow this replaces had
five). AIInfluencerDrivingVideo does it in one place, and trims the video *before* decoding it, so a
long 4K phone clip isn't loaded into memory whole.

AIInfluencerMotionPrompt is the prompt side. Like the MiniMax H3 description (nodes_video.py) it always
adds MANDATORY_VIDEO_LOOK, which is not an input and so can't be edited away in one workflow.
"""

from __future__ import annotations

import torch

import comfy.utils
from comfy_api.latest import IO, Input, ui

from .nodes_video import MANDATORY_VIDEO_LOOK

WINDOW_FRAMES = 81  # Wan's trained clip length; longer clips are sampled in overlapping context windows


def snap_length(n: int) -> int:
    """Largest 4k+1 frame count that fits in n frames (Wan's VAE packs 4 frames per latent after the first)."""
    return max(1, (n - 1) // 4 * 4 + 1)


def target_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """The video's aspect ratio with its longer side at long_side, both sides in multiples of 16."""
    scale = long_side / max(width, height)
    return max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)


def resample(frames: torch.Tensor, src_fps: float, dst_fps: float) -> torch.Tensor:
    """Pick the source frame nearest to each output frame's time. Slower output rates drop frames, faster
    ones repeat them; either way the motion keeps its real speed."""
    if not dst_fps or abs(src_fps - dst_fps) < 0.01:
        return frames
    count = max(1, int(frames.shape[0] * dst_fps / src_fps))
    idx = (torch.arange(count, dtype=torch.float64) * src_fps / dst_fps).round().long().clamp(max=frames.shape[0] - 1)
    return frames[idx]


def trim_audio(audio: dict | None, seconds: float) -> dict | None:
    if audio is None:
        return None
    keep = int(round(seconds * audio["sample_rate"]))
    return {**audio, "waveform": audio["waveform"][..., :keep]}


class AIInfluencerDrivingVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerDrivingVideo",
            display_name="Driving Video (AI Influencer, Wan Animate)",
            category="ai_influencer",
            search_aliases=["motion transfer", "wan animate", "dance", "motion control", "driving video"],
            description=(
                "Cuts the part of the driving video to copy, sets its frame rate and size, and works out the "
                "frame count Wan Animate needs. Its sound is kept and trimmed to match."
            ),
            inputs=[
                IO.Video.Input("video", tooltip="The video whose movement the person copies (dance, walk, gesture...)."),
                IO.Float.Input("start_seconds", default=0.0, min=0.0, max=3600.0, step=0.1,
                               tooltip="Skip this much of the video first."),
                IO.Float.Input("duration_seconds", default=5.0, min=0.0, max=60.0, step=0.5,
                               tooltip="How much of it to use. 0 = to the end. Up to about 2.7 s (81 frames at 30 fps) is "
                                       "one pass; longer clips are made in overlapping windows, which takes "
                                       "proportionally longer."),
                IO.Float.Input("fps", default=30.0, min=0.0, max=60.0, step=1.0,
                               tooltip="Frame rate of the result. 30 is what Wan Animate is trained on; lower is faster to "
                                       "generate (fewer frames for the same seconds) but choppier. 0 = keep the video's own rate."),
                IO.Int.Input("long_side", default=832, min=256, max=1920, step=16,
                             tooltip="Size of the longer side in pixels; the video's aspect ratio is kept. 832 = 480p class "
                                     "(fast), 1280 = 720p class (about 2.5x slower and needs more memory)."),
            ],
            outputs=[
                IO.Image.Output(display_name="frames", tooltip="Resized, resampled frames for the pose_video input."),
                IO.Audio.Output(display_name="audio", tooltip="The video's sound for the same span (empty if it has none)."),
                IO.Float.Output(display_name="fps"),
                IO.Int.Output(display_name="length", tooltip="Frame count for Wan Animate (4k+1)."),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
                IO.Boolean.Output(display_name="long_clip", tooltip="True when the clip needs context windows."),
                IO.String.Output(display_name="duration", tooltip="For the Save & Log node's duration_seconds."),
            ],
        )

    @classmethod
    def execute(cls, video: Input.Video, start_seconds: float, duration_seconds: float, fps: float, long_side: int) -> IO.NodeOutput:
        total = video.get_duration()
        if start_seconds >= total:
            raise ValueError(f"start_seconds {start_seconds:g} is past the end of the video ({total:.1f} s).")
        clip = video.as_trimmed(start_seconds, duration_seconds, strict_duration=False) if (start_seconds or duration_seconds) else video
        parts = clip.get_components()
        if parts.images.shape[0] == 0:
            raise ValueError(f"No frames between {start_seconds:g} s and the end of the video ({total:.1f} s).")
        src_fps = float(parts.frame_rate)
        frames = resample(parts.images, src_fps, fps)
        out_fps = fps or src_fps
        length = snap_length(frames.shape[0])
        frames = frames[:length]
        width, height = target_size(frames.shape[2], frames.shape[1], long_side)
        frames = comfy.utils.common_upscale(frames.movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
        seconds = length / out_fps
        audio = trim_audio(parts.audio, seconds)
        long_clip = length > WINDOW_FRAMES
        shown = (f"{length} frames at {out_fps:g} fps ({seconds:.2f} s), {width}x{height}"
                 f"{', in context windows' if long_clip else ''}{', no sound' if audio is None else ''}")
        return IO.NodeOutput(frames, audio, out_fps, length, width, height, long_clip, f"{seconds:.1f}",
                             ui=ui.PreviewText(shown))


class AIInfluencerMotionPrompt(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerMotionPrompt",
            display_name="Motion Description (AI Influencer, Wan Animate)",
            category="ai_influencer",
            search_aliases=["motion transfer", "wan animate", "motion prompt"],
            description=(
                "The two prompts Wan Animate 2 reads: who the person is (with the blurred background and "
                "handheld-phone look always added) and what the driving video's movement is."
            ),
            inputs=[
                IO.String.Input("look", multiline=True,
                                default="Photoreal vertical social clip of the person from the reference image: same face, "
                                        "same hair, same outfit, same light and location. Keep their identity, skin texture "
                                        "and clothing identical to the reference image.",
                                tooltip="Who and what must stay the same. The model sees the reference image, so don't "
                                        "describe it in detail."),
                IO.String.Input("motion", multiline=True,
                                default="The person moves exactly like the driving video, matching the timing of every step, "
                                        "turn, arm movement, hand gesture and facial expression.",
                                tooltip="What the movement is. Naming it ('a slow dance', 'walks toward the camera and waves') "
                                        "helps the pose branch more than adjectives do."),
                IO.String.Input("avoid", multiline=True,
                                default="oversaturated, overexposed, static, blurry details, subtitles, text, watermark, painting, "
                                        "illustration, cartoon, airbrushed plastic skin, HDR, oversharpened, worst quality, low "
                                        "quality, JPEG artifacts, deformed, extra fingers, fused fingers, poorly drawn hands, "
                                        "poorly drawn face, extra limbs, warped face, flickering, scene cut",
                                tooltip="The negative prompt."),
            ],
            outputs=[
                IO.String.Output(display_name="positive", tooltip="Character prompt: look + the mandatory video look."),
                IO.String.Output(display_name="motion", tooltip="For the pose branch (positive_pose)."),
                IO.String.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, look: str, motion: str, avoid: str) -> IO.NodeOutput:
        positive = "\n\n".join(p for p in (" ".join(look.split()), MANDATORY_VIDEO_LOOK) if p)
        motion = " ".join(motion.split())
        return IO.NodeOutput(positive, motion, " ".join(avoid.split()), ui=ui.PreviewText(f"{positive}\n\nMotion: {motion}"))

"""Pose Angle per Seed: the same pose, seen from a slightly different spot on every seed.

Workflow 09d starts each image from the pose photo (image-to-image), so the pose and framing are the
photo's. To get variety without losing the pose, this node moves the "camera" a little before the
photo is encoded: a small turn (perspective keystone), a slight tilt (roll), a little zoom and shift.
The skeleton is then detected on the moved photo, so the starting image, the skeleton and the result
agree. Every amount is a widget; `variation` scales them all at once (0 = the photo unchanged).

Before the camera moves, the photo is framed like a phone photo (`frame`): by default the location
photo's shape when the person fits in it, else the nearest standard phone shape that does (3:4, 1:1,
4:3, 3:2 ...) -- cropped around the person, never padded, so a 2:1 pose photo can't produce a panorama.

Like Placement & Angle, one queue can make several candidates (`candidates`), each with seed + i, and
everything is a pure function of the seed, so a saved PNG dragged back rebuilds the same view. Saved
images go to one folder per prompt by default (`folder_by`), so every seed of a prompt ends up together.
"""
from __future__ import annotations

import hashlib
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import IO


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 H with dst ~ H @ src, from 4 point pairs."""
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.asarray(a, np.float64), np.asarray(b, np.float64))
    return np.append(h, 1.0).reshape(3, 3)


def view(seed: int, variation: float, max_turn: float, max_tilt: float, max_zoom: float, max_shift: float):
    """(turn, tilt_deg, zoom, shift_x, shift_y) for a seed. turn > 0 = the camera moved to the right."""
    # consecutive seeds must give unrelated angles: scramble the seed first (random.Random(seed) gives
    # near-identical first draws for neighbouring seeds)
    rng = random.Random(int.from_bytes(hashlib.sha256(f"pose-angle:{seed}".encode()).digest()[:8], "big"))
    v = max(0.0, variation)
    turn = rng.uniform(-1, 1) * max_turn * v
    tilt = rng.uniform(-1, 1) * max_tilt * v
    zoom = rng.uniform(0, 1) * max_zoom * v
    sx, sy = rng.uniform(-1, 1) * max_shift * v, rng.uniform(-1, 1) * max_shift * v * 0.5
    return turn, tilt, zoom, sx, sy


def warp(image: torch.Tensor, turn: float, tilt_deg: float, zoom: float, sx: float, sy: float) -> torch.Tensor:
    """image [B,H,W,C] -> the same scene from a slightly moved camera. Works in normalized [-1, 1]
    coordinates; zooms in just enough that no empty corners show."""
    b, h, w, c = image.shape
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
    # turn: the side the camera moved toward gets taller (nearer), the other side shorter, and the
    # whole frame is squeezed a little horizontally -- a keystone, like a small yaw
    k = turn
    keyed = np.array([[-1 + abs(k) * 0.3, -(1 - k)], [1 - abs(k) * 0.3, -(1 + k)],
                      [1 - abs(k) * 0.3, (1 + k)], [-1 + abs(k) * 0.3, (1 - k)]], np.float64)
    r = math.radians(tilt_deg)
    rot = np.array([[math.cos(r), -math.sin(r) * h / w], [math.sin(r) * w / h, math.cos(r)]])
    # zoom in enough to cover the frame after the keystone and the tilt, then the extra zoom
    cover = (1 + abs(k)) * (abs(math.cos(r)) + abs(math.sin(r)) * max(h / w, w / h)) + 2 * max(abs(sx), abs(sy))
    scale = cover * (1 + zoom)
    dst = (keyed @ rot.T) * scale + np.array([sx, sy])
    hm = _homography(corners, dst)                 # where the photo's corners land
    inv = np.linalg.inv(hm)                        # for each output pixel: where to read from
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    pts = torch.stack([xs, ys, torch.ones_like(xs)], -1).double() @ torch.from_numpy(inv).T
    grid = (pts[..., :2] / pts[..., 2:3]).float()[None].expand(b, h, w, 2)
    out = F.grid_sample(image.movedim(-1, 1).float(), grid.to(image.device), mode="bicubic",
                        padding_mode="border", align_corners=True)
    return out.clamp(0, 1).movedim(1, -1)


FRAMES = ["auto (location shape if the person fits)", "location photo's shape", "pose photo as is",
          "2:3 portrait", "3:4 portrait", "9:16 portrait", "1:1 square", "4:3 landscape", "3:2 landscape"]
FOLDER_BY = ["prompt (all seeds of a prompt together)", "queue (one folder per queue)"]
PHONE_SHAPES = [2 / 3, 3 / 4, 9 / 16, 1.0, 4 / 3, 3 / 2, 16 / 9]  # width / height: what phones take
SCORE = 0.3


def _person_box(keypoints, w: int, h: int, room: float = 1.0):
    """(x0, y0, x1, y1) in pixels around the pose photo's best person, with headroom (scaled by `room`;
    0 = tight around the detected points); None if no person."""
    if not keypoints or not keypoints[0].get("people"):
        return None
    f = keypoints[0]
    sx, sy = w / f["canvas_width"], h / f["canvas_height"]
    best = None
    for person in f["people"]:
        pts = []
        for k in ("pose_keypoints_2d", "foot_keypoints_2d", "hand_right_keypoints_2d", "hand_left_keypoints_2d"):
            a = np.asarray(person.get(k) or [], np.float32).reshape(-1, 3)
            pts += [(x * sx, y * sy) for x, y, c in a if c >= SCORE and x >= 0 and y >= 0]
        if len(pts) >= 4 and (best is None or len(pts) > len(best)):
            best = pts
    if best is None:
        return None
    a = np.asarray(best)
    x0, y0 = a.min(0); x1, y1 = a.max(0)
    bw, bh = x1 - x0, y1 - y0
    size = max(bw, bh)
    # room around the body: the head's top and hair are above the nose/eyes, feet below the ankles
    m = room * size
    return (max(0.0, x0 - 0.12 * m), max(0.0, y0 - 0.25 * m), min(w, x1 + 0.12 * m), min(h, y1 + 0.12 * m))


def _crop_for(w: int, h: int, aspect: float, box) -> tuple[int, int, int, int] | None:
    """The largest crop of the photo with this aspect that contains `box`, centred on it; None if impossible."""
    cw, ch = (w, w / aspect) if w / h < aspect else (h * aspect, h)
    if box is not None and ((box[2] - box[0]) > cw + 1 or (box[3] - box[1]) > ch + 1):
        return None
    cx = (box[0] + box[2]) / 2 if box else w / 2
    cy = (box[1] + box[3]) / 2 if box else h / 2
    x0 = min(max(cx - cw / 2, 0), w - cw)
    y0 = min(max(cy - ch / 2, 0), h - ch)
    if box is not None:  # keep the whole person inside even when centring pushed them to an edge
        x0 = min(max(x0, box[2] - cw), box[0]); y0 = min(max(y0, box[3] - ch), box[1])
        x0 = min(max(x0, 0), w - cw); y0 = min(max(y0, 0), h - ch)
    return int(round(x0)), int(round(y0)), int(round(cw)), int(round(ch))


def frame_photo(image: torch.Tensor, keypoints, frame: str, location_aspect: float | None):
    """(framed image at ~1 MP, label). Crops only -- never pads."""
    _, h, w, _ = image.shape
    box = _person_box(keypoints, w, h)
    fixed = {"2:3 portrait": 2 / 3, "3:4 portrait": 3 / 4, "9:16 portrait": 9 / 16, "1:1 square": 1.0,
             "4:3 landscape": 4 / 3, "3:2 landscape": 3 / 2}
    if frame == "pose photo as is":
        crop, shape = (0, 0, w, h), w / h
    else:
        if frame in fixed:
            wanted = [fixed[frame]]
        elif frame == "location photo's shape":
            wanted = [location_aspect or w / h]
        else:  # auto: the location's shape first, then the phone shape closest to it that fits the person
            first = location_aspect or 3 / 4
            wanted = [first] + sorted(PHONE_SHAPES, key=lambda a: abs(math.log(a / first)))
        crop, shape = None, None
        for b in (box, _person_box(keypoints, w, h, room=0.0)):   # with headroom first, then tight
            for a in wanted:
                crop = _crop_for(w, h, a, b)
                if crop:
                    shape = a
                    break
            if crop:
                break
        if crop is None:
            # the person is wider/taller than any phone shape: a real phone photo would cut a hand or a foot
            # at the edge, which looks natural; a 2:1 strip doesn't. Phone shape closest to the photo's,
            # centred on the person.
            shape = min(wanted if frame in fixed or frame == "location photo's shape" else PHONE_SHAPES,
                        key=lambda a: abs(math.log(a / (w / h))))
            crop = _crop_for(w, h, shape, None)  # largest crop of that shape; recentred on the person below
            if box is not None:
                x0, y0, cw, ch = crop
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                crop = (int(min(max(cx - cw / 2, 0), w - cw)), int(min(max(cy - ch / 2, 0), h - ch)), cw, ch)
    x0, y0, cw, ch = crop
    out = image[:, y0:y0 + ch, x0:x0 + cw, :]
    # ~1 MP, sides multiples of 16
    scale = math.sqrt(1024 * 1024 / (cw * ch))
    tw, th = max(16, int(round(cw * scale / 16)) * 16), max(16, int(round(ch * scale / 16)) * 16)
    out = F.interpolate(out.movedim(-1, 1).float(), size=(th, tw), mode="bicubic", antialias=True,
                        align_corners=False).clamp(0, 1).movedim(1, -1)
    return out, f"frame {tw}x{th} ({shape:.2f}{'' if box else ', no person found: centred'})"


def prompt_folder(prompt: str | None, seed: int, folder_by: str) -> str:
    if folder_by.startswith("prompt") and prompt:
        return "9d_" + hashlib.sha1(prompt.strip().encode()).hexdigest()[:8]
    return f"9d_{seed}"


class AIInfluencerPoseAngle(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerPoseAngle",
            display_name="Pose Angle per Seed (AI Influencer)",
            category="ai_influencer",
            search_aliases=["camera angle", "pose variation", "viewpoint", "framing"],
            description=(
                "Frames the pose photo like a phone photo around the person, then moves the camera a little for "
                "every seed (turn, tilt, zoom, shift), so image-to-image keeps the pose but each seed is seen from a "
                "slightly different spot."
            ),
            inputs=[
                IO.Image.Input("pose_photo", tooltip="The pose reference photo (already scaled)."),
                IO.Custom("POSE_KEYPOINT").Input("pose_keypoints", optional=True,
                                                 tooltip="SDPose keypoints of the ORIGINAL pose photo, to frame around the person."),
                IO.Image.Input("location", optional=True, tooltip="The location photo: its shape is the preferred frame."),
                IO.String.Input("prompt", optional=True, force_input=True, tooltip="The prompt, to name the save folder after it."),
                IO.Int.Input("seed", default=0, min=0, max=2**31 - 1, control_after_generate=True,
                             tooltip="Base seed: candidate i uses seed + i. Also the sampler's seed."),
                IO.Int.Input("candidates", default=1, min=1, max=8, tooltip="Images per queue, each from its own angle."),
                IO.Combo.Input("frame", options=FRAMES, default=FRAMES[0],
                               tooltip="The photo's shape, cropped around the person (never padded). 'auto' uses the location "
                                       "photo's shape when the person fits, else the nearest phone shape that does."),
                IO.Float.Input("variation", default=0.6, min=0.0, max=1.0, step=0.05,
                               tooltip="How different the angles are. 0 = the framed photo exactly, 1 = the maximums below."),
                IO.Float.Input("max_turn", default=0.12, min=0.0, max=0.35, step=0.01,
                               tooltip="Largest sideways camera move (perspective). 0.12 = a few steps to the side."),
                IO.Float.Input("max_tilt", default=4.0, min=0.0, max=15.0, step=0.5, tooltip="Largest camera tilt, degrees."),
                IO.Float.Input("max_zoom", default=0.1, min=0.0, max=0.5, step=0.01, tooltip="Largest extra zoom-in (0.1 = 10%)."),
                IO.Float.Input("max_shift", default=0.04, min=0.0, max=0.2, step=0.01,
                               tooltip="Largest framing shift, fraction of the frame."),
                IO.Combo.Input("folder_by", options=FOLDER_BY, default=FOLDER_BY[0],
                               tooltip="Save folder: one per prompt (every seed of a prompt together) or one per queue."),
            ],
            outputs=[
                IO.Image.Output(display_name="pose_photo", is_output_list=True, tooltip="The framed photo from this seed's angle."),
                IO.Int.Output(display_name="seed", is_output_list=True),
                IO.String.Output(display_name="folder", is_output_list=True, tooltip="For the Save node's subfolder."),
                IO.String.Output(display_name="label", is_output_list=True, tooltip="The frame and angle each candidate got."),
            ],
        )

    @classmethod
    def execute(cls, pose_photo, seed, candidates, variation, max_turn, max_tilt, max_zoom, max_shift,
                frame=FRAMES[0], folder_by=FOLDER_BY[0], pose_keypoints=None, location=None, prompt=None) -> IO.NodeOutput:
        loc_aspect = (location.shape[2] / location.shape[1]) if location is not None else None
        framed, frame_label = frame_photo(pose_photo, pose_keypoints, frame, loc_aspect)
        folder = prompt_folder(prompt, seed, folder_by)
        images, seeds, labels = [], [], []
        for i in range(candidates):
            s = (seed + i) % (2**31 - 1)
            turn, tilt, zoom, sx, sy = view(s, variation, max_turn, max_tilt, max_zoom, max_shift)
            images.append(warp(framed, turn, tilt, zoom, sx, sy) if variation > 0 else framed)
            seeds.append(s)
            side = "right" if turn > 0 else "left"
            labels.append(f"#{i + 1} seed {s}: {frame_label}; camera {abs(turn) / max(max_turn, 1e-6) * 100:.0f}% of max "
                          f"to the {side}, tilt {tilt:+.1f} deg, zoom +{zoom * 100:.0f}%")
        return IO.NodeOutput(images, seeds, [folder] * candidates, labels)

"""Placement & Angle per seed: several candidates per queue in Workflow 09b, each one with her
standing somewhere else in the location photo and turned a different way.

Without this, every seed in 09b puts her in the same spot (the mask comes from the pose photo or a
fixed centre area) at the same angle (the skeleton comes from the pose photo), so re-rolling only
changes small details. This node sits between the pose extractor / Prompt Builder and the rest of
the graph and turns one queue into `candidates` separate jobs, each with its own:

  * seed              base seed + i, fed to the sampler (and logged by the Save node);
  * place             a horizontal slot across the plate and a depth (nearer = bigger and lower,
                      further back = smaller and higher, pinned to a horizon line so her eye line
                      stays plausible for an eye-level photo);
  * angle             a body turn (front, three-quarter, profile, back-over-the-shoulder, each to
                      either side) -- drawn as the OpenPose skeleton *and* written into the prompt,
                      so the two never disagree;
  * keypoints         that skeleton in plate pixels: the Auto Mask draws her mask from it and the
                      skeleton drawer feeds it to Krea 2 as picture 3.

Within one queue no two candidates share a slot or an angle (up to 5 candidates for slots, 8 for
angles), and everything is a pure function of (seed, candidate index), so dragging a saved PNG
back onto the canvas rebuilds the same candidates.

With `pose` = "reference pose", the pose photo's skeleton is kept (its arms and legs are the point
of giving one) and only moved, rescaled and mirrored on alternate candidates; a 2D skeleton cannot
be turned to a new angle honestly, so the prompt gets no angle words in that mode.

All outputs are lists (one entry per candidate), so ComfyUI runs everything downstream once per
candidate -- the sampler, the composite, the filters and the Save node.
"""
from __future__ import annotations

import math
import random

import numpy as np

from comfy_api.latest import IO

VARY = ["placement + angle", "placement only", "off"]
POSE = ["new angle per seed", "reference pose (moved, mirrored)"]
SLOTS = 5
SCORE = 0.3

# (yaw in degrees, prompt wording). Positive yaw = facing frame-right. Editable on the node.
ANGLES = [
    (0, "the body facing the camera"),
    (35, "the body turned slightly toward frame-right, face toward the camera"),
    (-35, "the body turned slightly toward frame-left, face toward the camera"),
    (65, "three-quarter view, the body turned toward frame-right, head turned back toward the camera"),
    (-65, "three-quarter view, the body turned toward frame-left, head turned back toward the camera"),
    (90, "seen in profile, facing frame-right"),
    (-90, "seen in profile, facing frame-left"),
    (160, "seen from behind, looking back over the shoulder toward the camera"),
]
DEFAULT_ANGLES = "\n".join(f"{yaw}: {words}" for yaw, words in ANGLES)


def parse_angles(text: str) -> list[tuple[float, str]]:
    """'degrees: words' per line; blank lines and lines starting with # are ignored."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        yaw, sep, words = line.partition(":")
        try:
            out.append((float(yaw), words.strip() if sep else ""))
        except ValueError:
            raise ValueError(f"angles: '{line}' should look like '35: the body turned slightly toward frame-right'")
    return out or ANGLES

# A standing body in OpenPose-18 order, 3D, unit height: x to the image right when she faces the
# camera (so her right side is at negative x), y down from the top of the head, z toward the camera.
_BODY = np.array([
    [0.0, 0.075, 0.06],     # 0 nose
    [0.0, 0.17, 0.0],       # 1 neck
    [-0.115, 0.19, 0.0],    # 2 R shoulder
    [-0.14, 0.33, -0.01],   # 3 R elbow
    [-0.15, 0.46, 0.02],    # 4 R wrist
    [0.115, 0.19, 0.0],     # 5 L shoulder
    [0.14, 0.33, -0.01],    # 6 L elbow
    [0.15, 0.46, 0.02],     # 7 L wrist
    [-0.065, 0.50, 0.0],    # 8 R hip
    [-0.07, 0.72, 0.01],    # 9 R knee
    [-0.075, 0.95, 0.0],    # 10 R ankle
    [0.065, 0.50, 0.0],     # 11 L hip
    [0.07, 0.72, 0.01],     # 12 L knee
    [0.075, 0.95, 0.0],     # 13 L ankle
    [-0.025, 0.06, 0.05],   # 14 R eye
    [0.025, 0.06, 0.05],    # 15 L eye
    [-0.06, 0.07, -0.01],   # 16 R ear
    [0.06, 0.07, -0.01],    # 17 L ear
], dtype=np.float32)

# arm variants: (R elbow, R wrist, L elbow, L wrist) replacements, or None for relaxed
_ARMS = [
    None,
    {6: [0.2, 0.36, -0.02], 7: [0.08, 0.50, 0.03]},                             # left hand on hip
    {3: [-0.2, 0.36, -0.02], 4: [-0.08, 0.50, 0.03]},                           # right hand on hip
    {6: [0.13, 0.30, 0.06], 7: [0.07, 0.40, 0.10]},                             # left forearm across, holding something
]

# left/right swaps for mirroring a real skeleton (OpenPose body, SDPose feet)
_BODY_SWAP = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
_FOOT_SWAP = [3, 4, 5, 0, 1, 2]


def _people_arrays(person: dict) -> dict[str, np.ndarray]:
    out = {}
    for key, n in (("pose_keypoints_2d", 18), ("foot_keypoints_2d", 6), ("face_keypoints_2d", 70),
                   ("hand_right_keypoints_2d", 21), ("hand_left_keypoints_2d", 21)):
        flat = person.get(key)
        arr = np.asarray(flat, dtype=np.float32).reshape(n, 3) if flat else np.zeros((n, 3), np.float32)
        out[key] = arr.copy()
    return out


def _best_person(keypoints) -> tuple[dict[str, np.ndarray], int, int] | None:
    if not keypoints:
        return None
    frame = keypoints[0]
    best, best_n = None, 5
    for person in frame.get("people", []):
        arrs = _people_arrays(person)
        body = arrs["pose_keypoints_2d"]
        n = int(((body[:, 2] >= SCORE) & (body[:, 0] > 0)).sum())
        if n > best_n:
            best, best_n = arrs, n
    if best is None:
        return None
    return best, int(frame["canvas_width"]), int(frame["canvas_height"])


def _synthetic(yaw_deg: float, arms: int) -> dict[str, np.ndarray]:
    """The standing body turned by yaw, projected orthographically; points on the far side of the
    head are hidden (score 0) the way a real detector would miss them."""
    body = _BODY.copy()
    for i, xyz in (_ARMS[arms] or {}).items():
        body[i] = xyz
    t = math.radians(yaw_deg)
    x = body[:, 0] * math.cos(t) + body[:, 2] * math.sin(t)
    z = -body[:, 0] * math.sin(t) + body[:, 2] * math.cos(t)
    score = np.ones(18, np.float32)
    for i in (0, 14, 15):          # nose and eyes: only when turned toward the camera
        if z[i] < 0.0:
            score[i] = 0.0
    for i in (16, 17):             # ears: hidden behind the head once turned away enough
        if z[i] < -0.035:
            score[i] = 0.0
    pose = np.stack([x, body[:, 1], score], axis=1).astype(np.float32)
    empty = lambda n: np.zeros((n, 3), np.float32)  # noqa: E731
    return {"pose_keypoints_2d": pose, "foot_keypoints_2d": empty(6), "face_keypoints_2d": empty(70),
            "hand_right_keypoints_2d": empty(21), "hand_left_keypoints_2d": empty(21)}


def _mirror(arrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in arrs.items()}
    for v in out.values():
        valid = v[:, 2] >= SCORE
        v[valid, 0] = -v[valid, 0]  # _place re-centres it afterwards
    out["pose_keypoints_2d"] = out["pose_keypoints_2d"][_BODY_SWAP]
    out["foot_keypoints_2d"] = out["foot_keypoints_2d"][_FOOT_SWAP]
    out["hand_right_keypoints_2d"], out["hand_left_keypoints_2d"] = out["hand_left_keypoints_2d"], out["hand_right_keypoints_2d"]
    return out


def _place(arrs: dict[str, np.ndarray], w: int, h: int, center_x: float, height_px: float, horizon: float) -> dict[str, np.ndarray]:
    """Scale the skeleton so its body is `height_px` tall, stand its feet on the ground line that
    puts an eye-level camera's horizon through its eyes, and centre it on `center_x`."""
    body = arrs["pose_keypoints_2d"]
    valid = body[:, 2] >= SCORE
    ys = body[valid, 1]
    src_h = max(float(ys.max() - ys.min()), 1e-6)
    s = height_px / src_h
    top = float(ys.min())
    cx = float(body[valid, 0].mean())
    hz = horizon * h
    # eyes sit ~6% below the top of a standing body; put them on the horizon
    new_top = hz - 0.06 * height_px
    bottom = new_top + height_px
    if bottom > 0.99 * h:                        # never below the frame: lift her instead
        new_top -= bottom - 0.99 * h
    half = 0.5 * s * float(body[valid, 0].max() - body[valid, 0].min())
    center_x = min(max(center_x, half + 0.02 * w), w - half - 0.02 * w)

    out = {}
    for k, v in arrs.items():
        v = v.copy()
        ok = v[:, 2] >= SCORE  # undetected / hidden points stay at -1, as SDPose leaves them
        v[ok, 0] = (v[ok, 0] - cx) * s + center_x
        v[ok, 1] = (v[ok, 1] - top) * s + new_top
        v[~ok, :2] = -1.0
        out[k] = v
    return out


def _frame(arrs: dict[str, np.ndarray], w: int, h: int) -> list[dict]:
    return [{"canvas_width": w, "canvas_height": h,
             "people": [{k: v.reshape(-1).tolist() for k, v in arrs.items()}]}]


def plan(seed: int, candidates: int, vary: str, pose_mode: str, has_reference: bool,
         x_min: float, x_max: float, size_min: float, size_max: float, n_angles: int = len(ANGLES)) -> list[dict]:
    """What each candidate gets. Deterministic in (seed, index)."""
    run = random.Random(seed)
    slots = list(range(SLOTS))
    run.shuffle(slots)
    angles = list(range(n_angles))
    run.shuffle(angles)
    out = []
    for i in range(candidates):
        rng = random.Random(seed * 1000003 + i)
        item = {"seed": (seed + i) % 2**31, "index": i + 1}
        if vary == "off":
            item.update(slot=None, angle=None, mirror=False, size=None, arms=0)
        else:
            slot = slots[i % SLOTS]
            lo = x_min + (x_max - x_min) * slot / SLOTS
            item["x"] = lo + (x_max - x_min) / SLOTS * rng.uniform(0.25, 0.75)
            item["slot"] = slot
            item["size"] = rng.uniform(size_min, size_max)
            if vary == "placement + angle" and not (has_reference and pose_mode == POSE[1]):
                item["angle"] = angles[i % n_angles]
                item["arms"] = rng.randrange(len(_ARMS))
                item["mirror"] = False
            else:
                item["angle"] = None
                item["arms"] = 0
                item["mirror"] = vary == "placement + angle" and i % 2 == 1
        out.append(item)
    return out


class AIInfluencerPlacementVariations(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerPlacementVariations",
            display_name="Placement & Angle per Seed (AI Influencer)",
            category="ai_influencer",
            search_aliases=["candidates", "variations", "placement", "angle"],
            description=(
                "Turns one queue into several candidates, each with its own seed, its own place in the "
                "location photo and its own body angle (skeleton and prompt wording agree). Everything "
                "downstream runs once per candidate."
            ),
            inputs=[
                IO.Image.Input("plate", tooltip="The location plate at the size it is generated at."),
                IO.String.Input("prompt", multiline=True, force_input=True, tooltip="The Prompt Builder's output."),
                IO.Custom("POSE_KEYPOINT").Input("pose_keypoints", optional=True, tooltip="SDPose keypoints of the pose reference photo."),
                IO.Int.Input("seed", default=0, min=0, max=2**31 - 1, control_after_generate=True,
                             tooltip="Base seed: candidate i uses seed + i - 1. Randomize it to get a fresh set per queue."),
                IO.Int.Input("candidates", default=4, min=1, max=8, tooltip="Images per queue. Each one costs a full generation."),
                IO.Combo.Input("vary", options=VARY, default=VARY[0],
                               tooltip="What changes between candidates. 'off' = old behaviour: same place and pose, only the seed changes."),
                IO.Combo.Input("pose", options=POSE, default=POSE[0],
                               tooltip="'new angle per seed' draws a fresh standing skeleton at a new angle for every candidate. "
                                       "'reference pose' keeps the pose photo's skeleton and only moves, rescales and mirrors it."),
                IO.Float.Input("x_min", default=0.15, min=0.0, max=1.0, step=0.01, tooltip="Leftmost centre position, fraction of the plate width."),
                IO.Float.Input("x_max", default=0.85, min=0.0, max=1.0, step=0.01, tooltip="Rightmost centre position, fraction of the plate width."),
                IO.Float.Input("size_min", default=0.45, min=0.1, max=1.0, step=0.01, tooltip="Smallest body height, fraction of the plate height (further back)."),
                IO.Float.Input("size_max", default=0.7, min=0.1, max=1.0, step=0.01, tooltip="Largest body height, fraction of the plate height (nearer)."),
                IO.Float.Input("horizon", default=0.35, min=0.0, max=1.0, step=0.01,
                               tooltip="Height of the photo's horizon (eye level), fraction from the top. The person's eyes are put on it, "
                                       "so nearer = bigger and lower, further = smaller and higher."),
                IO.String.Input("angles", multiline=True, default=DEFAULT_ANGLES,
                                tooltip="The body angles candidates are drawn from, one per line: 'degrees: words for the prompt'. "
                                        "0 = facing the camera, positive = turned toward frame-right, 180 = back to the camera. "
                                        "Delete lines you don't want; add your own."),
            ],
            outputs=[
                IO.Custom("POSE_KEYPOINT").Output(display_name="keypoints", is_output_list=True),
                IO.String.Output(display_name="prompt", is_output_list=True),
                IO.Int.Output(display_name="seed", is_output_list=True),
                IO.String.Output(display_name="folder", is_output_list=True, tooltip="One folder per queue, for the Save node's subfolder."),
                IO.String.Output(display_name="label", is_output_list=True, tooltip="What each candidate got."),
            ],
        )

    @classmethod
    def execute(cls, plate, prompt, seed, candidates, vary, pose, x_min, x_max, size_min, size_max, horizon,
                angles=DEFAULT_ANGLES, pose_keypoints=None) -> IO.NodeOutput:
        h, w = int(plate.shape[1]), int(plate.shape[2])
        x_min, x_max = sorted((x_min, x_max))
        size_min, size_max = sorted((size_min, size_max))
        reference = _best_person(pose_keypoints)
        angle_list = parse_angles(angles)
        items = plan(seed, candidates, vary, pose, reference is not None, x_min, x_max, size_min, size_max, len(angle_list))

        kps, prompts, seeds, labels = [], [], [], []
        for it in items:
            if vary == "off":
                kps.append(pose_keypoints or [])
                prompts.append(prompt)
                labels.append(f"#{it['index']} seed {it['seed']}: same place and pose as the reference")
            else:
                if it["angle"] is not None:
                    yaw, words = angle_list[it["angle"]]
                    arrs = _synthetic(yaw, it["arms"])
                elif reference is not None:
                    arrs, words = reference[0], ""
                    if it["mirror"]:
                        arrs = _mirror(arrs)
                else:  # "placement only" with no pose photo: a plain front-facing stance
                    arrs, words = _synthetic(0, 0), ""
                arrs = _place(arrs, w, h, it["x"] * w, it["size"] * h, horizon)
                kps.append(_frame(arrs, w, h))
                side = ["far left", "left", "centre", "right", "far right"][it["slot"]]
                what = words or ("reference pose, mirrored" if it["mirror"] else "reference pose" if reference else "standing")
                prompts.append(f"{prompt}, {words}" if words else prompt)
                labels.append(f"#{it['index']} seed {it['seed']}: {side}, {it['size']:.0%} of the height, {what}")
            seeds.append(it["seed"])
        folder = f"9b_{seed}"
        return IO.NodeOutput(kps, prompts, seeds, [folder] * len(items), labels)

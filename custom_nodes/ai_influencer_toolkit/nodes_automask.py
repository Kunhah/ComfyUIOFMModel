"""Auto Mask: where the character goes on a location plate, without painting it by hand.

Workflow 09b regenerates only inside a mask on the plate. This node builds that mask, picking per
image, in this order:

  1. painted  -- a mask painted on the plate in MaskEditor always wins;
  2. replace  -- the plate already shows a person (SDPose finds a confident body that sits on the
                 BiRefNet subject): mask = that person's cut-out, plus the new pose's body shape
                 scaled to that person (shoulder width) and anchored at their neck, so she takes
                 their place and the new pose's arms/legs aren't clipped by the old silhouette;
  3. pose     -- empty plate: the pose reference's body shape, at the same relative position it
                 has in the pose photo;
  4. fixed    -- no usable pose either: a person-sized area at `fixed_position`.

The body shape is drawn from the 18 OpenPose keypoints (limbs as thick strokes, a torso polygon,
a head ellipse) with widths relative to the body's size, then grown by `margin` so the model has
room. GrowMask after it still feathers the seam. Everything is cheap tensor/PIL work on the CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from comfy_api.latest import IO

MODES = ["auto", "painted", "replace person", "pose", "fixed"]
POSITIONS = ["center", "left third", "right third"]

# OpenPose-18: 0 nose, 1 neck, 2 R shoulder, 3 R elbow, 4 R wrist, 5 L shoulder, 6 L elbow, 7 L wrist,
# 8 R hip, 9 R knee, 10 R ankle, 11 L hip, 12 L knee, 13 L ankle, 14 R eye, 15 L eye, 16 R ear, 17 L ear
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0)]
TORSO = [2, 5, 11, 8]
SCORE = 0.3        # a keypoint counts when its score is at least this
MIN_POINTS = 6     # ...and a person counts when at least this many body keypoints do


def _best_person(keypoints) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """(xy [18,2] in pixels, valid [18] bool, canvas w, canvas h) of the most complete person."""
    if not keypoints:
        return None
    frame = keypoints[0]
    best = None
    for person in frame.get("people", []):
        flat = np.asarray(person.get("pose_keypoints_2d") or [], dtype=np.float32)
        if flat.size < 54:
            continue
        kp = flat[:54].reshape(18, 3)
        valid = (kp[:, 2] >= SCORE) & (kp[:, 0] > 0) & (kp[:, 1] > 0)
        if valid.sum() >= MIN_POINTS and (best is None or valid.sum() > best[1].sum()):
            best = (kp[:, :2].copy(), valid)
    if best is None:
        return None
    return best[0], best[1], int(frame["canvas_width"]), int(frame["canvas_height"])


def _body_size(xy: np.ndarray, valid: np.ndarray) -> float:
    ys, xs = xy[valid, 1], xy[valid, 0]
    return float(max(ys.max() - ys.min(), xs.max() - xs.min(), 1.0))


def _shoulders(xy: np.ndarray, valid: np.ndarray) -> float | None:
    if valid[2] and valid[5]:
        return float(np.hypot(*(xy[2] - xy[5])))
    return None


def _draw_body(xy: np.ndarray, valid: np.ndarray, w: int, h: int, margin_px: float) -> np.ndarray:
    """A filled body silhouette from keypoints, grown by margin_px. Returns float [h, w] in 0..1."""
    size = _body_size(xy, valid)
    limb = max(4.0, 0.11 * size) + 2 * margin_px
    canvas = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(canvas)
    pts = [tuple(map(float, p)) for p in xy]
    for a, b in LIMBS:
        if valid[a] and valid[b]:
            d.line([pts[a], pts[b]], fill=255, width=int(limb))
    for i in range(18):  # round the joints so strokes don't leave notches
        if valid[i]:
            r = limb / 2
            d.ellipse([pts[i][0] - r, pts[i][1] - r, pts[i][0] + r, pts[i][1] + r], fill=255)
    torso = [pts[i] for i in TORSO if valid[i]]
    if len(torso) >= 3:
        d.polygon(torso, fill=255)
    head = [i for i in (0, 14, 15, 16, 17) if valid[i]]
    if head:
        hx = float(np.mean([pts[i][0] for i in head]))
        hy = float(np.mean([pts[i][1] for i in head]))
        r = 0.13 * size + margin_px  # generous: hair and head tilt
        d.ellipse([hx - r, hy - 1.2 * r, hx + r, hy + r], fill=255)
    # feet: the ankles are the last body points; extend a little below so shoes fit
    for ank in (10, 13):
        if valid[ank]:
            x, y = pts[ank]
            r = limb / 2
            d.ellipse([x - r, y - r / 2, x + r, y + 1.5 * r], fill=255)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _fixed(w: int, h: int, position: str, margin_px: float) -> np.ndarray:
    cx = {"center": 0.5, "left third": 0.3, "right third": 0.7}.get(position, 0.5) * w
    half_w = 0.17 * h + margin_px
    canvas = Image.new("L", (w, h), 0)
    ImageDraw.Draw(canvas).ellipse([cx - half_w, 0.1 * h, cx + half_w, 1.02 * h], fill=255)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _resize_mask(mask: torch.Tensor, w: int, h: int) -> torch.Tensor:
    m = mask[0] if mask.dim() == 3 else mask
    return F.interpolate(m[None, None].float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0]


def _grow(mask: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return mask
    k = 2 * px + 1
    return F.max_pool2d(mask[None, None], kernel_size=k, stride=1, padding=px)[0, 0]


class AIInfluencerAutoMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="AIInfluencerAutoMask",
            display_name="Auto Mask (AI Influencer)",
            category="ai_influencer",
            search_aliases=["automatic mask", "person mask", "inpaint mask"],
            description=(
                "Builds the 'where she goes' mask for a location plate: your painted mask if there is one; "
                "otherwise the person already in the photo plus the new pose's shape; otherwise the pose "
                "reference's shape; otherwise a centered person-sized area."
            ),
            inputs=[
                IO.Image.Input("plate", tooltip="The location plate, at the size it is generated at."),
                IO.Mask.Input("painted_mask", optional=True, tooltip="Load Image's MASK output: used when something is painted."),
                IO.Custom("POSE_KEYPOINT").Input("pose_keypoints", optional=True, tooltip="SDPose keypoints of the POSE reference."),
                IO.Custom("POSE_KEYPOINT").Input("plate_keypoints", optional=True, tooltip="SDPose keypoints of the PLATE (finds a person already in it)."),
                IO.Mask.Input("plate_subject", optional=True, tooltip="Remove Background mask of the PLATE (that person's cut-out)."),
                IO.Combo.Input("mode", options=MODES, default="auto", tooltip="auto = painted > replace person > pose > fixed."),
                IO.Float.Input("margin", default=4.0, min=0.0, max=20.0, step=0.5, tooltip="Extra room around the body, % of the short side."),
                IO.Combo.Input("fixed_position", options=POSITIONS, default="center", tooltip="Where the fallback area goes."),
            ],
            outputs=[
                IO.Mask.Output(display_name="mask"),
                IO.Image.Output(display_name="preview", tooltip="The plate with the mask in red, to check it."),
                IO.String.Output(display_name="method"),
            ],
        )

    @classmethod
    def execute(cls, plate, painted_mask=None, pose_keypoints=None, plate_keypoints=None, plate_subject=None,
                mode="auto", margin=4.0, fixed_position="center") -> IO.NodeOutput:
        h, w = int(plate.shape[1]), int(plate.shape[2])
        margin_px = margin / 100.0 * min(h, w)
        mask, method = None, ""

        if mode in ("auto", "painted") and painted_mask is not None:
            pm = _resize_mask(painted_mask, w, h)
            if pm.max() > 0.5 and pm.mean() < 0.98:  # LoadImage gives all-zero for a photo with nothing painted
                mask, method = pm, "painted"

        pose = _best_person(pose_keypoints)
        person = _best_person(plate_keypoints)
        subject = _resize_mask(plate_subject, w, h) if plate_subject is not None else None
        if person is not None:
            pxy, pvalid, pcw, pch = person
            pxy = pxy * np.array([w / pcw, h / pch], dtype=np.float32)
            # a real person, not a hallucinated skeleton: most confident points sit on the subject cut-out
            if subject is not None:
                inside = [subject[int(min(max(y, 0), h - 1)), int(min(max(x, 0), w - 1))] > 0.5 for x, y in pxy[pvalid]]
                if np.mean(inside) < 0.6:
                    person = None
            if person is not None:
                person = (pxy, pvalid)

        if mask is None and mode in ("auto", "replace person") and person is not None:
            pxy, pvalid = person
            region = torch.from_numpy(_draw_body(pxy, pvalid, w, h, 3 * margin_px))  # generous box around them
            old = (subject * (region > 0.5)) if subject is not None else torch.from_numpy(_draw_body(pxy, pvalid, w, h, margin_px))
            new = torch.zeros_like(old)
            if pose is not None:
                qxy, qvalid, qcw, qch = pose
                qxy = qxy * np.array([w / qcw, h / qch], dtype=np.float32)
                ps, qs = _shoulders(pxy, pvalid), _shoulders(qxy, qvalid)
                scale = (ps / qs) if ps and qs else _body_size(pxy, pvalid) / _body_size(qxy, qvalid)
                anchor_p = pxy[1] if pvalid[1] else pxy[pvalid].mean(0)
                anchor_q = qxy[1] if qvalid[1] else qxy[qvalid].mean(0)
                new = torch.from_numpy(_draw_body((qxy - anchor_q) * scale + anchor_p, qvalid, w, h, margin_px))
            mask = torch.maximum(_grow(old, int(margin_px)), new)
            method = "replace person" + (" + pose" if pose is not None else "")

        if mask is None and mode in ("auto", "pose") and pose is not None:
            qxy, qvalid, qcw, qch = pose
            qxy = qxy * np.array([w / qcw, h / qch], dtype=np.float32)
            mask, method = torch.from_numpy(_draw_body(qxy, qvalid, w, h, margin_px)), "pose"

        if mask is None:
            mask, method = torch.from_numpy(_fixed(w, h, fixed_position, margin_px)), f"fixed ({fixed_position})"
            if mode not in ("auto", "fixed"):
                method += f" -- '{mode}' had nothing to work with"

        mask = mask.clamp(0, 1).float()
        img = plate[0].float().cpu()
        red = torch.tensor([1.0, 0.1, 0.1])
        preview = img * (1 - 0.45 * mask[..., None]) + red * (0.45 * mask[..., None])
        return IO.NodeOutput(mask[None], preview[None], f"{method}: {mask.mean().item() * 100:.0f}% of the image")

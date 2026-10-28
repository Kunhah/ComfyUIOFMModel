"""Let core `LoadImage` see project images kept in subfolders of `input/ai_influencer/`.

Core `LoadImage` only lists files at the top level of `input/`, so a workflow that points at
`ai_influencer/<project>/dataset/41.jpg` shows the file as a missing input in the frontend even
though it exists and runs (the node's VALIDATE_INPUTS already accepts subfolder paths). This
wraps `LoadImage.INPUT_TYPES` to append those paths to the dropdown. Only file names are listed;
nothing is opened.

It also backs web/input_helper.js, which fixes the other kind of missing input: a workflow saved on
another computer names pictures this one doesn't have. `GET /ai_influencer/inputs` gives it this
computer's character project (to map `ai_influencer/<other>/...` onto) and the placeholder picture
it selects instead when there is no such file, so the node asks for a picture rather than erroring.
Load Video gets the same treatment with a placeholder video (Workflow 14's driving video).
"""

import logging
import os

import folder_paths
import nodes

try:
    from . import env_config
except ImportError:
    import env_config

_SUBTREE = "ai_influencer"
PLACEHOLDER = "ai_influencer_choose_a_picture.png"  # top level of input/, made on startup
VIDEO_PLACEHOLDER = "ai_influencer_choose_a_video.mp4"  # the same picture as a one-second video, for Load Video


def _project_images() -> list[str]:
    input_dir = folder_paths.get_input_directory()
    root = os.path.join(input_dir, _SUBTREE)
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            found.append(os.path.relpath(os.path.join(dirpath, f), input_dir).replace(os.sep, "/"))
    return folder_paths.filter_files_content_types(found, ["image"])


def install() -> None:
    cls = nodes.LoadImage
    if getattr(cls.INPUT_TYPES, "_ai_influencer_patched", False):
        return
    original = cls.INPUT_TYPES.__func__

    def INPUT_TYPES(s):
        types = original(s)
        files, opts = types["required"]["image"]
        types["required"]["image"] = (sorted(set(files) | set(_project_images())), opts)
        return types

    INPUT_TYPES._ai_influencer_patched = True
    cls.INPUT_TYPES = classmethod(INPUT_TYPES)


def default_project() -> str:
    """AI_INFLUENCER_DEFAULT_PROJECT when its input folder exists, else the only project there is, else ""."""
    root = os.path.join(folder_paths.get_input_directory(), _SUBTREE)
    try:
        projects = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    except FileNotFoundError:
        return ""
    name = env_config.get("AI_INFLUENCER_DEFAULT_PROJECT")
    if name in projects:
        return name
    return projects[0] if len(projects) == 1 else ""


def make_placeholder() -> None:
    """A grey picture that says what to do, selected in Load Image nodes whose file isn't on this computer,
    and the same as a short video for Load Video nodes."""
    path = os.path.join(folder_paths.get_input_directory(), PLACEHOLDER)
    if not os.path.exists(path):
        _draw_placeholder("Choose a picture", "click here, or drag a photo onto this node").save(path)
    video = os.path.join(folder_paths.get_input_directory(), VIDEO_PLACEHOLDER)
    if not os.path.exists(video):
        try:  # an encoder problem must not take the picture placeholder down with it
            _write_still_video(_draw_placeholder("Choose a video", "click upload, or drag a video onto this node"), video)
        except Exception as e:
            logging.warning("AI Influencer: could not make %s: %s", VIDEO_PLACEHOLDER, e)


def _draw_placeholder(title: str, hint: str):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (768, 768), (58, 58, 62))
    draw = ImageDraw.Draw(img)
    try:
        big, small = ImageFont.load_default(size=56), ImageFont.load_default(size=30)
    except TypeError:  # Pillow < 10.1 has one fixed-size font
        big = small = ImageFont.load_default()
    draw.rounded_rectangle((40, 40, 728, 728), radius=36, outline=(150, 150, 158), width=6)
    draw.text((384, 330), title, font=big, fill=(235, 235, 240), anchor="mm")
    draw.text((384, 420), hint, font=small, fill=(190, 190, 198), anchor="mm")
    os.makedirs(folder_paths.get_input_directory(), exist_ok=True)
    return img


def _write_still_video(img, path: str, seconds: int = 1, fps: int = 8) -> None:
    import av

    with av.open(path + ".tmp", mode="w", format="mp4") as out:
        stream = out.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = img.width, img.height, "yuv420p"
        frame = av.VideoFrame.from_image(img)
        for _ in range(seconds * fps):
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
    os.replace(path + ".tmp", path)


def install_routes() -> None:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/ai_influencer/inputs")
    async def inputs_info(request):
        return web.json_response({"project": default_project(), "placeholder": PLACEHOLDER,
                                  "video_placeholder": VIDEO_PLACEHOLDER})

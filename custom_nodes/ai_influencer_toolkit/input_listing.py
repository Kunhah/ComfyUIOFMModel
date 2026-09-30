"""Let core `LoadImage` see project images kept in subfolders of `input/ai_influencer/`.

Core `LoadImage` only lists files at the top level of `input/`, so a workflow that points at
`ai_influencer/<project>/dataset/41.jpg` shows the file as a missing input in the frontend even
though it exists and runs (the node's VALIDATE_INPUTS already accepts subfolder paths). This
wraps `LoadImage.INPUT_TYPES` to append those paths to the dropdown. Only file names are listed;
nothing is opened.
"""

import os

import folder_paths
import nodes

_SUBTREE = "ai_influencer"


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

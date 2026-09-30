"""Shared project-folder layout and append-only generation log used by the Save & Log nodes
and the character loader.

ComfyUI sandboxes file I/O for nodes: `LoadImage` only browses `input/`, and the image/video
save helpers (`folder_paths.get_save_image_path`) refuse to write outside `output/`. Rather than
bypassing that (which the ComfyUI maintainers explicitly guard against, see
folder_paths.get_save_image_path), the project layout is split across the two sandboxes:

    input/ai_influencer/<project>/          -- things you feed INTO generation
        references/{canonical,candidate}/      canonical identity photos / rejected candidates
        poses/                                  pose references
        scenes/                                 scene / style references
        dataset/                                LoRA training images (for checkpoint scoring)

    output/projects/<project>/              -- things generation PRODUCES
        character.json                         non-secret character description (see README)
        prompts/                               optional saved reusable prompt text
        outputs/images/{candidates,final,misc}/
        outputs/videos/

    logs/generations.jsonl                  -- one JSON object per line, one per API call.
                                                Kept outside input/output since it isn't media.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

import folder_paths

try:
    from . import env_config
except ImportError:  # the tools/ scripts import this file as a top-level module
    import env_config

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")

OUTPUT_STAGE_SUBDIR: dict[str, str] = {
    "character_candidate": os.path.join("outputs", "images", "candidates"),
    "character_canonical": os.path.join("outputs", "images", "final"),
    "character_sheet": os.path.join("outputs", "images", "sheets"),
    "krea_candidate": os.path.join("outputs", "images", "candidates"),
    "gpt_refined": os.path.join("outputs", "images", "final"),
    "local_candidate": os.path.join("outputs", "images", "candidates"),
    "local_final": os.path.join("outputs", "images", "final"),
    "misc_image": os.path.join("outputs", "images", "misc"),
    "seedance_video": os.path.join("outputs", "videos"),
    "misc_video": os.path.join("outputs", "videos"),
}


def sanitize_project_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        name = env_config.get("AI_INFLUENCER_DEFAULT_PROJECT")
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("._")
    return cleaned or "default"


def get_project_input_dir(project: str) -> str:
    """Where reference images for this project should be uploaded/copied so LoadImage can see them."""
    d = os.path.join(folder_paths.get_input_directory(), "ai_influencer", project)
    for sub in ("references/canonical", "references/candidate", "poses", "scenes", "dataset"):
        os.makedirs(os.path.join(d, *sub.split("/")), exist_ok=True)
    return d


def get_project_output_dir(project: str) -> str:
    d = os.path.join(folder_paths.get_output_directory(), "projects", project)
    os.makedirs(os.path.join(d, "prompts"), exist_ok=True)
    for sub in OUTPUT_STAGE_SUBDIR.values():
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


def ensure_project(project: str) -> tuple[str, str]:
    """Create both sandboxed halves of a project's folder tree. Returns (input_dir, output_dir)."""
    input_dir = get_project_input_dir(project)
    output_dir = get_project_output_dir(project)
    character_json = os.path.join(output_dir, "character.json")
    if not os.path.exists(character_json):
        with open(character_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": project,
                    "trigger": "",
                    "permanent_features": [],
                    "variable_features": [],
                    "default_generation_preferences": {},
                },
                f,
                indent=2,
            )
    return input_dir, output_dir


def character_json_path(project: str) -> str:
    return os.path.join(get_project_output_dir(project), "character.json")


def output_filename_prefix(project: str, stage: str, basename: str, subfolder: str = "") -> str:
    """A filename_prefix relative to ComfyUI's output/ directory, safe to pass straight to
    folder_paths.get_save_image_path / ImageSaveHelper / SaveVideo's own path resolution.
    `subfolder` is one extra folder level inside the stage's folder (sanitized to a single name)."""
    sub = OUTPUT_STAGE_SUBDIR.get(stage, os.path.join("outputs", "images", "misc"))
    ensure_project(project)
    basename = _SAFE_NAME_RE.sub("_", basename or "gen").strip("._") or "gen"
    extra = _SAFE_NAME_RE.sub("_", subfolder or "").strip("._")
    return os.path.join("projects", project, sub, *([extra] if extra else []), basename)


def get_logs_path() -> str:
    log_dir = os.path.join(folder_paths.base_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "generations.jsonl")


def make_prompt_id(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:10]


def provider_spend_to_date(provider: str) -> float:
    """Sum of estimated_cost_usd logged for a provider. Estimates, not fal's own accounting:
    a floor for what you've spent there, used by the tools' --budget guard."""
    path = get_logs_path()
    if not os.path.isfile(path):
        return 0.0
    total = 0.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("provider") == provider and isinstance(rec.get("estimated_cost_usd"), (int, float)):
                total += rec["estimated_cost_usd"]
    return round(total, 4)


def check_budget(provider: str, additional: float, budget: float) -> None:
    """Raise SystemExit when this spend would push the provider's logged total past `budget`."""
    if budget <= 0:
        return
    spent = provider_spend_to_date(provider)
    if spent + additional > budget:
        raise SystemExit(
            f"Budget guard: {provider} has ~${spent:.2f} logged, this would add ${additional:.2f}, "
            f"over the ${budget:.2f} budget. Raise --budget (or FAL_BUDGET_USD) if that's intended."
        )


def append_generation_log(record: dict) -> None:
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
    with open(get_logs_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Character Sheet Studio

Welcome. This folder helps an AI coding agent turn your reference images into a
character sheet.

Copy your dataset images into the four folders inside anchors/ — face/, expressions/,
full-body/, and upper-body/ (each folder has a note saying what belongs in it). Then say
which sheet you want: face-turnaround, expression-grid, full-body-360, or upper-body.
The agent looks at every image before generating and rejects unusable ones with a reason.
Dragging images into the chat works too.

You can simply say: Create my character sheet. If you do not know which sheet to start
with, say that and the agent will recommend one.

Any agent working in this folder must follow the workflow defined in
.claude/skills/character-sheet-studio/SKILL.md, including its non-negotiable
state_rules block. It is the single source of truth for this framework. Accepted sheets
are saved in sheets/.

Portable supported inputs are static PNG and JPEG. WebP and HEIC/HEIF are a best-effort
convenience on macOS only; Windows refuses them before upload or cost with conversion
guidance.

The shipped key template is .env.example. When a key is first saved, the agent creates
.env by COPYING .env.example only if .env does not already exist, then writes the key
into that local file. Never overwrite an existing .env. The agent never prints, cats,
sources, or executes .env; it parses exactly the two named values WAVESPEED_API_KEY and
MODEL as data. Never use curl verbose or trace flags.
A future update unzipped over this folder cannot overwrite the key because .env is local
and is created only when absent.

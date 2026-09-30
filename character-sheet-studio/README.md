# Character Sheet Studio

A ready-to-use folder that turns your reference images into character sheets —
reference boards that help keep an AI subject consistent across generations.

It is designed for AI coding agents that can read files, run shell commands, and inspect
images. Exact capabilities vary by agent. The agent looks at your reference images,
builds the selected template prompt, generates the sheet through your WaveSpeed account
(or guides you through the manual Google AI Studio fallback), and delivers it with a
short self-check list. You judge the result — one sentence re-rolls a sheet you do not
like.

The same approach works for a person or an object. The prompt uses the reference images as
the only identity source, so it never invents a written description of the subject.

## Supported images

Portable supported formats are static PNG and JPEG. WebP and HEIC/HEIF are a best-effort
convenience on macOS only; Windows refuses them before upload or cost and gives
zero-install JPEG conversion guidance.

Large ordinary PNG/JPEG images are optimized automatically. Corrupt, unsupported, or
extreme images are named and rejected before anything is uploaded or paid. Your original
files are not overwritten or deleted.

## Install in 3 steps

1. Download the zip file.
2. Extract it to a location you can find again.
3. Open the extracted folder in your AI coding agent.

That is the setup. The agent reads its instructions from this folder automatically
(CLAUDE.md for Claude and AGENTS.md for other agents; both point at the same workflow).

## What you need

- Three to six clear reference images of one subject.
- A WaveSpeed API key for the automatic path, or a Google account for the separate manual
  Google AI Studio fallback.
- A clear front view and at least one angled view. Use full views when you want a body
  sheet. Avoid blurry, cropped, group, or watermarked images.

The safe shipped key template is .env.example. When a key is first saved, the agent
creates .env by COPYING .env.example only if .env does not already exist, then writes
the key into that local file. The agent never prints, cats, sources, or executes .env;
it parses exactly the two named values WAVESPEED_API_KEY and MODEL as data. Never use
curl verbose or trace flags.
A future update unzipped over this folder cannot overwrite the key because .env is local
and is created only when absent.

The model is selectable from config.json. The default is gpt-image-2; tell the agent to
use any model listed there, or set MODEL in the local .env.

## Start the wizard

Copy your dataset images into the four folders inside anchors/:

- face/ for face-turnaround
- expressions/ for expression-grid
- full-body/ for full-body-360
- upper-body/ for upper-body

Each folder has a short note saying what belongs there, and you only need to fill the
folders for the sheets you want. Then say:

> Create my character sheet.

You can name a sheet in the same message: face-turnaround, expression-grid,
full-body-360, or upper-body. If you are unsure, start with face-turnaround. Dragging
images straight into the chat works too.

Ready-to-paste prompts live in `PROMPTS.md` — including the one-message complete-board
prompt and three universal standalone JSON prompts that work in any image tool.

The agent guides you through one sheet at a time. Each finished sheet lands in sheets/
together with a short self-check list. A complete character board has all four sheets.

## Good to know

- The framework is free. Image generation is paid by your WaveSpeed balance: expect a
  few cents per sheet. A complete board is four paid generations; every re-roll you
  request is one more.
- Reference images are uploaded to WaveSpeed to generate the sheet.
- If this folder lives in a cloud-synced location such as OneDrive or iCloud, your
  anchors and results sync there too.
- Google AI Studio is a manual, best-effort fallback, not a guaranteed free equivalent.
  Availability and limits can vary.
- The [WaveSpeed signup link](https://wavespeed.ai/?ref=matrix) is a referral link.

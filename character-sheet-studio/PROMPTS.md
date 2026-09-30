# Character Sheet Studio — Ready-to-use Prompts

Two kinds of prompts in this file:

- **Part A — drive the framework.** Paste these into the chat after opening this folder
  in your AI coding agent. The agent does everything: checks your images, generates
  through your WaveSpeed account, and delivers the sheets.
- **Part B — universal standalone prompts.** Three complete JSON prompts you can paste
  directly into ANY image tool (Google AI Studio, or your own provider) together with
  your reference images — no agent needed.

---

## Part A — drive the framework

### The complete board (all four sheets, one message)

```
I opened the Character Sheet Studio folder. My reference images are in your anchors
folders — face, expressions, full body, and upper body — and my WaveSpeed key is in
the .env file.

Build the complete character board in one go: all four sheets — face turnaround,
expression grid, full body 360, and upper body. Generate everything in 16:9 at 4K
resolution with the GPT Image 2 model over the WaveSpeed API. Check my images first
and tell me what the whole board will cost before you start. When you are done, I
want all four sheets in the sheets folder — I will judge them myself and tell you if
one needs a re-roll.
```

When the agent names the price, answer:

```
Go.
```

### One single sheet

```
My face images are in your anchors face folder. Build only the face turnaround sheet.
Tell me the cost first.
```

(Swap the sheet name: `expression grid` · `full body 360` · `upper body`.)

### Not sure which images you need?

```
Which images do you need from me for the face turnaround, and which folder do they go
in?
```

### Re-roll a sheet you do not like

```
The full body 360 has a duplicated profile view — re-roll that sheet with your fix.
```

(Name what you see; the agent maps it to the template and re-rolls once. Each re-roll
is one paid generation.)

### Switch the image model

```
Generate this one with nano-banana-pro instead, I want to compare the models.
```

(Available: gpt-image-2 · nano-banana-pro · nano-banana-2 · seedream · seedream-v5-pro)

### A product instead of a person

```
This is not a person — it is a product. Same framework, same rules: build me a
turnaround sheet for it. The images are in your anchors face folder.
```

### Show what was actually sent

```
Show me the exact prompt you sent to the image model.
```

---

## Part B — universal standalone prompts (any image tool)

How to use, for all three prompts:

1. Attach **6–8 reference images** of ONE subject. The **first image is the identity
   lock** (the clearest front view of the face — or the clearest full view of a
   product).
2. Fill in the `INFO` placeholders (NAME, ROLE, STYLE, HEIGHT) — or delete the lines
   you don't need.
3. Copy the complete JSON as your prompt. Do NOT add any description of the subject —
   the reference images are the only identity source. Extra words cause drift.
4. Settings: **16:9, highest resolution available** (4K if your tool offers it).
5. Works for a person, a product, or a mascot. For an object, the model reads
   "expressions" as detail views automatically — you can also swap that section label
   to `DETAIL VIEWS`.

### PROMPT 1 — THE PRODUCTION BOARD (full reference board)

```json
{
  "task": "Character sheet poster, one single canvas, 16:9, professional character design reference board",
  "identity": {
    "source": "the attached reference images only",
    "rule": "The subject is the subject in the reference images. Copy the appearance exactly from the references in every panel. Do not invent features."
  },
  "style": "Clean professional character-design reference board on an off-white paper background, thin dark hairline rules separating the sections, small uppercase sans-serif section labels, the look of a film production character bible page, soft even lighting in every photo panel",
  "layout": {
    "header_top_left": {
      "title": "YOUR CHARACTER NAME",
      "subtitle": "CHARACTER SHEET",
      "info_lines": ["NAME: YOUR CHARACTER NAME", "ROLE: YOUR ROLE", "STYLE: YOUR STYLE", "HEIGHT: YOUR HEIGHT"]
    },
    "left_section": {
      "label": "TURNAROUND",
      "content": "the same subject head to toe in the exact look from the reference images, three full views side by side labelled FRONT, SIDE, BACK, identical scale, identical framing, standing on one shared baseline"
    },
    "middle_top_section": {
      "label": "FACIAL EXPRESSIONS",
      "content": "six head-and-shoulders portraits in two rows of three, labelled NEUTRAL, SOFT SMILE, OPEN SMILE, SERIOUS, SURPRISED, WINK, all frontal, identical framing and scale"
    },
    "middle_bottom_section": {
      "label": "OUTFIT & DETAILS",
      "content": "the individual items from the reference images laid out as small isolated product shots with one short label each"
    },
    "right_top_section": {
      "label": "COLOR PALETTE",
      "content": "six flat rectangular color swatches sampled from the subject, each with its hex value printed under it"
    },
    "right_bottom_section": {
      "label": "SETTING",
      "content": "four small mood photographs of locations that fit the subject's world, empty of people"
    }
  },
  "consistency": "Every photo panel shows the same subject from the reference images at consistent quality. The turnaround views share one scale. The portrait panels share one scale.",
  "text_rules": "Render only the exact texts specified above, spelled exactly as written, plus the hex values under the palette swatches. No other text, no watermark, no logo.",
  "must_avoid": [
    "No spelling errors in any label.",
    "No cropping of the subject at a panel edge.",
    "No invented extra text blocks or paragraphs.",
    "No borders or drop shadows around photo panels."
  ]
}
```

### PROMPT 2 — THE STUDIO MODEL SHEET (clean, minimal text)

```json
{
  "task": "Studio model sheet, one single canvas, 16:9, clean character turnaround reference",
  "identity": {
    "source": "the attached reference images only",
    "rule": "The subject is the subject in the reference images. Copy the appearance exactly from the references in every panel. Do not invent features."
  },
  "style": "Neutral seamless grey studio background across the whole canvas, flat even softbox lighting, no hard shadows, photographic, consistent scale in every panel",
  "layout": {
    "hero": {
      "position": "left third of the canvas",
      "content": "the subject seen straight from the front, large, head to mid-body"
    },
    "grid": {
      "position": "right two thirds of the canvas, two rows",
      "cells": [
        "the subject turned three-quarters toward frame-left",
        "the subject turned three-quarters toward frame-right",
        "the subject in full left profile, only the near side visible",
        "the subject in full right profile, only the near side visible",
        "the subject seen mostly from behind, turned slightly toward frame-left",
        "the subject seen straight from behind"
      ]
    }
  },
  "consistency": "Every panel shows the same subject at the same scale, framed the same way, with the eyes on one shared horizontal line across the sheet. Every panel shows a different angle and no two panels repeat or mirror the same view.",
  "text_rules": "No text, no labels, no watermark anywhere on the sheet.",
  "must_avoid": [
    "No borders and no drop shadows between the panels.",
    "Do not crop the subject at a panel edge.",
    "No duplicated or mirrored views."
  ]
}
```

### PROMPT 3 — THE ACTION BOARD (poses and life)

```json
{
  "task": "Character action board, one single canvas, 16:9, dynamic pose reference sheet",
  "identity": {
    "source": "the attached reference images only",
    "rule": "The subject is the subject in the reference images. Copy the appearance exactly from the references in every panel. Do not invent features."
  },
  "style": "Clean light studio background across the whole canvas, soft directional lighting, photographic, magazine-quality, small uppercase sans-serif labels under each panel",
  "layout": {
    "top_strip": {
      "label": "TURNAROUND",
      "content": "three matching full views of the subject side by side labelled FRONT, SIDE, BACK, identical scale, one shared baseline"
    },
    "main_grid": {
      "label": "IN MOTION",
      "content": "six larger panels in two rows of three showing the subject in natural dynamic moments, labelled WALKING, SITTING, LAUGHING, LOOKING BACK, LEANING, CLOSE-UP, each from a different camera angle, the look and identity identical to the reference images in every panel"
    }
  },
  "consistency": "Every panel shows the same subject from the reference images. The turnaround shares one scale. The motion panels can vary in framing but never in identity or wardrobe.",
  "text_rules": "Render only the labels specified above, spelled exactly as written. No other text, no watermark, no logo.",
  "must_avoid": [
    "No spelling errors in any label.",
    "No cropping of the subject's important edges.",
    "No change of outfit, hair or identity between panels.",
    "No borders or drop shadows around the panels."
  ]
}
```

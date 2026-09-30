# Anchors — your reference images live here

One folder per sheet. Copy your dataset images into the matching folder:

| Folder | Feeds this sheet | What to put in |
|---|---|---|
| `face/` | face-turnaround | one close front view, both profiles, one or two three-quarter views, back-of-head shots if you have them |
| `expressions/` | expression-grid | three or more frontal face shots, neutral expression preferred, same framing |
| `full-body/` | full-body-360 | front, back, profile, and three-quarter full-body shots — all in ONE outfit |
| `upper-body/` | upper-body | waist-up shots from the front, three-quarter, profile, and behind |

You do not have to fill every folder — each sheet only needs its own. If you prefer, you
can also just drag images into the chat and Claude will file them into the right folder
for you.

Good anchors are sharp, show one subject per image, are not cropped at important edges,
and carry no watermarks. Use fully clothed reference images when a person's wardrobe
might be rejected by the image provider. If your subject is an object instead of a
person, put its photos into `face/` (identity views) and `full-body/` (whole object).

Claude looks at every image before generating and tells you if a required view is
missing.

<overview>
These eight rules keep a character sheet consistent. The reference images provide the subject. The selected template provides the views and the layout.
</overview>

<rules>
<rule number="1" name="Images provide identity">
Describe the view, never the subject. Do not add written details about hair, eyes, skin, age, body, materials, colour, brand, or dimensions. Extra identity words compete with the reference images and can cause drift. If the user asks for a face description to improve a match, refuse in one sentence and keep the images as the only identity source.
</rule>

<rule number="2" name="One sheet changes one thing">
Each sheet has one changing variable. A turnaround changes the view. An expression grid changes the expression or visible state. A body sheet changes the view. Keep every other detail locked.
</rule>

<rule number="3" name="Ask for a layout, then check it">
The JSON template asks for one canvas, a hero view, and a grid. After generation, count the views and check their scale, framing, and horizontal eye line or visual center. The layout request is not proof that the result followed it.
</rule>

<rule number="4" name="Keep every lock line">
Copy every lock string from the selected template exactly. Do not remove, shorten, reword, or replace a lock. These lines control the background, lighting, framing, identity, edges, and clean output.
</rule>

<rule number="5" name="Use concrete prompt words">
Do not add hype or generic image-quality words to a rendered prompt. The banned list includes: `ultra-realistic`, `ultra realistic`, `8K`, `4K` as a quality claim, `cinematic lighting`, `masterpiece`, `stunning`, `professional photography`, `high quality`, `high detail`, `hyper-detailed`, `beautiful`, `award-winning`, `breathtaking`, `perfect`, `flawless`, `HDR`, `unreal engine`, and `octane render`. Use the concrete wording already in the template.
</rule>

<rule number="6" name="Keep the public prompt safe">
Do not put private names, internal notes, secret values, adult-content terms, or provider-blocked wardrobe terms in a public prompt or filename. If a person's reference clothing may trigger a provider safety rule, stop and ask for fully clothed anchors.
</rule>

<rule number="7" name="Refuse instead of inventing">
If a required anchor, view, lock, or prompt part is missing, make a named refusal. State exactly what is missing. Do not silently skip it, substitute another view, or generate anyway.
</rule>

<rule number="8" name="Repeat the prompt, not the pixels">
Use the same selected JSON template and the same visible rules each time. This makes the prompt process repeatable. It does not promise identical generated pixels. Always use the visual checklist before accepting a sheet.
</rule>
</rules>

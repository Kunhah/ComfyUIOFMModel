// "Pod queue": hold Queue presses on this PC, then run them all on a vast.ai pod with one button.
// Server side: ../pod_queue.py. While "Hold for the pod" is on, ComfyUI's own Queue button (batch
// count and seed randomizing included) stores each prompt instead of running it here.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HOLD_KEY = "ai_influencer.pod_queue.hold";
const BASE = "/ai_influencer/pod_queue";

let hold = false;
try { hold = localStorage.getItem(HOLD_KEY) === "1"; } catch {}
let open = false, data = { items: [], run: {}, project: "" }, pollTimer = null;
let editing = null;  // { id, n, tab } while a held job is open on the canvas for editing

const toast = (summary, detail, severity = "info") => {
  try { app.extensionManager?.toast?.add({ severity, summary, detail, life: 2500 }); } catch {}
};
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const post = async (path, body) => (await api.fetchApi(BASE + path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}),
})).json();

function workflowName() {
  try {
    const w = app.extensionManager?.workflow?.activeWorkflow;
    return (w?.filename || w?.path || "workflow").replace(/\.json$/, "");
  } catch { return "workflow"; }
}

// --- the Queue button, intercepted while holding ---------------------------------------------
const originalQueuePrompt = api.queuePrompt.bind(api);
api.queuePrompt = async function (number, promptData, options) {
  if (!hold) return originalQueuePrompt(number, promptData, options);
  const res = await post("/add", { prompt: promptData.output, workflow: promptData.workflow, name: workflowName() });
  if (res.error) throw new Error(res.error);
  toast("Held for the pod", `#${res.n} · ${res.count} waiting` + (res.notes?.length ? ` · ${res.notes.join("; ")}` : ""),
    res.notes?.some(n => !n.startsWith("added")) ? "warn" : "info");
  refresh();
  // an empty prompt_id tells the frontend nothing is running, so it doesn't wait for this job
  return { prompt_id: "", number: 0, node_errors: {} };
};

// --- the panel ---------------------------------------------------------------------------------
const css = document.createElement("style");
css.textContent = `
#aiinf-podq { position:fixed; right:16px; bottom:64px; z-index:1000; font:13px/1.4 system-ui, sans-serif;
  color:var(--fg-color, #ddd); }
#aiinf-podq .pill { display:flex; gap:8px; align-items:center; padding:7px 12px; border-radius:99px; cursor:pointer;
  background:var(--comfy-menu-bg, #202020); border:1px solid var(--border-color, #444); box-shadow:0 2px 10px rgba(0,0,0,.35); user-select:none; }
#aiinf-podq .dot { width:9px; height:9px; border-radius:50%; background:#777; }
#aiinf-podq .dot.on { background:#e0a030; } #aiinf-podq .dot.run { background:#4caf7d; }
#aiinf-podq .panel { width:min(440px, calc(100vw - 32px)); max-height:min(640px, calc(100vh - 140px)); overflow:auto; margin-bottom:8px;
  padding:12px; border-radius:10px; background:var(--comfy-menu-bg, #202020); border:1px solid var(--border-color, #444);
  box-shadow:0 4px 20px rgba(0,0,0,.45); }
#aiinf-podq h3 { margin:0 0 8px; font-size:14px; }
#aiinf-podq label { display:flex; gap:8px; align-items:center; margin:6px 0; }
#aiinf-podq .muted { opacity:.65; font-size:12px; }
#aiinf-podq ul { list-style:none; margin:8px 0; padding:0; border-top:1px solid var(--border-color, #444); }
#aiinf-podq li { display:flex; gap:8px; align-items:flex-start; padding:6px 0; border-bottom:1px solid var(--border-color, #444); }
#aiinf-podq li .t { flex:1; min-width:0; } #aiinf-podq li .s { opacity:.7; font-size:12px; word-break:break-word; }
#aiinf-podq button { font:inherit; color:inherit; background:var(--comfy-input-bg, #2a2a2a); border:1px solid var(--border-color, #555);
  border-radius:6px; padding:5px 10px; cursor:pointer; }
#aiinf-podq button.go { background:#2f6f5e; border-color:#2f6f5e; color:#fff; font-weight:600; }
#aiinf-podq button.x { padding:0 7px; }
#aiinf-podq button:disabled { opacity:.45; cursor:default; }
#aiinf-podq input[type=number] { width:64px; font:inherit; color:inherit; background:var(--comfy-input-bg, #2a2a2a);
  border:1px solid var(--border-color, #555); border-radius:5px; padding:3px 5px; }
#aiinf-podq .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:8px; }
#aiinf-podq pre { max-height:170px; overflow:auto; white-space:pre-wrap; font-size:11.5px; margin:8px 0 0; padding:6px;
  background:var(--comfy-input-bg, #111); border-radius:6px; }
#aiinf-podq .bar { height:6px; border-radius:99px; background:var(--comfy-input-bg, #333); overflow:hidden; margin-top:8px; }
#aiinf-podq .bar div { height:100%; background:#4caf7d; transition:width .4s; }`;
document.head.appendChild(css);

const root = document.createElement("div");
root.id = "aiinf-podq";
document.body.appendChild(root);

let hours = 2, autoDestroy = true;

function render() {
  const items = data.items || [], r = data.run || {};
  const total = r.total || 0, finished = (r.done || 0) + (r.failed || 0);
  const pill = `<div class="pill" data-a="toggle"><span class="dot ${r.running ? "run" : hold ? "on" : ""}"></span>
    Pod queue · ${items.length} held${hold ? " · <b>holding</b>" : ""}${editing ? ` · <b>editing #${editing.n}</b>` : ""}${r.running ? ` · ${esc(r.phase)}` : ""}</div>`;
  if (!open) { root.innerHTML = pill; return; }
  const log = root.querySelector("pre"), keepScroll = log && log.scrollTop + log.clientHeight < log.scrollHeight - 20 ? log.scrollTop : null;
  root.innerHTML = `<div class="panel">
    <h3>Pod queue${data.project ? ` · ${esc(data.project)}` : ""}</h3>
    <label><input type="checkbox" data-a="hold" ${hold ? "checked" : ""}> Hold for the pod: <b>Queue</b> stores the job here instead of running it</label>
    <div class="muted">Set up the workflow, press Queue, change it, press Queue again… The batch count and seed randomizing work as usual.</div>
    ${editing ? `<div style="margin-top:10px;padding:8px;border:1px solid #e0a030;border-radius:8px">
      <b>Editing #${editing.n}</b> in the tab “${esc(editing.tab)}”. Change anything on the canvas, then save it back.
      <div class="muted">Queue in that tab still adds a <i>new</i> job; this button replaces #${editing.n}.</div>
      <div class="row"><button class="go" data-a="save-edit">Save changes to #${editing.n}</button><button data-a="cancel-edit">Stop editing</button></div></div>` : ""}
    <ul>${items.map(it => `<li${editing?.id === it.id ? ' style="background:rgba(224,160,48,.12)"' : ""}><div class="t"><b>#${it.n}</b> ${esc(it.name)}<div class="s">${esc(it.summary)}</div>${(it.notes || []).length ? `<div class="s" style="color:#e0a030">${esc(it.notes.join(" · "))}</div>` : ""}</div>
      <button class="x" data-a="edit" data-id="${it.id}" title="Open this job on the canvas to change it" ${r.running ? "disabled" : ""}>Edit</button>
      <button class="x" data-a="reseed" data-id="${it.id}" data-n="${it.n}" title="Add copies of this job with new seeds" ${r.running ? "disabled" : ""}>+ seeds</button>
      <button class="x" data-a="rm" data-id="${it.id}" title="Remove" ${r.running ? "disabled" : ""}>×</button></li>`).join("")
      || '<li class="muted">Nothing held yet.</li>'}</ul>
    <div class="row">
      <label style="margin:0">Max <input type="number" min="0.5" step="0.5" data-a="hours" value="${hours}"> h</label>
      <label style="margin:0"><input type="checkbox" data-a="auto" ${autoDestroy ? "checked" : ""}> destroy the pod when done</label>
    </div>
    <div class="row">
      <button class="go" data-a="run" ${r.running || !items.length ? "disabled" : ""}>Start pod &amp; run all (${items.length})</button>
      <button data-a="local" title="Queue every held job on this PC's ComfyUI instead" ${r.running || !items.length ? "disabled" : ""}>Run here (${items.length})</button>
      <button data-a="stop" ${r.running ? "" : "disabled"}>Stop</button>
      <span style="flex:1"></span>
      <button data-a="clear" ${r.running || !items.length ? "disabled" : ""}>Clear</button>
      <button data-a="destroy" title="Destroy the running pod now">Destroy pod</button>
    </div>
    ${r.phase ? `<div class="muted" style="margin-top:8px">${esc(r.phase)}${total ? ` · ${r.done || 0}/${total} done${r.failed ? `, ${r.failed} failed` : ""}` : ""}</div>
      <div class="bar"><div style="width:${total ? (100 * finished / total) : 0}%"></div></div>` : ""}
    ${(r.log || []).length ? `<pre>${esc(r.log.join("\n"))}</pre>` : ""}
    <div class="muted" style="margin-top:8px">Results are downloaded into this PC's output folder, at the same path they'd have had here. Finished jobs leave the list; failed ones stay so you can fix them and run again.</div>
  </div>${pill}`;
  const pre = root.querySelector("pre");
  if (pre) pre.scrollTop = keepScroll ?? pre.scrollHeight;
}

root.addEventListener("click", async e => {
  const el = e.target.closest("[data-a]"); if (!el) return;
  const a = el.dataset.a;
  if (a === "toggle") { open = !open; render(); if (open) refresh(); }
  else if (a === "reseed") {
    const count = parseInt(prompt(`How many more versions of #${el.dataset.n}? Same settings, a new seed each.`, "4"), 10);
    if (!count || count < 1) return;
    const res = await post("/reseed", { id: el.dataset.id, count });
    if (res.error) return alert(res.error);
    data = res; render();
    toast("Added to the pod queue", `${res.added} new version(s) of #${el.dataset.n}`);
  }
  else if (a === "edit") await startEdit(el.dataset.id);
  else if (a === "save-edit") await saveEdit();
  else if (a === "cancel-edit") { editing = null; render(); }
  else if (a === "rm") { data = await post("/remove", { ids: [el.dataset.id] }); render(); }
  else if (a === "clear") { if (confirm(`Remove all ${data.items.length} held jobs?`)) { data = await post("/remove", { all: true }); render(); } }
  else if (a === "stop") { if (confirm("Stop the run? What is still waiting on the pod is cancelled.")) { await post("/stop"); refresh(); } }
  else if (a === "destroy") { if (confirm("Destroy the pod now? Anything not downloaded yet is lost.")) { const r = await post("/destroy"); alert(r.message); refresh(); } }
  else if (a === "run") {
    const n = data.items.length;
    if (!confirm(`Run ${n} held job(s) on a vast.ai pod?\n\nIf no pod is running, one is rented now (hard limit ${hours} h). `
      + "If the pod kit on your private HF repo is older than your files here, a new one is uploaded first."
      + "\nEvery finished image is downloaded before the pod is destroyed."
      + (autoDestroy ? "\nIt is destroyed as soon as everything is done." : "\nIt will KEEP RUNNING afterwards and bill until you destroy it."))) return;
    const r = await post("/run", { hours, auto_destroy: autoDestroy });
    if (r.error) alert(r.error);
    refresh();
  }
  else if (a === "local") await runLocally();
});

// Queues every held job on this PC's ComfyUI (bypassing "Hold"). A job leaves the list once ComfyUI
// accepts it -- from then on it is in ComfyUI's own queue; rejected ones stay so they can be fixed.
async function runLocally() {
  const items = data.items || [];
  if (!confirm(`Queue ${items.length} held job(s) on this PC's ComfyUI instead of a pod?\n\n`
    + "Each job leaves this list once ComfyUI accepts it; rejected ones stay.")) return;
  const queued = [], rejected = [];
  for (const { id, n } of items) {
    try {
      const it = await (await api.fetchApi(`${BASE}/item?id=${encodeURIComponent(id)}`)).json();
      if (it.error) throw new Error(it.error);
      await originalQueuePrompt(0, { output: it.prompt, workflow: it.workflow });
      queued.push(id);
    } catch (e) {
      rejected.push(`#${n}: ${String(e).slice(0, 400)}`);  // PromptExecutionError's toString lists the node errors
    }
  }
  if (queued.length) data = await post("/remove", { ids: queued });
  render();
  toast("Queued on this PC", `${queued.length} job(s)` + (rejected.length ? `, ${rejected.length} rejected` : ""),
    rejected.length ? "warn" : "info");
  if (rejected.length) alert(`ComfyUI here rejected ${rejected.length} job(s); they stay in the list:\n\n${rejected.join("\n\n")}`);
}
root.addEventListener("change", e => {
  const a = e.target.dataset.a;
  if (a === "hold") {
    hold = e.target.checked;
    try { localStorage.setItem(HOLD_KEY, hold ? "1" : "0"); } catch {}
    toast(hold ? "Holding Queue presses for the pod" : "Queue runs on this PC again", "");
    render();
  } else if (a === "hours") hours = parseFloat(e.target.value) || 2;
  else if (a === "auto") autoDestroy = e.target.checked;
});

async function startEdit(id) {
  const it = await (await api.fetchApi(`${BASE}/item?id=${encodeURIComponent(id)}`)).json();
  if (it.error) return alert(it.error);
  if (!it.workflow) return alert(`#${it.n} was held without its workflow, so it can't be opened for editing.`);
  const tab = `Pod job #${it.n}`;
  // a name that isn't a saved file opens as a new temporary tab, like dropping a PNG: your canvas stays as it is
  await app.loadGraphData(it.workflow, true, true, tab);
  // the held prompt is the truth for the seeds ("+ seeds" copies only changed the prompt): put them on the widgets
  for (const [nid, node] of Object.entries(it.prompt || {})) {
    const g = /^\d+$/.test(nid) ? app.graph.getNodeById(Number(nid)) : null;
    if (!g || g.type !== node.class_type) continue;
    for (const k of ["seed", "noise_seed"]) {
      const w = g.widgets?.find(w => w.name === k);
      if (w && typeof node.inputs?.[k] === "number") w.value = node.inputs[k];
    }
  }
  app.graph.setDirtyCanvas?.(true, true);
  editing = { id: it.id, n: it.n, tab };
  open = true;
  render();
  toast(`Editing #${it.n}`, "Change it on the canvas, then press “Save changes” in the Pod queue panel.");
}

async function saveEdit() {
  if (!editing) return;
  const current = workflowName();
  if (!current.startsWith(editing.tab) &&
      !confirm(`The open tab is “${current}”, not “${editing.tab}”. Save THIS tab into #${editing.n} anyway?`)) return;
  const p = await app.graphToPrompt();
  const res = await post("/update", { id: editing.id, prompt: p.output, workflow: p.workflow });
  if (res.error) return alert(res.error);
  data = res;
  toast(`#${editing.n} updated`, (res.notes || []).join("; "), (res.notes || []).some(n => !n.startsWith("added")) ? "warn" : "info");
  const n = editing.n;
  editing = null;
  render();
  // e.g. a mask painted on the plate: offer the new file to every other job that used the old one
  for (const sw of res.swaps || []) {
    const short = v => v.split(" [")[0].split("/").pop();
    if (!confirm(`#${n} now uses “${short(sw.new)}” instead of “${short(sw.old)}”.\n\n`
      + `Use it in the other ${sw.others} held job(s) that still use “${short(sw.old)}” too?`)) continue;
    const r = await post("/replace_input", { old: sw.old, new: sw.new });
    if (r.error) { alert(r.error); continue; }
    data = r; render();
    toast("Updated the other jobs", `${r.changed} job(s) now use ${short(sw.new)}`);
  }
}

async function refresh() {
  clearTimeout(pollTimer);
  try { data = await (await api.fetchApi(BASE)).json(); } catch {}
  if (editing && !(data.items || []).some(it => it.id === editing.id)) editing = null;  // it ran or was removed
  // don't redraw under the cursor while typing the hours
  if (!(open && document.activeElement?.dataset?.a === "hours")) render();
  pollTimer = setTimeout(refresh, data.run?.running ? 3000 : open ? 8000 : 30000);
}

app.registerExtension({
  name: "ai_influencer.pod_queue",
  actionBarButtons: [{ icon: "pi pi-cloud", label: "Pod queue", tooltip: "Hold jobs and run them all on a vast.ai pod",
    onClick: () => { open = !open; render(); if (open) refresh(); } }],
  setup() { refresh(); },
});

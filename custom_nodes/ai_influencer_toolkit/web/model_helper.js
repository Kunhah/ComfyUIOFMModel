// "Models": model files for the local workflows, asked for only when they are needed. Server side: ../model_routes.py.
// - Opening a workflow swaps Krea 2 to the version that is installed (or chosen), so ComfyUI's own missing-model
//   list asks for the right file.
// - Pressing Run with a model file missing opens a window that says what each file is for, how big it is, whether
//   this graphics card can run it, lets you pick the Krea 2 version, and downloads into the right folder.
// - Extensions menu -> "AI Influencer: Models" opens the same window any time.
// - The Low memory version (GGUF) loads with ComfyUI-GGUF's UnetLoaderGGUF node, so switching to or from it also
//   swaps the loader node (same MODEL output, same links).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BASE = "/ai_influencer/models";
const HOLD_KEY = "ai_influencer.pod_queue.hold";  // web/pod_queue.js: jobs held for the pod use the pod's models
const KREA_GROUPS = ["krea2", "krea2_te"];

let info = null;  // GET /ai_influencer/models
const getInfo = async (fresh = false) => {
  if (!info || fresh) info = await (await api.fetchApi(BASE)).json();
  return info;
};
const post = async (path, body) => (await api.fetchApi(BASE + path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}),
})).json();
const toast = (summary, detail, severity = "info", life = 6000) => {
  try { app.extensionManager?.toast?.add({ severity, summary, detail, life }); } catch {}
};
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const gb = bytes => `${(bytes / 1e9).toFixed(1)} GB`;
const byName = () => Object.fromEntries(info.models.map(m => [m.name, m]));

// The file to use for a Krea 2 file `name`: the chosen version if installed, else any installed version, else the
// chosen version (so whatever asks for a download asks for that one). `strict`: the chosen version, installed or not
// (the window and a new choice use this, so picking a version is what gets downloaded).
// A file whose loader node isn't registered (GGUF before ComfyUI-GGUF is loaded) is never picked by opening a workflow.
function pickVersion(name, strict = false) {
  const m = byName()[name];
  if (!m || !KREA_GROUPS.includes(m.group)) return name;
  const order = [info.version, "fp8", "bf16", "nvfp4", "gguf"];
  const files = order.map(v => info.versions[v][m.group]);
  if (strict) return files[0];
  const loadable = f => !byName()[f].loader || info.addon.loaded;
  return files.find(f => byName()[f].installed && loadable(f)) || files.find(loadable);
}

const UNET_LOADERS = ["UNETLoader", "UnetLoaderGGUF"];
const loaderOf = file => byName()[file]?.loader || "UNETLoader";
const isKreaUnet = (type, value) => UNET_LOADERS.includes(type) && byName()[value]?.group === "krea2";

// --- opening a workflow ----------------------------------------------------------------------------------------
function allNodes(graphData) {
  const nodes = [...(graphData?.nodes || [])];
  for (const sg of graphData?.definitions?.subgraphs || []) nodes.push(...(sg.nodes || []));
  return nodes;
}

async function swapVersions(graphData) {
  await getInfo(true);
  const models = byName(), swapped = new Set();
  let needed = 0;
  for (const node of allNodes(graphData)) {
    if (!Array.isArray(node.widgets_values)) continue;
    if (isKreaUnet(node.type, node.widgets_values[0])) {
      const from = node.widgets_values[0], to = pickVersion(from), type = loaderOf(to);
      if (to !== from) {
        swapped.add(models[to].what);
        const meta = node.properties?.models?.find(e => e.name === from);
        if (meta) Object.assign(meta, { name: to, url: models[to].url, directory: models[to].folder });
      }
      if (type !== node.type) {  // same single MODEL output, so its links stay valid
        node.type = type;
        node.properties = { ...node.properties, "Node name for S&R": type };
        node.widgets_values = type === "UnetLoaderGGUF" ? [to] : [to, "default"];
      } else node.widgets_values[0] = to;
      if (node.mode !== 2 && node.mode !== 4 && !models[to].installed) needed += models[to].size;
      continue;
    }
    node.widgets_values = node.widgets_values.map(v => {
      if (typeof v !== "string" || !models[v]) return v;
      const to = pickVersion(v);
      if (to !== v) {
        swapped.add(models[to].what);
        const meta = node.properties?.models?.find(e => e.name === v);
        if (meta) Object.assign(meta, { name: to, url: models[to].url, directory: models[to].folder });
      }
      if (node.mode !== 2 && node.mode !== 4 && !models[to].installed) needed += models[to].size;
      return to;
    });
  }
  if (swapped.size) toast("Krea 2 version", `Using: ${[...swapped].join("; ")}.`);
  if (needed && localStorage.getItem(HOLD_KEY) !== "1") {
    toast("This workflow runs on your graphics card",
      `It needs ${gb(needed)} of model files that aren't downloaded yet. You can look around without them. ` +
      "When you press Run, a window explains each file and checks your graphics card first.", "info", 12000);
  }
}

// --- pressing Run ----------------------------------------------------------------------------------------------
let windowOpen = false;
const originalQueuePrompt = api.queuePrompt.bind(api);
api.queuePrompt = async function (number, promptData, options) {
  let hold = false;
  try { hold = localStorage.getItem(HOLD_KEY) === "1"; } catch {}
  if (!hold) {
    const names = new Set();
    for (const n of Object.values(promptData?.output || {}))
      for (const v of Object.values(n.inputs || {})) if (typeof v === "string") names.add(v);
    let missing = [];
    try { missing = (await post("/check", { names: [...names] })).missing || []; } catch {}
    await getInfo(true).catch(() => {});
    const unloadable = [...names].filter(n => info?.models.some(m => m.name === n && m.loader) && !info.addon.loaded);
    if (missing.length || unloadable.length) {
      if (!windowOpen) openWindow([...missing.map(m => m.name), ...unloadable], "run");
      return { prompt_id: "", number: 0, node_errors: {} };  // nothing queued, nothing to wait for
    }
  }
  return originalQueuePrompt(number, promptData, options);
};

// --- the window ------------------------------------------------------------------------------------------------
const css = document.createElement("style");
css.textContent = `
#aiinf-models { position:fixed; inset:0; z-index:5000; background:rgba(0,0,0,.55); display:flex; align-items:center; justify-content:center;
  font:14px/1.45 system-ui, sans-serif; color:var(--fg-color, #ddd); }
#aiinf-models .box { width:min(620px, calc(100vw - 32px)); max-height:calc(100vh - 48px); overflow:auto; padding:18px 20px;
  border-radius:12px; background:var(--comfy-menu-bg, #202020); border:1px solid var(--border-color, #444); box-shadow:0 8px 30px rgba(0,0,0,.5); }
#aiinf-models h2 { margin:0 0 6px; font-size:18px; } #aiinf-models h3 { margin:16px 0 6px; font-size:14px; }
#aiinf-models p { margin:6px 0; } #aiinf-models .muted { opacity:.7; font-size:12.5px; }
#aiinf-models .advice { padding:8px 10px; border-radius:8px; background:rgba(128,128,128,.14); }
#aiinf-models label.v { display:block; padding:8px 10px; margin:6px 0; border-radius:8px; border:1px solid var(--border-color, #444); cursor:pointer; }
#aiinf-models label.v input { margin-right:6px; }
#aiinf-models .file { display:grid; grid-template-columns:1fr auto; gap:2px 12px; padding:7px 0; border-top:1px solid var(--border-color, #333); }
#aiinf-models .bar { grid-column:1 / -1; height:6px; border-radius:3px; background:rgba(128,128,128,.25); overflow:hidden; }
#aiinf-models .bar i { display:block; height:100%; background:#4caf7d; }
#aiinf-models .err { color:#e57373; } #aiinf-models .ok { color:#4caf7d; }
#aiinf-models .buttons { display:flex; gap:8px; justify-content:flex-end; margin-top:14px; flex-wrap:wrap; }
#aiinf-models button { padding:7px 14px; border-radius:8px; border:1px solid var(--border-color, #555); cursor:pointer;
  background:var(--comfy-input-bg, #2a2a2a); color:inherit; font:inherit; }
#aiinf-models button.primary { background:#2f6fd0; border-color:#2f6fd0; color:#fff; }
#aiinf-models button:disabled { opacity:.5; cursor:default; }`;
document.head.appendChild(css);

// Catalog files the switched-on nodes of the open workflow load.
function canvasModels() {
  const models = byName(), names = [];
  for (const node of app.graph?._nodes || app.graph?.nodes || []) {
    if (node.mode === 2 || node.mode === 4) continue;
    for (const w of node.widgets || []) if (models[w.value] && !names.includes(w.value)) names.push(w.value);
  }
  return names;
}

// Swap a Krea 2 loader node on the canvas for one of `type`, keeping its place and the links of its MODEL output.
function replaceLoader(old, type, file) {
  const LG = globalThis.LiteGraph, graph = old.graph || app.graph;
  const node = LG?.createNode(type);
  if (!node) return false;  // type not registered: ComfyUI-GGUF not loaded yet
  node.pos = [...old.pos];
  node.mode = old.mode;
  graph.add(node);
  node.widgets[0].value = file;
  const link = id => graph.links?.get ? graph.links.get(id) : graph.links?.[id];
  for (const id of [...(old.outputs?.[0]?.links || [])]) {
    const l = link(id);
    if (l) node.connect(0, graph.getNodeById(l.target_id), l.target_slot);
  }
  graph.remove(old);
  return true;
}

// Put the chosen Krea 2 version into the open workflow's loader nodes.
function applyVersionToCanvas() {
  for (const node of [...(app.graph?._nodes || app.graph?.nodes || [])])
    for (const w of node.widgets || []) {
      const to = byName()[w.value] ? pickVersion(w.value, true) : w.value;
      if (isKreaUnet(node.type, w.value) && loaderOf(to) !== node.type) {
        if (!replaceLoader(node, loaderOf(to), to)) w.value = to;  // becomes a GGUF node after the restart
        break;
      }
      if (to !== w.value) { w.value = to; w.callback?.(to); }
    }
  app.graph?.setDirtyCanvas?.(true, true);
}

async function openWindow(names, reason) {
  windowOpen = true;
  await getInfo(true);
  if (reason === "run") applyVersionToCanvas();  // what the canvas loads = the version this window offers
  const el = document.createElement("div");
  el.id = "aiinf-models";
  document.body.appendChild(el);
  let timer = null;
  const close = () => { clearInterval(timer); el.remove(); windowOpen = false; };
  el.addEventListener("click", e => { if (e.target === el) close(); });

  const render = (progress = {}) => {
    const models = byName();
    names = [...new Set(names.map(n => pickVersion(n, true)))];
    const list = names.map(n => models[n]);
    const missing = list.filter(m => !m.installed && progress[m.name]?.status !== "finished");
    const usesKrea = reason === "menu" || list.some(m => KREA_GROUPS.includes(m.group));
    const running = Object.values(progress).some(p => p.status === "running");
    const total = missing.reduce((s, m) => s + m.size, 0);
    const needsAddon = list.some(m => m.loader) && !info.addon.loaded;
    const addonNote = !needsAddon ? "" : info.addon.present
      ? `<p class="advice">The ComfyUI-GGUF add-on is installed. <b>Restart ComfyUI</b> (close the black window and start it again) to finish.</p>`
      : `<p class="muted">The Low memory version also needs the free ComfyUI-GGUF add-on (it loads GGUF files). Downloading installs it; restart ComfyUI afterwards.</p>`;
    el.innerHTML = `<div class="box">
      <h2>${reason === "run" ? "This workflow needs model files first" : "Models for the local workflows"}</h2>
      <p>${reason === "run" ? "This workflow makes" : "The local workflows (07-10, 09b, 12) make"} the picture on <b>your own graphics card</b>, so ${reason === "run" ? "it needs" : "they need"} the AI model files on this computer.
        They are downloaded once into <code>ComfyUI/models</code> and reused by every local workflow.
        The online workflows (01, 04, 05) need no downloads, but they spend Comfy credits.</p>
      <p class="advice">${esc(info.advice)}</p>
      ${usesKrea ? `<h3>Krea 2 version</h3>
        ${Object.entries(info.versions).map(([k, v]) => {
          const size = models[v.krea2].size + models[v.krea2_te].size;
          const tags = [k === info.recommended && info.fits ? "recommended for this computer" : "",
                        models[v.krea2].installed ? "downloaded" : gb(size)].filter(Boolean).join(" · ");
          return `<label class="v"><input type="radio" name="ver" value="${k}" ${k === info.version ? "checked" : ""} ${running ? "disabled" : ""}>
            <b>${esc(v.label)}</b> <span class="muted">(${tags})</span><br><span class="muted">${esc(v.who)}</span></label>`;
        }).join("")}` : ""}
      <h3>Files this workflow uses</h3>
      ${list.length ? list.map(m => {
        const p = progress[m.name];
        const status = m.installed || p?.status === "finished" ? `<span class="ok">downloaded</span>`
          : p?.status === "running" ? `${gb(p.done)} of ${gb(p.total)}`
          : p?.status === "failed" ? `<span class="err">failed</span>` : gb(m.size);
        return `<div class="file"><span>${esc(m.what)}<br><span class="muted">${esc(m.name)} → models/${esc(m.folder)}</span></span>
          <span>${status}</span>
          ${p?.status === "running" ? `<div class="bar"><i style="width:${(100 * p.done / p.total).toFixed(1)}%"></i></div>` : ""}
          ${p?.status === "failed" ? `<span class="err muted" style="grid-column:1/-1">${esc(p.error)}</span>` : ""}</div>`;
      }).join("") : `<p class="muted">The open workflow loads no local model files.</p>`}
      ${addonNote}
      <p class="muted">${missing.length ? `To download: ${gb(total)}. Free disk space: ${info.free_gb.toFixed(0)} GB.` : list.length ? "Everything this workflow needs is downloaded." : ""}
        ${running ? " You can keep using ComfyUI; closing this window doesn't stop the download." : ""}</p>
      <div class="buttons">
        <button data-a="close">${running ? "Hide" : missing.length ? "Not now" : "Close"}</button>
        ${missing.length || (needsAddon && !info.addon.present) ? `<button class="primary" data-a="download" ${running || total > info.free_gb * 1e9 ? "disabled" : ""}>
          ${running ? "Downloading…" : missing.length ? `Download ${gb(total)}` : "Install the add-on"}</button>` : ""}
      </div></div>`;

    el.querySelectorAll("input[name=ver]").forEach(r => r.addEventListener("change", async () => {
      await post("/version", { version: r.value });
      await getInfo(true);
      applyVersionToCanvas();
      render(progress);
    }));
    el.querySelector("[data-a=close]").onclick = close;
    const dl = el.querySelector("[data-a=download]");
    if (dl) dl.onclick = async () => {
      if (needsAddon && !info.addon.present) {
        dl.disabled = true; dl.textContent = "Installing the add-on…";
        const r = await post("/addon");
        if (r.error) { toast("ComfyUI-GGUF add-on", r.error, "error", 15000); render(progress); return; }
        await getInfo(true);
      }
      if (missing.length) { await post("/download", { names: missing.map(m => m.name) }); poll(); }
      else render(progress);
    };
  };

  const poll = () => {
    clearInterval(timer);
    timer = setInterval(async () => {
      const progress = await (await api.fetchApi(BASE + "/progress")).json();
      const mine = Object.fromEntries(Object.entries(progress).filter(([k]) => names.includes(k)));
      if (!Object.values(mine).some(p => p.status === "running")) {
        clearInterval(timer);
        await getInfo(true);
        try { await app.refreshComboInNodes(); } catch {}
        const failed = Object.values(mine).some(p => p.status === "failed");
        const restart = names.some(n => byName()[n]?.loader) && !info.addon.loaded;
        if (!failed) toast("Models ready", restart ? "Restart ComfyUI (close the black window and start it again), then press Run."
                                                   : "Press Run again.", "success", restart ? 15000 : 6000);
      }
      if (document.body.contains(el)) render(mine);
    }, 1000);
  };

  render();
  // A download started earlier (then hidden) keeps showing its progress.
  const progress = await (await api.fetchApi(BASE + "/progress")).json();
  if (names.some(n => progress[pickVersion(n, true)]?.status === "running")) poll();
}

app.registerExtension({
  name: "ai_influencer.models",
  async beforeConfigureGraph(graphData) {
    try { await swapVersions(graphData); } catch (e) { console.warn("[AI Influencer models]", e); }
  },
  commands: [{
    id: "ai_influencer.models", label: "AI Influencer: Models", icon: "pi pi-download",
    function: async () => { await getInfo(true); openWindow(canvasModels(), "menu"); },
  }],
  menuCommands: [{ path: ["Extensions"], commands: ["ai_influencer.models"] }],
});

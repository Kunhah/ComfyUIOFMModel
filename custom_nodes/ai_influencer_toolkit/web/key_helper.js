// Keys: pressing Run on a workflow whose nodes need a key that isn't set yet (FAL_KEY for the fal.ai nodes of
// Workflow 06) opens a window to paste it. It is saved to .env, like tools/configure.py, and the run goes ahead.
// Server side: ../key_routes.py.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const post = async (path, body) => (await api.fetchApi(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}),
})).json();
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const css = document.createElement("style");
css.textContent = `
#aiinf-keys { position:fixed; inset:0; z-index:5000; background:rgba(0,0,0,.55); display:flex; align-items:center; justify-content:center;
  font:14px/1.45 system-ui, sans-serif; color:var(--fg-color, #ddd); }
#aiinf-keys .box { width:min(520px, calc(100vw - 32px)); padding:18px 20px; border-radius:12px;
  background:var(--comfy-menu-bg, #202020); border:1px solid var(--border-color, #444); box-shadow:0 8px 30px rgba(0,0,0,.5); }
#aiinf-keys h2 { margin:0 0 6px; font-size:18px; } #aiinf-keys p { margin:6px 0; } #aiinf-keys .muted { opacity:.7; font-size:12.5px; }
#aiinf-keys a { color:#6aa5ff; } #aiinf-keys .err { color:#e57373; }
#aiinf-keys label { display:block; margin-top:12px; font-weight:600; }
#aiinf-keys input { width:100%; box-sizing:border-box; margin-top:4px; padding:7px 9px; border-radius:8px; font:inherit;
  border:1px solid var(--border-color, #555); background:var(--comfy-input-bg, #2a2a2a); color:inherit; }
#aiinf-keys .buttons { display:flex; gap:8px; justify-content:flex-end; margin-top:14px; }
#aiinf-keys button { padding:7px 14px; border-radius:8px; border:1px solid var(--border-color, #555); cursor:pointer;
  background:var(--comfy-input-bg, #2a2a2a); color:inherit; font:inherit; }
#aiinf-keys button.primary { background:#2f6fd0; border-color:#2f6fd0; color:#fff; }`;
document.head.appendChild(css);

// Resolves true once every key is saved, false when closed without saving.
function askForKeys(missing) {
  return new Promise(resolve => {
    const el = document.createElement("div");
    el.id = "aiinf-keys";
    el.innerHTML = `<div class="box">
      <h2>This workflow needs ${missing.length === 1 ? "a key" : "some keys"} first</h2>
      ${missing.map(k => `<label>${esc(k.name)}
          <input type="password" autocomplete="off" spellcheck="false" data-key="${esc(k.name)}" placeholder="paste it here"></label>
        <p class="muted">${esc(k.help)}${k.url ? ` Get one at <a href="${esc(k.url)}" target="_blank" rel="noopener">${esc(k.url)}</a>.` : ""}</p>`).join("")}
      <p class="muted">Saved in <code>.env</code> on this computer only (never committed). Change it later with SETTINGS_API_KEYS.</p>
      <p class="err"></p>
      <div class="buttons"><button data-a="cancel">Not now</button><button class="primary" data-a="save">Save and run</button></div></div>`;
    document.body.appendChild(el);
    const done = ok => { el.remove(); resolve(ok); };
    el.querySelector("input")?.focus();
    el.addEventListener("click", e => { if (e.target === el) done(false); });
    el.querySelector("[data-a=cancel]").onclick = () => done(false);
    el.querySelector("[data-a=save]").onclick = async () => {
      const values = Object.fromEntries([...el.querySelectorAll("input[data-key]")].map(i => [i.dataset.key, i.value.trim()]));
      if (Object.values(values).some(v => !v)) { el.querySelector(".err").textContent = "Paste every key first."; return; }
      const r = await post("/ai_influencer/keys", { values }).catch(e => ({ error: String(e) }));
      if (r.error) el.querySelector(".err").textContent = r.error;
      else done(true);
    };
  });
}

const originalQueuePrompt = api.queuePrompt.bind(api);
api.queuePrompt = async function (number, promptData, options) {
  const types = [...new Set(Object.values(promptData?.output || {}).map(n => n.class_type))];
  let missing = [];
  try { missing = (await post("/ai_influencer/keys/check", { types })).missing || []; } catch {}
  if (missing.length && !(await askForKeys(missing)))
    return { prompt_id: "", number: 0, node_errors: {} };  // nothing queued, nothing to wait for
  return originalQueuePrompt(number, promptData, options);
};

app.registerExtension({ name: "ai_influencer.keys" });

// Picture inputs a workflow names that aren't on this computer. Server side: ../input_listing.py.
// - Opening a workflow maps each such Load Image file onto this computer's character project
//   (ai_influencer/MyCharacter/dataset/62.png -> ai_influencer/<project>/dataset/62.png) or, when there is no such file,
//   selects a "Choose a picture" placeholder, so ComfyUI never reports a missing input.
// - Pressing Run while a placeholder is still selected says which node needs a picture, and queues nothing.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PICTURE_NODES = new Map();  // node type -> name of its picture input (the first one: LoadImage's "image")
let placeholder = "ai_influencer_choose_a_picture.png";

const toast = (summary, detail, severity = "info", life = 8000) => {
  try { app.extensionManager?.toast?.add({ severity, summary, detail, life }); } catch {}
};

function allNodes(graphData) {
  const nodes = [...(graphData?.nodes || [])];
  for (const sg of graphData?.definitions?.subgraphs || []) nodes.push(...(sg.nodes || []));
  return nodes;
}

// The files a picture input can choose right now (fresh, so a photo added since startup counts).
async function pictureOptions(type) {
  const def = (await (await api.fetchApi(`/object_info/${encodeURIComponent(type)}`)).json())[type];
  const spec = def?.input?.required?.[PICTURE_NODES.get(type)];
  return new Set(Array.isArray(spec?.[0]) ? spec[0] : spec?.[1]?.options || []);
}

async function fixPictures(graphData) {
  const nodes = allNodes(graphData).filter(n => PICTURE_NODES.has(n.type)
    && Array.isArray(n.widgets_values) && typeof n.widgets_values[0] === "string" && n.widgets_values[0].trim()
    && !/\s\[output\]$/.test(n.widgets_values[0]));  // "x.png [output]" is checked against the output folder
  if (!nodes.length) return;
  const settings = await (await api.fetchApi("/ai_influencer/inputs")).json();
  placeholder = settings.placeholder || placeholder;
  const options = {};
  for (const type of new Set(nodes.map(n => n.type))) options[type] = await pictureOptions(type);

  const toChoose = [];
  for (const node of nodes) {
    const have = options[node.type], value = node.widgets_values[0];
    if (have.has(value) || have.has(value.replace(/\s\[input\]$/, ""))) continue;
    const rest = value.match(/^ai_influencer\/[^/]+\/(.+)$/)?.[1];
    const mapped = rest && settings.project ? `ai_influencer/${settings.project}/${rest}` : "";
    if (mapped && have.has(mapped)) { node.widgets_values[0] = mapped; continue; }
    if (!have.has(placeholder)) continue;
    node.properties = { ...node.properties, ai_influencer_wanted: value };
    node.widgets_values[0] = placeholder;
    if (node.mode !== 2 && node.mode !== 4) toChoose.push(node.title || node.type);  // bypassed ones: once switched on
  }
  if (toChoose.length)
    toast("Choose your pictures", `${toChoose.length === 1 ? "One picture node needs" : `${toChoose.length} picture nodes need`} ` +
      `a photo from this computer: ${toChoose.join(", ")}. Click the "Choose a picture" image in each, or drag a photo onto it.`,
      "info", 12000);
}

const originalQueuePrompt = api.queuePrompt.bind(api);
api.queuePrompt = async function (number, promptData, options) {
  const waiting = [];
  for (const [id, n] of Object.entries(promptData?.output || {}))
    if (Object.values(n.inputs || {}).includes(placeholder)) {
      const wanted = app.graph?.getNodeById?.(id)?.properties?.ai_influencer_wanted;
      waiting.push(`${n._meta?.title || n.class_type}${wanted ? ` (was ${wanted})` : ""}`);
    }
  if (waiting.length) {
    toast("Choose a picture first", `Still showing "Choose a picture": ${waiting.join(", ")}. ` +
      "Click the image in that node to pick a photo, or drag one onto it, then press Run again.", "warn", 15000);
    return { prompt_id: "", number: 0, node_errors: {} };  // nothing queued, nothing to wait for
  }
  return originalQueuePrompt(number, promptData, options);
};

app.registerExtension({
  name: "ai_influencer.inputs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const [name, spec] = Object.entries(nodeData?.input?.required || {})[0] || [];
    if (spec?.[1]?.image_upload) PICTURE_NODES.set(nodeData.name, name);
  },
  async beforeConfigureGraph(graphData) {
    try { await fixPictures(graphData); } catch (e) { console.warn("[AI Influencer inputs]", e); }
  },
});

// SPDX-License-Identifier: MPL-2.0
import {apis, defaults, eligible, executors, fields, fromHash, layouts, ownersOf,
  scalars, serialize, snippet, traversals, tuple, workers} from "./model.mjs";

const $ = id => document.getElementById(id);
const colors = ["#4964b8", "#208776", "#b3633c", "#9561a6", "#9b7b28", "#347c9d"];
let config;
try { config = fromHash(location.hash); } catch { config = defaults(); }
let engine, manifest, trace, owners, selectedWorker = 0, selectedAssignment = null;
let outerIteration = 0, innerIteration = 0;
const status = (message, error = false) => {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
};

for (const select of document.querySelectorAll("[data-policy]")) {
  const values = select.dataset.policy === "layout" ? layouts : traversals;
  values.forEach((name, value) => select.add(new Option(name, value)));
}
workers.forEach((name, value) => $("worker").add(new Option(name, value)));

function controls() {
  for (const key of scalars) if ($(key)) $(key).value = config[key];
  $("executor").replaceChildren();
  [0, ...(config.api === 0 ? [1, 2, 3] : [config.api + 3])].forEach(value =>
    $("executor").add(new Option(executors[value], value)));
  $("executor").value = config.executor;
  for (const container of document.querySelectorAll("[data-vector]")) {
    const key = container.dataset.vector;
    const label = document.createElement("span");
    label.textContent = container.dataset.label;
    const row = document.createElement("div");
    row.className = "vector-inputs";
    for (let d = 0; d < config.dim; ++d) {
      if (d) row.append("×");
      const input = document.createElement("input");
      input.type = "number";
      input.min = key === "begin" ? 0 : 1;
      input.max = 64;
      input.step = 1;
      input.value = config[key][d];
      input.dataset.field = key;
      input.dataset.dimension = d;
      input.setAttribute("aria-label", `${container.dataset.label}, ${config.dim === 1 ? "x" : d === 0 ? "row" : "column"}`);
      row.append(input);
    }
    container.replaceChildren(label, row);
  }
  $("frames-field").hidden = !config.spec;
  $("outer-card").hidden = !config.hierarchy;
  $("inner-level").textContent = config.hierarchy ? "2" : "1";
  $("inner-title").textContent = config.hierarchy ? "Elements inside tile" : "Logical range";
  $("inner-hint").textContent = config.hierarchy ? "Range: [0, tile extent), stride = 1. global = tileStart + localIdx."
    : "Range: [begin, end), using the configured stride.";
  $("assumption").textContent = config.spec
    ? "FrameSpec does not fix the worker layout. Enter an assumed ThreadSpec below; no backend resolution is performed."
    : "Explicit block/thread extents. This does not execute a GPU kernel.";
  $("assumption").classList.toggle("assumption", Boolean(config.spec));
  for (const option of $("worker").options) {
    option.disabled = config.dim === 2 && [1, 3, 5].includes(Number(option.value));
  }
  $("outerWorker").options[1].disabled = config.dim === 2;
  for (const input of document.querySelectorAll('[data-field="stride"]')) input.disabled = Boolean(config.hierarchy);
}

function currentWorker() { return trace?.workers.find(worker => worker.id === selectedWorker); }
function currentAssignments() {
  return (currentWorker()?.assignments ?? []).filter(a => $("all-iterations").checked
    || (a.outerIteration === outerIteration && a.innerIteration === innerIteration));
}
function updateIterations() {
  const assignments = currentWorker()?.assignments ?? [];
  const maxOuter = Math.max(0, ...assignments.map(a => a.outerIteration));
  const maxInner = Math.max(0, ...assignments.map(a => a.innerIteration));
  outerIteration = Math.min(outerIteration, maxOuter);
  innerIteration = Math.min(innerIteration, maxInner);
  for (const [name, value, max] of [["outer", outerIteration, maxOuter], ["inner", innerIteration, maxInner]]) {
    const input = $(`${name}-iteration`);
    input.value = value;
    input.max = max;
    input.disabled = $("all-iterations").checked || (name === "outer" && !config.hierarchy);
  }
}

function drawWorkers() {
  $("worker-count").textContent = `${trace.workers.length} workers`;
  let html = "", lastBlock = "";
  for (const worker of trace.workers) {
    const key = worker.block.join(",");
    if (lastBlock !== key) {
      if (lastBlock) html += "</div></section>";
      html += `<section class="block-group"><h4>Block ${tuple(worker.block)}</h4><div class="thread-buttons">`;
      lastBlock = key;
    }
    html += `<button type="button" data-worker="${worker.id}" aria-pressed="${worker.id === selectedWorker}"
      class="thread ${worker.id === selectedWorker ? "active" : ""}"
      title="Worker ${worker.id}: ${worker.assignments.length} assignments">${tuple(worker.thread)}<small>T${worker.id}</small></button>`;
  }
  $("workers").innerHTML = html + "</div></section>";
}

function drawData() {
  const rows = config.dim === 2 ? config.end[0] : 1;
  const cols = config.end.at(-1);
  const cell = config.dim === 2 ? 48 : 56;
  const pad = 30, width = cols * cell + pad * 2, height = rows * cell + pad * 2;
  const selected = new Set(currentAssignments().filter(a => a.inBounds).map(a => a.index.join(",")));
  const blockColor = worker => {
    const block = worker.block.join(",");
    const keys = [...new Set(trace.workers.map(w => w.block.join(",")))];
    return colors[keys.indexOf(block) % colors.length];
  };
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="group" aria-label="Logical data indices">
    <defs><pattern id="excluded" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M0 6L6 0" stroke="#d7dce1" stroke-width="1"/></pattern></defs>`;
  for (let row = 0; row < rows; ++row) for (let col = 0; col < cols; ++col) {
    const index = config.dim === 2 ? [row, col] : [col], key = index.join(",");
    const entries = owners.get(key) ?? [], unique = [...new Set(entries.map(entry => entry.worker.id))];
    const active = selected.has(key), isSelected = selectedAssignment?.index.join(",") === key;
    const valid = eligible(config, index);
    const fill = !valid ? "url(#excluded)" : active ? "#d9e8f0" : entries.length ? "#eef1f5" : "#fff";
    const label = $("labels").value === "linear" ? (entries[0]?.assignment.linear ?? "—")
      : $("labels").value === "owner" ? (unique.length > 1 ? `${unique.length}×` : unique[0] ?? "—")
      : config.dim === 2 ? `${row},${col}` : `${col}`;
    svg += `<g data-cell="${key}" data-owner-count="${unique.length}" data-highlighted="${active}" role="button" tabindex="0"
      aria-label="Index ${tuple(index)}, ${unique.length} owners${active ? ", assigned to selected worker" : ""}">
      <title>Index ${tuple(index)} · ${entries.length} assignments · workers ${unique.join(", ") || "none"}</title>
      <rect x="${pad + col * cell + 3}" y="${pad + row * cell + 3}" width="${cell - 6}" height="${cell - 6}" rx="6"
        fill="${fill}" stroke="${isSelected ? "#c46b26" : active ? "#286a85" : "#d7dce1"}" stroke-width="${isSelected || active ? 2 : 1}"/>
      <text x="${pad + col * cell + cell / 2}" y="${pad + row * cell + cell / 2 + 4}" text-anchor="middle">${label}</text>
      ${entries.length ? `<circle cx="${pad + col * cell + cell - 11}" cy="${pad + row * cell + 11}" r="3" fill="${unique.length > 1 ? "#b06c22" : blockColor(entries[0].worker)}"/>` : ""}</g>`;
  }
  if (config.hierarchy) {
    for (let row = config.dim === 2 ? config.begin[0] : 0; row < rows; row += config.dim === 2 ? config.tile[0] : rows)
      for (let col = config.begin.at(-1); col < cols; col += config.tile.at(-1)) {
        const tileRows = Math.min(config.dim === 2 ? config.tile[0] : 1, rows - row);
        const tileCols = Math.min(config.tile.at(-1), cols - col);
        svg += `<rect class="tile-outline" x="${pad + col * cell}" y="${pad + row * cell}"
          width="${tileCols * cell}" height="${tileRows * cell}" rx="8"/>`;
      }
  }
  for (let col = 0; col < cols; ++col) svg += `<text class="axis" x="${pad + col * cell + cell / 2}" y="17" text-anchor="middle">${col}</text>`;
  if (config.dim === 2) for (let row = 0; row < rows; ++row) svg += `<text class="axis" x="15" y="${pad + row * cell + cell / 2 + 4}" text-anchor="middle">${row}</text>`;
  $("data").innerHTML = svg + "</svg>";
  $("domain-size").textContent = `${rows} × ${cols}`;
}

function inspect() {
  const worker = currentWorker();
  const list = currentAssignments();
  const assignment = list.find(a => a === selectedAssignment)
    ?? list.find(a => a.outerIteration === outerIteration && a.innerIteration === innerIteration) ?? list[0];
  selectedAssignment = assignment ?? null;
  const info = worker.inner;
  let html = `<p class="selection-label">Worker T${worker.id} · Block ${tuple(worker.block)} · Thread ${tuple(worker.thread)}</p>`;
  if (config.hierarchy) html += `<div class="calculation-step"><span class="level">1</span><div><h3>Block → tile origin</h3>
    <p>Group index ${tuple(worker.outer.workerIndex)} of ${tuple(worker.outer.workerCount)}</p>
    <code>${worker.outer.traversal} traversal · ${worker.outer.layout} layout</code></div></div>`;
  html += `<div class="calculation-step"><span class="level">${config.hierarchy ? 2 : 1}</span><div><h3>${workers[config.worker]} → ${config.hierarchy ? "local index" : "range index"}</h3>
    <p>Group index ${tuple(info.workerIndex)} of ${tuple(info.workerCount)}</p>
    <code>${info.traversal} traversal · ${info.layout} layout</code></div></div>`;
  if (assignment) {
    html += `<div class="formula">${config.hierarchy
      ? `global = tileStart + localIdx<br><strong>${tuple(assignment.index)} = ${tuple(assignment.origin)} + ${tuple(assignment.local)}</strong>`
      : `Returned index<br><strong>${tuple(assignment.index)}</strong>`}</div>
      <p>Outer iteration <strong>${assignment.outerIteration}</strong> · Inner iteration <strong>${assignment.innerIteration}</strong></p>
      <p>${assignment.inBounds ? `Linear offset (origin-zero buffer): <strong>${assignment.linear}</strong>` : "<strong>Outside the logical range — rejected by the bounds guard.</strong>"}</p>`;
  } else html += "<p>This worker has no assignments in the selected iteration.</p>";
  html += "<p class=\"hint\">Worker identities, effective policies, returned indices, composition, and linear offsets above come from C++. Intermediate iterator arithmetic is not yet traced.</p>";
  $("calculation").innerHTML = html;
  $("code").textContent = snippet(config, trace);
  $("table").innerHTML = `<p class="hint">Selected worker · ${list.length} assignments (including guarded indices)</p><div class="table-scroll"><table><thead><tr><th>Outer</th><th>Inner</th><th>Origin</th><th>Local</th><th>Global</th><th>Guard</th></tr></thead><tbody>${list.map(a =>
    `<tr><td>${a.outerIteration}</td><td>${a.innerIteration}</td><td>${tuple(a.origin)}</td><td>${tuple(a.local)}</td><td>${tuple(a.index)}</td><td>${a.inBounds ? "inside" : "outside"}</td></tr>`).join("")}</tbody></table></div>`;
  let expected = 0;
  const rows = config.dim === 2 ? config.end[0] : 1, cols = config.end.at(-1);
  for (let row = 0; row < rows; ++row) for (let col = 0; col < cols; ++col)
    if (eligible(config, config.dim === 2 ? [row, col] : [col])) ++expected;
  const repeated = [...owners.values()].filter(entries => entries.length > 1).length;
  const all = trace.workers.flatMap(w => w.assignments);
  $("diagnostics").innerHTML = `<div class="metrics"><div><strong>${owners.size}/${expected}</strong><span>eligible indices covered</span></div>
    <div><strong>${repeated}</strong><span>indices with repeated assignments</span></div>
    <div><strong>${all.filter(a => !a.inBounds).length}</strong><span>assignments rejected by guard</span></div>
    <div><strong>${trace.workers.filter(w => !w.assignments.length).length}</strong><span>idle workers</span></div></div>
    <p>Repeated assignments can be intentional: multiple physical workers may share a worker-group identity. Coverage alone does not prove a kernel is race-free.</p>
    <p><strong>Context:</strong> ${apis[config.api]} API tags in synthetic worker contexts; no device kernel is executed.</p>
    ${config.spec ? `<p class="assumption">FrameSpec logical capacity ${tuple(trace.frameCapacity)}. Blocks and threads are explicit assumptions, not a backend-derived launch.</p>` : ""}
    <p class="hint">No occupancy, timing, synchronization, memory-race, or hardware scheduling simulation. Warp and boundary overloads are not implemented.</p>`;
}

function render() {
  if (!trace) return;
  updateIterations();
  inspect(); // Updates the selected assignment before drawing its outline.
  drawWorkers();
  drawData();
}
function recompute() {
  if (!engine) return;
  try {
    const result = JSON.parse(engine.ccall("mapping_trace", "string", ["string"], [serialize(config)]));
    if (result.error) throw Error(result.error);
    if (result.build !== manifest.build) throw Error("Engine and manifest do not match. Rebuild the prototype.");
    trace = result;
    owners = ownersOf(trace);
    if (!currentWorker()) selectedWorker = 0;
    selectedAssignment = null;
    status(config.spec ? "FrameSpec input · worker layout explicitly assumed, NOT resolved by a backend."
      : "Live C++ / WebAssembly · real alpaka index iteration · synthetic worker contexts");
    render();
  } catch (error) {
    trace = null;
    for (const id of ["workers", "data", "calculation", "table", "diagnostics"]) $(id).replaceChildren();
    $("code").textContent = "";
    status(error.message, true);
  }
}

$("config-form").addEventListener("submit", event => event.preventDefault());
$("config-form").addEventListener("change", event => {
  const target = event.target;
  if (target.id === "preset") {
    config = defaults(config.dim);
    if (target.value === "direct") { config.hierarchy = 0; config.worker = 0; }
    if (target.value === "uneven") config.end = config.dim === 2 ? [3, 7] : [15];
    if (target.value === "host") { config.api = 0; config.threads = Array(config.dim).fill(1); }
  } else if (target.id === "dim") config = defaults(Number(target.value));
  else if (target.dataset.field) config[target.dataset.field][Number(target.dataset.dimension)] = Number(target.value);
  else if (scalars.includes(target.id)) {
    config[target.id] = Number(target.value);
    if (target.id === "api") config.executor = 0;
    if (target.id === "hierarchy") config.worker = config.hierarchy ? 4 : 0;
  }
  if (target.id !== "preset") $("preset").value = target.id === "dim" ? "tiles" : "custom";
  outerIteration = innerIteration = 0;
  controls();
  recompute();
});
$("workers").addEventListener("click", event => {
  const button = event.target.closest("[data-worker]");
  if (!button || !trace) return;
  selectedWorker = Number(button.dataset.worker);
  selectedAssignment = null;
  render();
});
function selectCell(event) {
  const cell = event.target.closest("[data-cell]");
  if (!cell || !trace) return;
  const entries = owners.get(cell.dataset.cell);
  if (!entries?.length) return;
  const current = entries.findIndex(entry => entry.worker.id === selectedWorker && entry.assignment === selectedAssignment);
  const entry = entries[(current + 1) % entries.length];
  selectedWorker = entry.worker.id;
  selectedAssignment = entry.assignment;
  outerIteration = entry.assignment.outerIteration;
  innerIteration = entry.assignment.innerIteration;
  const key = cell.dataset.cell;
  render();
  document.querySelector(`[data-cell="${key}"]`)?.focus({preventScroll: true});
}
$("data").addEventListener("click", selectCell);
$("data").addEventListener("keydown", event => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectCell(event); }
});
$("labels").addEventListener("change", render);
$("all-iterations").addEventListener("change", render);
for (const name of ["outer", "inner"]) $(`${name}-iteration`).addEventListener("change", event => {
  const value = Math.max(0, Number(event.target.value) || 0);
  if (name === "outer") outerIteration = value; else innerIteration = value;
  selectedAssignment = null;
  render();
});
for (const tab of document.querySelectorAll("[data-tab]")) tab.addEventListener("click", () => {
  for (const other of document.querySelectorAll("[data-tab]")) {
    const active = other === tab;
    other.setAttribute("aria-selected", String(active));
    $(other.dataset.tab).hidden = !active;
  }
});
$("reset").addEventListener("click", () => {
  config = defaults(config.dim); selectedWorker = 0; outerIteration = innerIteration = 0;
  $("preset").value = "tiles"; controls(); recompute();
});
$("share").addEventListener("click", async () => {
  const url = new URL(location.href);
  url.hash = encodeURIComponent(JSON.stringify(config));
  history.replaceState(null, "", url);
  try { await navigator.clipboard.writeText(url.href); status("Configuration link copied."); }
  catch { status("Configuration is in the address bar. Copy the URL to share it."); }
});
if (location.hash) $("preset").value = "custom";
controls();
try {
  manifest = await (await fetch("./manifest.json")).json();
  $("download-validator").hidden = !manifest.singleHeader;
  $("download-validator").title = `Godbolt imports ${manifest.godboltHeaderUrl}; this may differ from the local checkout.`;
  const {default: createEngine} = await import("./engine.mjs");
  engine = await createEngine();
  $("provenance").textContent = `alpaka ${manifest.revision.slice(0, 12)} · source hash ${manifest.sourceHash.slice(0, 16)} · no external runtime services`;
  recompute();
} catch (error) {
  status(`Could not load the WASM engine. Build it first and serve over HTTP (not file://). ${error.message}`, true);
}

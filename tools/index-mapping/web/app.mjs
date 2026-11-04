// SPDX-License-Identifier: MPL-2.0
import {defaults, eligible, executors, fromHash, innerRange, layouts, linkedRanges,
  resolutionDescription, scalars, serialize, traversals, tuple, workers} from "./model.mjs";
import {assignmentPhase, iterationKey, visibleVisits} from "./visualization.mjs";
import {initializeTimeline} from "./timeline.mjs";
import {assignmentsFor, createExploration, createTraceModel, resetTraceSelection, selectAssignment,
  selection, selectWorker, setIteration, setStep, toggleWorker} from "./exploration.mjs";
import {initializeInspector} from "./inspector.mjs";
import {initializeLayout} from "./layout.mjs";

const $ = id => document.getElementById(id);
let config;
try { config = fromHash(location.hash); } catch { config = defaults(); }
let engine, manifest, model, launch, playback = null;
const state = createExploration();
const cursor = () => state.position;
const focused = () => {
  const {worker, assignment} = selection(model, state);
  return state.preview ?? (assignment ? {worker, assignment} : null);
};
const renderInspector = initializeInspector(() => ({config, model, state}));
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const canHover = matchMedia("(hover: hover)");
const coarsePointer = matchMedia("(pointer: coarse)");
const status = (message, error = false) => {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
};

for (const select of document.querySelectorAll("[data-policy]")) {
  const values = select.dataset.policy === "layout" ? layouts : traversals;
  values.forEach((name, value) => select.add(new Option(name, value)));
}
workers.forEach((name, value) => $("worker").add(new Option(name, value)));
const zeroBasedFields = new Set(["begin", "outerBegin", "outerEnd", "innerBegin", "innerEnd", "directBegin", "directEnd"]);

function controls() {
  for (const key of scalars) if ($(key)) {
    if ($(key).type === "checkbox") $(key).checked = Boolean(config[key]);
    else $(key).value = config[key];
  }
  $("executor").replaceChildren();
  [0, ...(config.api === 0 ? [1, 2, 3] : [config.api + 3])].forEach(value =>
    $("executor").add(new Option(executors[value], value)));
  $("executor").value = config.executor;
  for (const container of document.querySelectorAll("[data-vector]")) {
    const key = container.dataset.vector;
    const values = config.spec && ["blocks", "threads"].includes(key) ? launch?.[key] : config[key];
    const inputs = container.querySelectorAll("input");
    // A change fires during blur: replacing inputs here also removes the next focus target.
    if (inputs.length === config.dim) {
      inputs.forEach((input, d) => { input.value = values?.[d] ?? ""; });
      continue;
    }
    const label = document.createElement("span");
    label.textContent = container.dataset.label;
    const row = document.createElement("div");
    row.className = "vector-inputs";
    for (let d = 0; d < config.dim; ++d) {
      if (d) row.append("×");
      const input = document.createElement("input");
      input.type = "number";
      input.min = zeroBasedFields.has(key) ? 0 : 1;
      input.max = 64;
      input.step = 1;
      input.value = values?.[d] ?? "";
      input.dataset.field = key;
      input.dataset.dimension = d;
      input.setAttribute("aria-label", `${container.dataset.label}, ${config.dim === 1 ? "x" : d === 0 ? "row" : "column"}`);
      row.append(input);
    }
    container.replaceChildren(label, row);
  }
  $("frames-field").hidden = !config.spec;
  $("frame-extent-field").hidden = !config.spec;
  $("frame-type-field").hidden = !config.spec;
  $("device-limit-field").hidden = !config.spec || config.api === 0;
  $("maxThreadsPerBlock").disabled = Boolean(config.frameType);
  $("compile-limit").hidden = !config.spec || !config.frameType || config.api === 0;
  $("compile-limit").textContent = `CVec uses alpaka's compile-time thread limit: ${launch?.threadLimit ?? "—"}. The runtime device limit above is ignored.`;
  $("inner-range-choice").hidden = !config.hierarchy;
  $("direct-range-fields").hidden = Boolean(config.hierarchy);
  $("tile-range-controls").hidden = !config.hierarchy || Boolean(config.innerRangeMode);
  $("inner-range-fields").hidden = !config.hierarchy || !config.innerRangeMode;
  for (const input of document.querySelectorAll('[data-field="blocks"], [data-field="threads"]'))
    input.disabled = Boolean(config.spec);
  for (const input of document.querySelectorAll('[data-field="outerBegin"], [data-field="outerEnd"]'))
    input.disabled = Boolean(config.outerDomainBounds);
  for (const input of document.querySelectorAll('[data-field="directBegin"], [data-field="directEnd"]'))
    input.disabled = Boolean(config.directDomainBounds);
  for (const input of document.querySelectorAll('[data-field="tile"]'))
    input.disabled = Boolean(config.tileOuterStride);
  $("outer-card").hidden = !config.hierarchy;
  $("show-composition").disabled = !config.hierarchy;
  $("labels").querySelector('[value="sum"]').disabled = !config.hierarchy;
  $("labels").value = state.labels;
  $("inner-level").textContent = config.hierarchy ? "2" : "1";
  $("inner-title").textContent = config.hierarchy ? "Relative inner indices" : "Logical range";
  const local = innerRange(config);
  $("inner-hint").textContent = config.hierarchy
    ? `Inner range: [${tuple(local.begin)}, ${tuple(local.end)}), stride ${tuple(local.stride)}. global = outerIdx + innerIdx.`
    : `Logical range: [${tuple(config.directBegin)}, ${tuple(config.directEnd)}), stride ${tuple(config.stride)}. Desired coverage is separate.`;
  $("assumption").textContent = config.spec
    ? resolutionDescription(config, launch) || "FrameSpec: resolving worker layout for the selected device profile…"
    : "Explicit block/thread extents. This does not execute a GPU kernel.";
  $("assumption").classList.toggle("assumption", Boolean(config.spec));
  for (const option of $("worker").options) {
    option.disabled = config.dim === 2 && [1, 3, 5].includes(Number(option.value));
  }
  $("outerWorker").options[1].disabled = config.dim === 2;
  for (const input of document.querySelectorAll('[data-field="stride"]')) input.disabled = Boolean(config.hierarchy);
  const blocks = config.spec ? launch?.blocks : config.blocks;
  const threads = config.spec ? launch?.threads : config.threads;
  $("launch-summary").textContent = `${["Host / CPU", "CUDA", "HIP", "oneAPI"][config.api] ?? "Unknown API"} · blocks ${blocks ? tuple(blocks) : "—"} · threads/block ${threads ? tuple(threads) : "—"}${config.spec ? " · derived ThreadSpec" : ""}`;
}

function updateIterations() {
  const {outerIteration, innerIteration} = selection(model, state);
  const maxOuter = model.timeline.steps.at(-1)?.outerIteration ?? 0;
  const maxInner = model.maxInner.get(outerIteration) ?? 0;
  for (const [name, value, max] of [["outer", outerIteration, maxOuter], ["inner", innerIteration, maxInner]]) {
    const input = $(`${name}-iteration`);
    input.value = value;
    input.max = max;
    input.disabled = state.allIterations || (name === "outer" && !config.hierarchy);
  }
}

function workerLabel(id) {
  return `T${id}${id === state.selectedWorker ? ' · <span class="selected-badge">Selected</span>' : ""}`;
}
let workersModel;
function drawWorkers() {
  if (workersModel !== model) {
    $("worker-count").textContent = `${model.trace.workers.length} workers`;
    $("workers").innerHTML = [...model.blocks.values()].map(group =>
      `<section class="block-group" style="--block-color:${model.palette.get(group[0].id).block}"><h4>Block ${tuple(group[0].block)}</h4><div class="thread-buttons">${group.map(worker =>
        `<button type="button" data-worker="${worker.id}" class="thread" style="--thread-color:${model.palette.get(worker.id).current}"
          title="Worker ${worker.id}: ${worker.assignments.length} assignments">${tuple(worker.thread)}<small></small></button>`).join("")}</div></section>`).join("");
    workersModel = model;
  }
  for (const button of $("workers").querySelectorAll("[data-worker]")) {
    const id = Number(button.dataset.worker), selected = id === state.selectedWorker;
    if (button.getAttribute("aria-pressed") === String(selected)) continue;
    button.setAttribute("aria-pressed", String(selected));
    button.title = `Worker ${id}: ${model.workersById.get(id).assignments.length} assignments${selected ? " · Click again to deselect" : ""}`;
    button.classList.toggle("active", selected);
    button.querySelector("small").innerHTML = workerLabel(id);
  }
  const selectedButton = $("workers").querySelector('[aria-pressed="true"]');
  if (!selectedButton) return;
  const button = selectedButton.getBoundingClientRect();
  const viewport = $("workers").getBoundingClientRect();
  if (button.top < viewport.top) $("workers").scrollTop -= viewport.top - button.top;
  else if (button.bottom > viewport.top + $("workers").clientHeight)
    $("workers").scrollTop += button.bottom - viewport.top - $("workers").clientHeight;
}

let dataModel;
const dataCells = new Map();
function drawData() {
  const {owners, timeline, palette} = model;
  const {selectedWorker, preview} = state;
  const {assignment: selectedAssignment} = selection(model, state);
  const initialize = dataModel !== model;
  if (initialize) dataCells.clear();
  const rows = config.dim === 2 ? config.end[0] : 1;
  const cols = config.end.at(-1);
  const cell = config.dim === 2 ? 48 : 56;
  const minimumCell = coarsePointer.matches ? 52 : 44;
  const pad = 30, width = cols * cell + pad * 2, height = rows * cell + pad * 2;
  const selected = new Set(assignmentsFor(model, state).filter(a => a.inBounds).map(a => a.index.join(",")));
  const hasSelection = selected.size > 0;
  const focus = focused();
  const showComposition = config.hierarchy && state.showComposition && focus;
  const step = cursor();
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" style="min-width:${cols * minimumCell + pad * 2}px;min-height:${rows * minimumCell + pad * 2}px" role="group" aria-label="Logical data indices">
    <defs><pattern id="excluded" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M0 6L6 0" stroke="#d7dce1" stroke-width="1"/></pattern>
    <marker id="offset-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="context-stroke"/></marker></defs>`;
  for (let row = 0; row < rows; ++row) for (let col = 0; col < cols; ++col) {
    const index = config.dim === 2 ? [row, col] : [col], key = index.join(",");
    const entries = owners.get(key) ?? [], unique = [...new Set(entries.map(entry => entry.worker.id))];
    const active = selected.has(key);
    const isFocused = selectedAssignment?.inBounds && selectedAssignment.index.join(",") === key;
    const isPreview = preview?.assignment.inBounds && preview.assignment.index.join(",") === key;
    const bounds = `x="${pad + col * cell + 3}" y="${pad + row * cell + 3}" width="${cell - 6}" height="${cell - 6}" rx="6"`;
    const valid = eligible(config, index);
    const visits = visibleVisits(entries, timeline, step, state.keepHistory);
    const current = visits.filter(e => assignmentPhase(e.assignment, timeline, step) === "current");
    const phase = current.length ? "current" : visits.length ? "past" : "future";
    const paint = current.length ? current : visits;
    const coverage = !valid ? "outside" : !entries.length ? "uncovered" : "assigned";
    // Only assigned cells are softened; uncovered and outside-target indicators keep their meaning.
    const emphasis = active ? "selected" : hasSelection && coverage === "assigned" ? "muted" : "";
    const fill = !valid ? "#eff1f3" : !entries.length ? "url(#excluded)"
      : paint.length ? palette.get(paint[0].worker.id)[phase] : "#fff";
    const labelEntry = entries.find(e => e.worker.id === focus?.worker.id && e.assignment === focus.assignment)
      ?? current[0] ?? entries[0];
    const label = state.labels === "linear" ? (entries[0]?.assignment.linear ?? "—")
      : state.labels === "owner" ? (unique.length > 1 ? `${unique.length}×` : unique[0] ?? "—")
      : state.labels === "sum" && labelEntry ? (config.dim === 1
        ? `${labelEntry.assignment.origin[0]} + ${labelEntry.assignment.local[0]}`
        : `<tspan x="${pad + col * cell + cell / 2}" dy="-5">${tuple(labelEntry.assignment.origin)}</tspan><tspan x="${pad + col * cell + cell / 2}" dy="12">+${tuple(labelEntry.assignment.local)}</tspan>`)
      : config.dim === 2 ? `${row},${col}` : `${col}`;
    const ink = paint.length ? palette.get(paint[0].worker.id).ink : "#000";
    const ariaLabel = `Index ${tuple(index)}, ${coverage === "uncovered" ? "uncovered target" : coverage === "outside" ? "outside desired domain" : `${unique.length} owners, ${phase} at logical step ${step + 1}`}${active ? ", assigned to selected worker" : ""}`;
    const content = `<title>Index ${tuple(index)} · ${entries.length} assignments · workers ${unique.join(", ") || "none"}${labelEntry && config.hierarchy ? ` · ${tuple(labelEntry.assignment.origin)} + ${tuple(labelEntry.assignment.local)}` : ""}</title>
      <rect class="cell-fill" ${bounds} fill="${fill}" stroke="#d7dce1" stroke-width="1"/>
      ${active ? `<rect class="selection-halo" ${bounds} aria-hidden="true"/><rect class="selection-ring" ${bounds} aria-hidden="true"/>` : ""}
      ${isPreview ? `<rect class="preview-ring" x="${pad + col * cell + 8}" y="${pad + row * cell + 8}" width="${cell - 16}" height="${cell - 16}" rx="3" aria-hidden="true"/>` : ""}
      ${visits.length > 1 ? visits.slice(-8).map((entry, i, stripes) => `<rect class="visit-stripe" x="${pad + col * cell + 6 + i * (cell - 12) / stripes.length}" y="${pad + row * cell + cell - 10}" width="${(cell - 12) / stripes.length}" height="4" fill="${palette.get(entry.worker.id)[assignmentPhase(entry.assignment, timeline, step)]}"/>`).join("") : ""}
      <text class="${phase === "current" ? "on-color" : ""} ${state.labels === "sum" ? "sum-label" : ""}" x="${pad + col * cell + cell / 2}" y="${pad + row * cell + cell / 2 + 4}" text-anchor="middle">${label}</text>
      ${entries.length > 1 ? `<text class="visit-count ${phase === "current" ? "on-color" : ""}" x="${pad + col * cell + cell - 6}" y="${pad + row * cell + 13}" text-anchor="end">${entries.length}×</text>` : ""}
      ${isFocused ? `<path class="assignment-marker" d="M${pad + col * cell + cell / 2 - 4},${pad + row * cell + 7} h8 l-4,7 Z" aria-hidden="true"/>` : ""}`;
    if (initialize) {
      svg += `<g data-cell="${key}" data-owner-count="${unique.length}" role="button" tabindex="0">${content}</g>`;
      dataCells.set(key, {content});
    } else {
      const cached = dataCells.get(key);
      if (cached.content !== content) { cached.node.innerHTML = content; cached.content = content; }
    }
    const cached = dataCells.get(key);
    cached.attributes = {highlighted: active, emphasis, phase, coverage, visitCount: visits.length, ink, ariaLabel};
  }
  const local = innerRange(config);
  const rangeRect = origin => {
    const start = origin.map((v, d) => Math.min(config.end[d], v + local.begin[d]));
    const end = origin.map((v, d) => Math.min(config.end[d], v + local.end[d]));
    return `x="${pad + start.at(-1) * cell}" y="${pad + (config.dim === 2 ? start[0] : 0) * cell}" width="${Math.max(0, end.at(-1) - start.at(-1)) * cell}" height="${(config.dim === 2 ? Math.max(0, end[0] - start[0]) : 1) * cell}"`;
  };
  if (initialize && config.hierarchy) {
    const origins = new Map(model.trace.workers.flatMap(w => w.assignments.map(a => [a.origin.join(","), a.origin])));
    for (const origin of origins.values())
      if (origin.every((v, d) => v + local.begin[d] < config.end[d]))
        svg += `<rect class="tile-outline" ${rangeRect(origin)} rx="8"/>`;
  }
  let overlay = "";
  if (showComposition) {
    const {worker, assignment: a} = focus;
    // Route through cell corners, keeping coordinate/sum labels unobscured.
    const x = v => pad + v.at(-1) * cell + 10;
    const y = v => pad + (config.dim === 2 ? v[0] : 0) * cell + 10;
    overlay = `<g class="composition-overlay" aria-hidden="true">
      <rect ${rangeRect(a.origin)} rx="8" fill="none" stroke="${palette.get(worker.id).block}" stroke-width="3"/>
      <circle cx="${x(a.origin)}" cy="${y(a.origin)}" r="4" fill="white" stroke="${palette.get(worker.id).block}" stroke-width="2"/>
      ${a.inBounds && a.local.some(v => v !== 0) ? `<path d="M${x(a.origin)},${y(a.origin)}${y(a.origin) !== y(a.index) ? ` V${y(a.index)}` : ""}${x(a.origin) !== x(a.index) ? ` H${x(a.index)}` : ""}" fill="none" stroke="${palette.get(worker.id).current}" stroke-width="3" stroke-dasharray="5 3" marker-end="url(#offset-arrow)"/>` : ""}</g>`;
  }
  if (initialize) {
    for (let col = 0; col < cols; ++col) svg += `<text class="axis" x="${pad + col * cell + cell / 2}" y="17" text-anchor="middle">${col}</text>`;
    if (config.dim === 2) for (let row = 0; row < rows; ++row) svg += `<text class="axis" x="15" y="${pad + row * cell + cell / 2 + 4}" text-anchor="middle">${row}</text>`;
    $("data").innerHTML = svg + '<g id="composition-overlay"></g></svg>';
    for (const node of $("data").querySelectorAll("[data-cell]")) dataCells.get(node.dataset.cell).node = node;
    dataModel = model;
  }
  for (const {node, attributes} of dataCells.values()) {
    node.dataset.highlighted = attributes.highlighted;
    node.dataset.emphasis = attributes.emphasis;
    node.dataset.phase = attributes.phase;
    node.dataset.coverage = attributes.coverage;
    node.dataset.visitCount = attributes.visitCount;
    node.style.setProperty("--cell-ink", attributes.ink);
    node.setAttribute("aria-label", attributes.ariaLabel);
  }
  $("composition-overlay").innerHTML = overlay;
  const selectionKey = selectedAssignment ? `${selectedWorker}:${iterationKey(selectedAssignment)}:${selectedAssignment.index}` : "";
  if ($("data").dataset.selectionKey !== selectionKey && selectedAssignment?.inBounds) {
    const target = $("data").querySelector(`[data-cell="${selectedAssignment.index.join(",")}"] > rect`).getBoundingClientRect();
    const viewport = $("data").getBoundingClientRect();
    if (target.left < viewport.left) $("data").scrollLeft -= viewport.left - target.left + 8;
    else if (target.right > viewport.left + $("data").clientWidth)
      $("data").scrollLeft += target.right - viewport.left - $("data").clientWidth + 8;
    if (target.top < viewport.top) $("data").scrollTop -= viewport.top - target.top + 8;
    else if (target.bottom > viewport.top + $("data").clientHeight)
      $("data").scrollTop += target.bottom - viewport.top - $("data").clientHeight + 8;
  }
  $("data").dataset.selectionKey = selectionKey;
  $("domain-size").textContent = `${rows} × ${cols}`;
}

function drawComposition() {
  const {palette} = model;
  const {preview} = state;
  const container = $("composition");
  container.hidden = !config.hierarchy || !state.showComposition;
  if (container.hidden) return;
  const content = $("composition-content");
  const focus = focused();
  if (!focus) {
    const hasWorker = state.selectedWorker !== null;
    $("composition-summary").textContent = hasWorker ? "No assignment at this step" : "No worker selected";
    content.innerHTML = `<p class="hint">${hasWorker ? "No assignment for this worker at the current step. Select another worker or timeline event." : "Select a worker or timeline event to inspect its index composition."}</p>`;
    return;
  }
  const {worker, assignment: a} = focus;
  $("composition-summary").textContent = `${tuple(a.index)} = ${tuple(a.origin)} + ${tuple(a.local)}${a.inBounds ? "" : " · rejected by guard"}`;
  const color = palette.get(worker.id);
  let inset = "";
  if (config.dim === 2) {
    const local = innerRange(config);
    const [rows, cols] = local.end.map((v, d) => v - local.begin[d]);
    const cell = Math.min(24, 140 / Math.max(1, rows, cols));
    const w = cols * cell, h = rows * cell;
    inset = `<div class="local-inset"><span>${config.innerRangeMode ? "Relative inner range" : "Local tile · zero-based"}</span><svg viewBox="-20 -20 ${w + 40} ${h + 40}" role="img" aria-label="Inner index ${tuple(a.local)} in range [${tuple(local.begin)}, ${tuple(local.end)}), stride ${tuple(local.stride)}">
      ${Array.from({length: rows + 1}, (_, i) => `<path d="M0 ${i * cell}H${w}"/>`).join("")}
      ${Array.from({length: cols + 1}, (_, i) => `<path d="M${i * cell} 0V${h}"/>`).join("")}
      <rect x="${(a.local[1] - local.begin[1]) * cell}" y="${(a.local[0] - local.begin[0]) * cell}" width="${cell}" height="${cell}" fill="${color.current}"/>
      <text x="-4" y="-5">${tuple(local.begin)}</text><text x="${w / 2}" y="-5" text-anchor="middle">column →</text>
      </svg><span>↓ ${a.local[0]} rows · → ${a.local[1]} columns</span></div>`;
  }
  content.innerHTML = `<div class="composition-body"><p class="hint">${preview ? "Preview" : "Selected"} · Block ${tuple(worker.block)} / Thread ${tuple(worker.thread)} · outer ${a.outerIteration}, inner ${a.innerIteration}</p>
    <div class="sum-equation" data-origin="${a.origin.join(",")}" data-local="${a.local.join(",")}" data-global="${a.index.join(",")}">
      <div><small>Global index</small><strong>${tuple(a.index)}</strong></div><b>=</b>
      <div style="--term-color:${color.block}"><small>Outer-loop offset</small><strong>${tuple(a.origin)}</strong></div><b>+</b>
      <div style="--term-color:${color.current}"><small>${[2, 3].includes(config.worker) ? "Inner-loop local index" : "Thread-level local index"}</small><strong>${tuple(a.local)}</strong></div>
    </div><p class="hint">Loop-returned indices, not raw block/thread IDs. ${workers[config.worker]} supplies the local index.${a.inBounds ? "" : ' <strong class="guard-message">Outside the logical range — rejected by guard.</strong>'}</p></div>${inset}`;
}

function updateTimelineControls() {
  const timeline = model?.timeline;
  const count = timeline?.steps.length ?? 0, step = timeline?.steps[cursor()];
  $("timeline-step").max = Math.max(0, count - 1);
  $("timeline-step").value = cursor();
  $("timeline-step").disabled = count < 2;
  $("timeline-step").setAttribute("aria-valuetext", step
    ? `Step ${cursor() + 1} of ${count}, outer ${step.outerIteration}, inner ${step.innerIteration}` : "No assignments");
  $("step-label").textContent = count ? `${cursor() + 1} / ${count}` : "0 / 0";
  $("step-back").disabled = !count || cursor() === 0;
  $("step-forward").disabled = !count || cursor() === count - 1;
  $("play").disabled = count < 2 || reducedMotion.matches;
  $("play").title = reducedMotion.matches ? "Autoplay disabled for reduced motion; use single-step or the slider." : "Play logical steps (not execution time)";
  $("timeline-summary").textContent = step
    ? `Outer ${step.outerIteration} · Inner ${step.innerIteration} · ${step.events.filter(e => e.assignment.inBounds).length} in-bounds assignments · ${step.events.filter(e => !e.assignment.inBounds).length} guarded`
    : "No assignments in this range.";
}

const renderTimeline = initializeTimeline($("timeline"));
function drawTimeline() {
  updateTimelineControls();
  if (!$("timeline-details").hidden) renderTimeline(model, config, state);
}

function stopPlayback() {
  if (playback !== null) clearInterval(playback);
  playback = null;
  $("play").textContent = "Play";
  $("play").setAttribute("aria-pressed", "false");
}
function moveToStep(position) {
  if (!model) return;
  setStep(model, state, position);
  render();
}

function reportPresentationError(error) {
  stopPlayback();
  status(`Could not render the visualization: ${error.message}`, true);
  console.error(error);
}
function render() {
  if (!model) return;
  try {
    const {worker, assignment} = selection(model, state);
    $("worker-selection").textContent = worker
      ? `Selected: T${worker.id} · Block ${tuple(worker.block)} · Thread ${tuple(worker.thread)}${assignment ? "" : " · Idle at this step"}`
      : "No worker selected";
    // Browsers can restore form values independently of application state on reload/back.
    // Always render checkbox state from the same source that drives the SVG overlays.
    $("show-composition").checked = state.showComposition;
    $("keep-history").checked = state.keepHistory;
    $("all-iterations").checked = state.allIterations;
    $("timeline-scope").disabled = !worker;
    $("timeline-scope").value = worker ? state.timelineScope : "all";
    updateIterations();
    drawWorkers();
    drawData();
    drawComposition();
    drawTimeline();
    renderInspector();
  } catch (error) {
    reportPresentationError(error);
  }
}
function recompute() {
  stopPlayback();
  state.preview = null;
  delete $("data").dataset.selectionKey;
  if (!engine) return;
  let result;
  try {
    result = JSON.parse(engine.ccall("mapping_trace", "string", ["string"], [serialize(config)]));
    launch = result;
    controls();
    if (result.error) throw Error(result.error);
    if (result.build !== manifest.build) throw Error("Engine and manifest do not match. Rebuild the prototype.");
  } catch (error) {
    model = null;
    for (const id of ["workers", "worker-count", "data", "domain-size", "composition-content", "composition-summary", "timeline", "worker-selection"]) $(id).replaceChildren();
    renderInspector();
    updateTimelineControls();
    $("composition").hidden = true;
    status(error.message, true);
    return;
  }
  try {
    model = createTraceModel(result, config);
    resetTraceSelection(model, state);
  } catch (error) {
    reportPresentationError(error);
    return;
  }
  status(config.spec ? "Live C++ / WebAssembly · FrameSpec resolved for this device profile · synthetic worker contexts"
    : "Live C++ / WebAssembly · real alpaka index iteration · synthetic worker contexts");
  render();
}

function updateConfiguration() {
  config = linkedRanges(config);
  launch = null;
  controls();
  recompute();
}

$("config-form").addEventListener("submit", event => event.preventDefault());
function changeConfiguration(event) {
  const target = event.target;
  if (target.id === "preset") {
    config = defaults(config.dim);
    if (target.value === "direct") { config.hierarchy = 0; config.worker = 0; }
    if (target.value === "uneven") config.end = config.dim === 2 ? [3, 7] : [15];
    if (target.value === "host") { config.api = 0; config.threads = Array(config.dim).fill(1); }
  } else if (target.id === "dim") config = defaults(Number(target.value));
  else if (target.dataset.field) config[target.dataset.field][Number(target.dataset.dimension)] = Number(target.value);
  else if (scalars.includes(target.id)) {
    config[target.id] = target.type === "checkbox" ? Number(target.checked) : Number(target.value);
    if (target.id === "api") {
      config.executor = 0;
      config.maxThreadsPerBlock = config.api === 3 ? 256 : 1024;
    }
    if (target.id === "hierarchy") config.worker = config.hierarchy ? 4 : 0;
  }
  if (target.id !== "preset") $("preset").value = target.id === "dim" ? "tiles" : "custom";
  if (!config.hierarchy && state.labels === "sum") state.labels = "index";
  updateConfiguration();
}
$("config-form").addEventListener("change", changeConfiguration);
$("workspace-toolbar").addEventListener("change", changeConfiguration);
$("workers").addEventListener("click", event => {
  const button = event.target.closest("[data-worker]");
  if (!button || !model) return;
  stopPlayback();
  toggleWorker(model, state, Number(button.dataset.worker));
  render();
});
function selectCell(event) {
  const cell = event.target.closest("[data-cell]");
  if (!cell || !model) return;
  const entries = model.owners.get(cell.dataset.cell);
  if (!entries?.length) return;
  const {assignment} = selection(model, state);
  const current = entries.findIndex(entry => entry.worker.id === state.selectedWorker && entry.assignment === assignment);
  stopPlayback();
  selectAssignment(model, state, entries[(current + 1) % entries.length]);
  render();
  cell.focus({preventScroll: true});
}
$("data").addEventListener("click", selectCell);
$("data").addEventListener("keydown", event => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectCell(event); }
});
for (const [id, key] of [["show-composition", "showComposition"], ["keep-history", "keepHistory"],
  ["timeline-scope", "timelineScope"], ["labels", "labels"], ["all-iterations", "allIterations"]]) {
  $(id).addEventListener("change", event => {
    state[key] = event.target.type === "checkbox" ? event.target.checked : event.target.value;
    render();
  });
}
$("expand-timeline").addEventListener("click", () => {
  const expanded = $("timeline-panel").classList.toggle("expanded");
  $("timeline-details").hidden = !expanded;
  $("expand-timeline").setAttribute("aria-expanded", String(expanded));
  $("expand-timeline").textContent = expanded ? "Collapse timeline" : "Expand timeline";
  $("timeline").dataset.revealSelection = "true";
  if (model) drawTimeline();
});
$("step-back").addEventListener("click", () => { stopPlayback(); moveToStep(cursor() - 1); });
$("step-forward").addEventListener("click", () => { stopPlayback(); moveToStep(cursor() + 1); });
$("timeline-step").addEventListener("input", event => { stopPlayback(); moveToStep(Number(event.target.value)); });
$("play").addEventListener("click", () => {
  if (playback !== null) { stopPlayback(); return; }
  if (!model || reducedMotion.matches || model.timeline.steps.length < 2) return;
  if (cursor() === model.timeline.steps.length - 1) moveToStep(0);
  $("play").textContent = "Pause";
  $("play").setAttribute("aria-pressed", "true");
  playback = setInterval(() => {
    moveToStep(cursor() + 1);
    if (cursor() === model.timeline.steps.length - 1) stopPlayback();
  }, 900);
});
reducedMotion.addEventListener("change", () => { stopPlayback(); updateTimelineControls(); });
document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlayback(); });
$("timeline").addEventListener("click", event => {
  if (!model) return;
  const button = event.target.closest("button");
  if (!button) return;
  stopPlayback();
  state.preview = null;
  if (button.dataset.collapseBlock !== undefined) {
    const key = button.dataset.collapseBlock;
    if (state.collapsedBlocks.has(key)) state.collapsedBlocks.delete(key); else state.collapsedBlocks.add(key);
  } else if (button.dataset.selectWorker !== undefined) {
    toggleWorker(model, state, Number(button.dataset.selectWorker));
  } else {
    const step = Number(button.dataset.eventStep ?? button.dataset.step);
    if (button.dataset.eventWorker !== undefined)
      selectWorker(model, state, Number(button.dataset.eventWorker));
    setStep(model, state, step);
  }
  render();
});

function previewEntry(entry) {
  if (!model) return;
  if (entry?.worker.id === state.selectedWorker && entry?.assignment === selection(model, state).assignment) entry = null;
  if (state.preview?.worker === entry?.worker && state.preview?.assignment === entry?.assignment) return;
  state.preview = entry;
  drawData();
  drawComposition();
}
$("workers").addEventListener("mouseover", event => {
  const button = event.target.closest("[data-worker]");
  if (!button || !model || playback !== null || !canHover.matches) return;
  const worker = model.workersById.get(Number(button.dataset.worker));
  const assignment = model.timeline.byWorker.get(worker.id).get(cursor());
  previewEntry(assignment ? {worker, assignment} : null);
});
$("data").addEventListener("mouseover", event => {
  const cell = event.target.closest("[data-cell]");
  if (!cell || !model || playback !== null || !canHover.matches) return;
  const entries = model.owners.get(cell.dataset.cell) ?? [];
  const atStep = entries.filter(e => model.timeline.positions.get(iterationKey(e.assignment)) === cursor());
  previewEntry(atStep.find(e => e.worker.id === state.selectedWorker) ?? atStep[0] ?? entries[0] ?? null);
});
for (const id of ["workers", "data"]) $(id).addEventListener("mouseleave", () => previewEntry(null));

for (const name of ["outer", "inner"]) $(`${name}-iteration`).addEventListener("change", event => {
  if (!model) return;
  stopPlayback();
  const value = Math.max(0, Math.trunc(Number(event.target.value)) || 0);
  setIteration(model, state, name, value);
  render();
});
$("reset").addEventListener("click", () => {
  config = defaults(config.dim);
  state.selectedWorker = 0;
  state.collapsedBlocks.clear();
  $("preset").value = "tiles";
  updateConfiguration();
});
$("share").addEventListener("click", async () => {
  const url = new URL(location.href);
  url.hash = encodeURIComponent(JSON.stringify(config));
  history.replaceState(null, "", url);
  try { await navigator.clipboard.writeText(url.href); status("Configuration link copied."); }
  catch { status("Configuration is in the address bar. Copy the URL to share it."); }
});
if (location.hash) $("preset").value = "custom";
initializeLayout(render, visible => { state.showComposition = visible; });
updateConfiguration();
try {
  manifest = await (await fetch("./manifest.json")).json();
  $("download-validator").hidden = !manifest.singleHeader;
  $("download-validator").title = `Godbolt imports ${manifest.godboltHeaderUrl}; this may differ from the local checkout.`;
  const {default: createEngine} = await import("./engine.mjs");
  engine = await createEngine();
  $("provenance").textContent = `alpaka ${manifest.revision.slice(0, 12)} · source hash ${manifest.sourceHash.slice(0, 16)} · no external runtime services`;
} catch (error) {
  status(`Could not load the WASM engine. Build it first and serve over HTTP (not file://). ${error.message}`, true);
}
if (engine) recompute();

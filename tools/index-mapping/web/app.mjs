// SPDX-License-Identifier: MPL-2.0
import {apis, defaults, eligible, executors, fromHash, layouts, ownersOf,
  scalars, serialize, snippet, traversals, tuple, workers} from "./model.mjs";

import {assignmentPhase, buildTimeline, iterationKey, timelineWindow, visibleVisits, workerPalette} from "./visualization.mjs";
import {initializeLayout} from "./layout.mjs";

const $ = id => document.getElementById(id);
let config;
try { config = fromHash(location.hash); } catch { config = defaults(); }
let engine, manifest, trace, owners, selectedWorker = 0, selectedAssignment = null;
let outerIteration = 0, innerIteration = 0;
let timeline, palette, preview = null, playback = null;
const collapsedBlocks = new Set();
const cursor = () => timeline?.positions.get(`${outerIteration}:${innerIteration}`) ?? 0;
const focused = () => preview ?? (selectedAssignment ? {worker: currentWorker(), assignment: selectedAssignment} : null);
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

function controls() {
  for (const key of scalars) if ($(key)) $(key).value = config[key];
  $("executor").replaceChildren();
  [0, ...(config.api === 0 ? [1, 2, 3] : [config.api + 3])].forEach(value =>
    $("executor").add(new Option(executors[value], value)));
  $("executor").value = config.executor;
  for (const container of document.querySelectorAll("[data-vector]")) {
    const key = container.dataset.vector;
    const inputs = container.querySelectorAll("input");
    // A change fires during blur: replacing inputs here also removes the next focus target.
    if (inputs.length === config.dim) {
      inputs.forEach((input, d) => { input.value = config[key][d]; });
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
  $("show-composition").disabled = !config.hierarchy;
  $("labels").querySelector('[value="sum"]').disabled = !config.hierarchy;
  if (!config.hierarchy && $("labels").value === "sum") $("labels").value = "index";
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
  $("launch-summary").textContent = `${["Host / CPU", "CUDA", "HIP", "oneAPI"][config.api] ?? "Unknown API"} · blocks ${tuple(config.blocks)} · threads/block ${tuple(config.threads)}${config.spec ? " · assumed ThreadSpec" : ""}`;
}

function currentWorker() { return trace?.workers.find(worker => worker.id === selectedWorker); }
function currentAssignments() {
  return (currentWorker()?.assignments ?? []).filter(a => $("all-iterations").checked
    || (a.outerIteration === outerIteration && a.innerIteration === innerIteration));
}
function updateIterations() {
  const steps = timeline.steps;
  const maxOuter = steps.at(-1)?.outerIteration ?? 0;
  outerIteration = Math.min(outerIteration, maxOuter);
  const maxInner = Math.max(0, ...steps.filter(a => a.outerIteration === outerIteration).map(a => a.innerIteration));
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
      html += `<section class="block-group" style="--block-color:${palette.get(worker.id).block}"><h4>Block ${tuple(worker.block)}</h4><div class="thread-buttons">`;
      lastBlock = key;
    }
    html += `<button type="button" data-worker="${worker.id}" aria-pressed="${worker.id === selectedWorker}"
      class="thread ${worker.id === selectedWorker ? "active" : ""}" style="--thread-color:${palette.get(worker.id).current}"
      title="Worker ${worker.id}: ${worker.assignments.length} assignments">${tuple(worker.thread)}<small>T${worker.id}${worker.id === selectedWorker ? ' · <span class="selected-badge">Selected</span>' : ""}</small></button>`;
  }
  $("workers").innerHTML = html + "</div></section>";
  const button = $("workers").querySelector('[aria-pressed="true"]').getBoundingClientRect();
  const viewport = $("workers").getBoundingClientRect();
  if (button.top < viewport.top) $("workers").scrollTop -= viewport.top - button.top;
  else if (button.bottom > viewport.top + $("workers").clientHeight)
    $("workers").scrollTop += button.bottom - viewport.top - $("workers").clientHeight;
}

function drawData() {
  const focusedCell = document.activeElement?.closest("[data-cell]")?.dataset.cell;
  const rows = config.dim === 2 ? config.end[0] : 1;
  const cols = config.end.at(-1);
  const cell = config.dim === 2 ? 48 : 56;
  const minimumCell = coarsePointer.matches ? 52 : 44;
  const pad = 30, width = cols * cell + pad * 2, height = rows * cell + pad * 2;
  const selected = new Set(currentAssignments().filter(a => a.inBounds).map(a => a.index.join(",")));
  const focus = focused();
  const showComposition = config.hierarchy && $("show-composition").checked && focus;
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
    const visits = visibleVisits(entries, timeline, step, $("keep-history").checked);
    const current = visits.filter(e => assignmentPhase(e.assignment, timeline, step) === "current");
    const phase = current.length ? "current" : visits.length ? "past" : "future";
    const paint = current.length ? current : visits;
    const fill = !valid ? "url(#excluded)" : paint.length ? palette.get(paint[0].worker.id)[phase] : "#fff";
    const labelEntry = entries.find(e => e.worker.id === focus?.worker.id && e.assignment === focus.assignment)
      ?? current[0] ?? entries[0];
    const label = $("labels").value === "linear" ? (entries[0]?.assignment.linear ?? "—")
      : $("labels").value === "owner" ? (unique.length > 1 ? `${unique.length}×` : unique[0] ?? "—")
      : $("labels").value === "sum" && labelEntry ? (config.dim === 1
        ? `${labelEntry.assignment.origin[0]} + ${labelEntry.assignment.local[0]}`
        : `<tspan x="${pad + col * cell + cell / 2}" dy="-5">${tuple(labelEntry.assignment.origin)}</tspan><tspan x="${pad + col * cell + cell / 2}" dy="12">+${tuple(labelEntry.assignment.local)}</tspan>`)
      : config.dim === 2 ? `${row},${col}` : `${col}`;
    svg += `<g data-cell="${key}" data-owner-count="${unique.length}" data-highlighted="${active}" data-phase="${phase}" data-visit-count="${visits.length}" style="--cell-ink:${paint.length ? palette.get(paint[0].worker.id).ink : '#000'}" role="button" tabindex="0"
      aria-label="Index ${tuple(index)}, ${unique.length} owners, ${phase} at logical step ${step + 1}${active ? ", assigned to selected worker" : ""}">
      <title>Index ${tuple(index)} · ${entries.length} assignments · workers ${unique.join(", ") || "none"}${labelEntry && config.hierarchy ? ` · ${tuple(labelEntry.assignment.origin)} + ${tuple(labelEntry.assignment.local)}` : ""}</title>
      <rect class="cell-fill" ${bounds} fill="${fill}" stroke="#d7dce1" stroke-width="1"/>
      ${active ? `<rect class="selection-halo" ${bounds} aria-hidden="true"/><rect class="selection-ring" ${bounds} aria-hidden="true"/>` : ""}
      ${isPreview ? `<rect class="preview-ring" x="${pad + col * cell + 8}" y="${pad + row * cell + 8}" width="${cell - 16}" height="${cell - 16}" rx="3" aria-hidden="true"/>` : ""}
      ${visits.length > 1 ? visits.slice(-8).map((entry, i, stripes) => `<rect x="${pad + col * cell + 6 + i * (cell - 12) / stripes.length}" y="${pad + row * cell + cell - 10}" width="${(cell - 12) / stripes.length}" height="4" fill="${palette.get(entry.worker.id)[assignmentPhase(entry.assignment, timeline, step)]}"/>`).join("") : ""}
      <text class="${phase === "current" ? "on-color" : ""} ${$("labels").value === "sum" ? "sum-label" : ""}" x="${pad + col * cell + cell / 2}" y="${pad + row * cell + cell / 2 + 4}" text-anchor="middle">${label}</text>
      ${entries.length > 1 ? `<text class="visit-count ${phase === "current" ? "on-color" : ""}" x="${pad + col * cell + cell - 6}" y="${pad + row * cell + 13}" text-anchor="end">${entries.length}×</text>` : ""}
      ${isFocused ? `<path class="assignment-marker" d="M${pad + col * cell + cell / 2 - 4},${pad + row * cell + 7} h8 l-4,7 Z" aria-hidden="true"/>` : ""}</g>`;
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
  if (showComposition) {
    const {worker, assignment: a} = focus;
    // Route through cell corners, keeping coordinate/sum labels unobscured.
    const x = v => pad + v.at(-1) * cell + 10;
    const y = v => pad + (config.dim === 2 ? v[0] : 0) * cell + 10;
    const tileRows = Math.min(config.dim === 2 ? config.tile[0] : 1, rows - (config.dim === 2 ? a.origin[0] : 0));
    const tileCols = Math.min(config.tile.at(-1), cols - a.origin.at(-1));
    svg += `<g class="composition-overlay" aria-hidden="true">
      <rect x="${x(a.origin) - 10}" y="${y(a.origin) - 10}" width="${tileCols * cell}" height="${tileRows * cell}" rx="8" fill="none" stroke="${palette.get(worker.id).block}" stroke-width="3"/>
      <circle cx="${x(a.origin)}" cy="${y(a.origin)}" r="4" fill="white" stroke="${palette.get(worker.id).block}" stroke-width="2"/>
      ${a.inBounds && a.local.some(v => v !== 0) ? `<path d="M${x(a.origin)},${y(a.origin)}${y(a.origin) !== y(a.index) ? ` V${y(a.index)}` : ""}${x(a.origin) !== x(a.index) ? ` H${x(a.index)}` : ""}" fill="none" stroke="${palette.get(worker.id).current}" stroke-width="3" stroke-dasharray="5 3" marker-end="url(#offset-arrow)"/>` : ""}</g>`;
  }
  for (let col = 0; col < cols; ++col) svg += `<text class="axis" x="${pad + col * cell + cell / 2}" y="17" text-anchor="middle">${col}</text>`;
  if (config.dim === 2) for (let row = 0; row < rows; ++row) svg += `<text class="axis" x="15" y="${pad + row * cell + cell / 2 + 4}" text-anchor="middle">${row}</text>`;
  $("data").innerHTML = svg + "</svg>";
  if (focusedCell) $("data").querySelector(`[data-cell="${focusedCell}"]`)?.focus({preventScroll: true});
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
  const container = $("composition");
  container.hidden = !config.hierarchy || !$("show-composition").checked;
  if (container.hidden) return;
  const content = $("composition-content");
  const focus = focused();
  if (!focus) {
    $("composition-summary").textContent = "No assignment at this step";
    content.innerHTML = '<p class="hint">No assignment for this worker at the current step. Select another worker or timeline event.</p>';
    return;
  }
  const {worker, assignment: a} = focus;
  $("composition-summary").textContent = `${tuple(a.index)} = ${tuple(a.origin)} + ${tuple(a.local)}${a.inBounds ? "" : " · rejected by guard"}`;
  const color = palette.get(worker.id);
  let inset = "";
  if (config.dim === 2) {
    const cell = Math.min(24, 140 / Math.max(...config.tile));
    const [rows, cols] = config.tile;
    const w = cols * cell, h = rows * cell;
    inset = `<div class="local-inset"><span>Local tile · zero-based</span><svg viewBox="-20 -20 ${w + 40} ${h + 40}" role="img" aria-label="Local index ${tuple(a.local)} in tile ${tuple(config.tile)}">
      ${Array.from({length: rows + 1}, (_, i) => `<path d="M0 ${i * cell}H${w}"/>`).join("")}
      ${Array.from({length: cols + 1}, (_, i) => `<path d="M${i * cell} 0V${h}"/>`).join("")}
      <rect x="${a.local[1] * cell}" y="${a.local[0] * cell}" width="${cell}" height="${cell}" fill="${color.current}"/>
      <text x="-4" y="-5">0</text><text x="${w / 2}" y="-5" text-anchor="middle">column →</text>
      </svg><span>↓ ${a.local[0]} rows · → ${a.local[1]} columns</span></div>`;
  }
  content.innerHTML = `<div class="composition-body"><p class="hint">${preview ? "Preview" : "Selected"} · Block ${tuple(worker.block)} / Thread ${tuple(worker.thread)} · outer ${a.outerIteration}, inner ${a.innerIteration}</p>
    <div class="sum-equation" data-origin="${a.origin.join(",")}" data-local="${a.local.join(",")}" data-global="${a.index.join(",")}">
      <div><small>Global index</small><strong>${tuple(a.index)}</strong></div><b>=</b>
      <div style="--term-color:${color.block}"><small>Block-level tile origin</small><strong>${tuple(a.origin)}</strong></div><b>+</b>
      <div style="--term-color:${color.current}"><small>${[2, 3].includes(config.worker) ? "Inner-loop local index" : "Thread-level local index"}</small><strong>${tuple(a.local)}</strong></div>
    </div><p class="hint">Loop-returned indices, not raw block/thread IDs. ${workers[config.worker]} supplies the local index.${a.inBounds ? "" : ' <strong class="guard-message">Outside the logical range — rejected by guard.</strong>'}</p></div>${inset}`;
}

function updateTimelineControls() {
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

function drawTimeline() {
  updateTimelineControls();
  if ($("timeline-details").hidden) return;
  const container = $("timeline"), position = cursor();
  const {start, end} = timelineWindow(timeline.steps.length, position);
  const steps = timeline.steps.slice(start, end);
  if (!steps.length) { container.innerHTML = '<p class="hint">All workers are idle.</p>'; return; }
  const scrollLeft = container.scrollLeft, scrollTop = container.scrollTop;
  const selectionChanged = container.dataset.selectedWorker !== String(selectedWorker);
  container.dataset.selectedWorker = selectedWorker;
  const selectedBlock = currentWorker().block.join(",");
  const blockWorkers = new Map();
  for (const worker of trace.workers) {
    const key = worker.block.join(",");
    if ($("timeline-scope").value === "block" && key !== selectedBlock) continue;
    if (!blockWorkers.has(key)) blockWorkers.set(key, []);
    blockWorkers.get(key).push(worker);
  }
  let html = `<table class="timeline-table"><caption>Steps ${start + 1}–${end} of ${timeline.steps.length}. Outer / inner loop iterations; blank cells indicate no assignment.</caption><thead><tr><th scope="col">Outer loop</th>`;
  for (let i = 0; i < steps.length;) {
    let j = i + 1;
    while (j < steps.length && steps[j].outerIteration === steps[i].outerIteration) ++j;
    html += `<th class="outer-group" scope="colgroup" colspan="${j - i}">${config.hierarchy ? steps[i].outerIteration : "Direct range"}</th>`;
    i = j;
  }
  html += '</tr><tr><th scope="col">Thread / inner loop</th>';
  steps.forEach((step, i) => {
    const n = start + i;
    html += `<th scope="col" class="${n === position ? "cursor-column" : ""}"><button type="button" data-step="${n}" aria-label="Go to outer ${step.outerIteration}, inner ${step.innerIteration}" ${n === position ? 'aria-current="step"' : ""}>${step.innerIteration}</button></th>`;
  });
  html += "</tr></thead>";
  for (const [key, workersInBlock] of blockWorkers) {
    const blockColor = palette.get(workersInBlock[0].id).block;
    const collapsed = collapsedBlocks.has(key);
    html += `<tbody><tr class="timeline-block" style="--block-color:${blockColor}"><th colspan="${steps.length + 1}" scope="rowgroup"><button type="button" data-collapse-block="${key}" aria-expanded="${!collapsed}">${collapsed ? "▸" : "▾"} Block ${tuple(workersInBlock[0].block)} · ${workersInBlock.length} threads</button></th></tr>`;
    if (!collapsed) for (const worker of workersInBlock) {
      const color = palette.get(worker.id);
      html += `<tr data-timeline-worker="${worker.id}" class="${worker.id === selectedWorker ? "selected-row" : ""}"><th scope="row"><button type="button" data-select-worker="${worker.id}" style="--thread-color:${color.current}" aria-pressed="${worker.id === selectedWorker}">Thread ${tuple(worker.thread)} <small>T${worker.id}${worker.id === selectedWorker ? ' · <span class="selected-badge">Selected</span>' : ""}</small></button></th>`;
      steps.forEach((step, i) => {
        const n = start + i, a = timeline.byWorker.get(worker.id).get(n);
        const phase = n < position ? "past" : n === position ? "current" : "future";
        html += `<td class="${n === position ? "cursor-column" : ""}">`;
        if (a) html += `<button type="button" data-event-worker="${worker.id}" data-event-step="${n}" data-phase="${phase}" data-in-bounds="${a.inBounds}"
          class="timeline-event ${phase} ${!a.inBounds ? "guarded" : ""} ${a === selectedAssignment ? "selected-event" : ""} ${phase === "past" && !$("keep-history").checked ? "hide-history" : ""}"
          style="--event-current:${color.current};--event-past:${color.past};--event-ink:${color.ink}"
          aria-label="Worker T${worker.id}, index ${tuple(a.index)}, outer ${a.outerIteration}, inner ${a.innerIteration}${a.inBounds ? "" : ", rejected by guard"}"
          title="${tuple(a.index)} = ${tuple(a.origin)} + ${tuple(a.local)}${a.inBounds ? "" : " · rejected by guard"}">${tuple(a.index)}</button>`;
        else html += '<span class="idle-event" aria-label="No assignment">—</span>';
        html += "</td>";
      });
      html += "</tr>";
    }
    html += "</tbody>";
  }
  container.innerHTML = html + "</table>";
  container.scrollLeft = scrollLeft;
  container.scrollTop = scrollTop;
  // Keep the current column visible without scrolling the page vertically.
  const column = container.querySelector('[aria-current="step"]').closest("th").getBoundingClientRect();
  const viewport = container.getBoundingClientRect();
  const rowHeaderWidth = container.querySelector('th[scope="row"]')?.getBoundingClientRect().width ?? 150;
  if (column.left < viewport.left + rowHeaderWidth)
    container.scrollLeft -= viewport.left + rowHeaderWidth - column.left;
  else if (column.right > viewport.left + container.clientWidth)
    container.scrollLeft += column.right - viewport.left - container.clientWidth;
  if (selectionChanged) {
    const row = container.querySelector(`[data-timeline-worker="${selectedWorker}"]`)?.getBoundingClientRect();
    const headerBottom = viewport.top + container.querySelector("thead").getBoundingClientRect().height;
    if (row?.top < headerBottom) container.scrollTop -= headerBottom - row.top;
    else if (row?.bottom > viewport.top + container.clientHeight)
      container.scrollTop += row.bottom - viewport.top - container.clientHeight;
  }
}

function stopPlayback() {
  if (playback !== null) clearInterval(playback);
  playback = null;
  $("play").textContent = "Play";
  $("play").setAttribute("aria-pressed", "false");
}
function setStep(position) {
  const step = timeline?.steps[position];
  if (!step) return;
  outerIteration = step.outerIteration;
  innerIteration = step.innerIteration;
  selectedAssignment = null;
  preview = null;
  render();
}

function inspect() {
  const worker = currentWorker();
  const list = currentAssignments();
  const assignment = list.find(a => a === selectedAssignment)
    ?? list.find(a => a.outerIteration === outerIteration && a.innerIteration === innerIteration);
  selectedAssignment = assignment ?? null;
  const info = worker.inner;
  $("worker-selection").textContent = `Selected: T${worker.id} · Block ${tuple(worker.block)} · Thread ${tuple(worker.thread)}${timeline.byWorker.get(worker.id).has(cursor()) ? "" : " · Idle at this step"}`;
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
  drawComposition();
  drawTimeline();
}
function recompute() {
  stopPlayback();
  preview = null;
  delete $("data").dataset.selectionKey;
  if (!engine) return;
  try {
    const result = JSON.parse(engine.ccall("mapping_trace", "string", ["string"], [serialize(config)]));
    if (result.error) throw Error(result.error);
    if (result.build !== manifest.build) throw Error("Engine and manifest do not match. Rebuild the prototype.");
    trace = result;
    owners = ownersOf(trace);
    timeline = buildTimeline(trace);
    palette = workerPalette(trace);
    if (!currentWorker()) selectedWorker = 0;
    selectedAssignment = null;
    status(config.spec ? "FrameSpec input · worker layout explicitly assumed, NOT resolved by a backend."
      : "Live C++ / WebAssembly · real alpaka index iteration · synthetic worker contexts");
    render();
  } catch (error) {
    trace = null;
    timeline = null;
    for (const id of ["workers", "data", "composition-content", "composition-summary", "timeline", "calculation", "table", "diagnostics", "worker-selection"]) $(id).replaceChildren();
    updateTimelineControls();
    $("composition").hidden = true;
    $("code").textContent = "";
    status(error.message, true);
  }
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
    config[target.id] = Number(target.value);
    if (target.id === "api") config.executor = 0;
    if (target.id === "hierarchy") config.worker = config.hierarchy ? 4 : 0;
  }
  if (target.id !== "preset") $("preset").value = target.id === "dim" ? "tiles" : "custom";
  outerIteration = innerIteration = 0;
  controls();
  recompute();
}
$("config-form").addEventListener("change", changeConfiguration);
$("workspace-toolbar").addEventListener("change", changeConfiguration);
$("workers").addEventListener("click", event => {
  const button = event.target.closest("[data-worker]");
  if (!button || !trace) return;
  stopPlayback();
  preview = null;
  selectedWorker = Number(button.dataset.worker);
  selectedAssignment = null;
  render();
  $("workers").querySelector(`[data-worker="${selectedWorker}"]`)?.focus({preventScroll: true});
});
function selectCell(event) {
  const cell = event.target.closest("[data-cell]");
  if (!cell || !trace) return;
  const entries = owners.get(cell.dataset.cell);
  if (!entries?.length) return;
  const current = entries.findIndex(entry => entry.worker.id === selectedWorker && entry.assignment === selectedAssignment);
  const entry = entries[(current + 1) % entries.length];
  stopPlayback();
  preview = null;
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
$("show-composition").addEventListener("change", render);
$("keep-history").addEventListener("change", render);
$("timeline-scope").addEventListener("change", render);
$("expand-timeline").addEventListener("click", () => {
  const expanded = $("timeline-panel").classList.toggle("expanded");
  $("timeline-details").hidden = !expanded;
  $("expand-timeline").setAttribute("aria-expanded", String(expanded));
  $("expand-timeline").textContent = expanded ? "Collapse timeline" : "Expand timeline";
  delete $("timeline").dataset.selectedWorker;
  if (trace) drawTimeline();
});
$("step-back").addEventListener("click", () => { stopPlayback(); setStep(cursor() - 1); });
$("step-forward").addEventListener("click", () => { stopPlayback(); setStep(cursor() + 1); });
$("timeline-step").addEventListener("input", event => { stopPlayback(); setStep(Number(event.target.value)); });
$("play").addEventListener("click", () => {
  if (playback !== null) { stopPlayback(); return; }
  if (!trace || reducedMotion.matches || timeline.steps.length < 2) return;
  if (cursor() === timeline.steps.length - 1) setStep(0);
  $("play").textContent = "Pause";
  $("play").setAttribute("aria-pressed", "true");
  playback = setInterval(() => {
    setStep(cursor() + 1);
    if (cursor() === timeline.steps.length - 1) stopPlayback();
  }, 900);
});
reducedMotion.addEventListener("change", () => { stopPlayback(); updateTimelineControls(); });
document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlayback(); });
$("timeline").addEventListener("click", event => {
  if (!trace) return;
  const button = event.target.closest("button");
  if (!button) return;
  stopPlayback();
  preview = null;
  let selector;
  if (button.dataset.collapseBlock !== undefined) {
    const key = button.dataset.collapseBlock;
    if (collapsedBlocks.has(key)) collapsedBlocks.delete(key); else collapsedBlocks.add(key);
    selector = `[data-collapse-block="${key}"]`;
    render();
  } else if (button.dataset.selectWorker !== undefined) {
    selectedWorker = Number(button.dataset.selectWorker);
    selectedAssignment = null;
    selector = `[data-select-worker="${selectedWorker}"]`;
    render();
  } else {
    const step = Number(button.dataset.eventStep ?? button.dataset.step);
    if (button.dataset.eventWorker !== undefined) {
      selectedWorker = Number(button.dataset.eventWorker);
      selector = `[data-event-worker="${selectedWorker}"][data-event-step="${step}"]`;
    } else selector = `[data-step="${step}"]`;
    setStep(step);
  }
  $("timeline").querySelector(selector)?.focus({preventScroll: true});
});

function previewEntry(entry) {
  if (entry?.worker.id === selectedWorker && entry?.assignment === selectedAssignment) entry = null;
  if (preview?.worker === entry?.worker && preview?.assignment === entry?.assignment) return;
  preview = entry;
  if (trace) { drawData(); drawComposition(); }
}
$("workers").addEventListener("mouseover", event => {
  const button = event.target.closest("[data-worker]");
  if (!button || !trace || playback !== null || !canHover.matches) return;
  const worker = trace.workers.find(w => w.id === Number(button.dataset.worker));
  const assignment = timeline.byWorker.get(worker.id).get(cursor());
  previewEntry(assignment ? {worker, assignment} : null);
});
$("data").addEventListener("mouseover", event => {
  const cell = event.target.closest("[data-cell]");
  if (!cell || !trace || playback !== null || !canHover.matches) return;
  const entries = owners.get(cell.dataset.cell) ?? [];
  const atStep = entries.filter(e => timeline.positions.get(iterationKey(e.assignment)) === cursor());
  previewEntry(atStep.find(e => e.worker.id === selectedWorker) ?? atStep[0] ?? entries[0] ?? null);
});
for (const id of ["workers", "data"]) $(id).addEventListener("mouseleave", () => previewEntry(null));

$("labels").addEventListener("change", render);
$("all-iterations").addEventListener("change", render);
for (const name of ["outer", "inner"]) $(`${name}-iteration`).addEventListener("change", event => {
  stopPlayback();
  preview = null;
  const value = Math.max(0, Math.trunc(Number(event.target.value)) || 0);
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
  collapsedBlocks.clear();
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
initializeLayout(render);
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

// SPDX-License-Identifier: MPL-2.0
import {apis, innerRange, resolutionDescription, tuple, workers} from "./model.mjs";
import {assignmentsFor, selection} from "./exploration.mjs";

const $ = id => document.getElementById(id);

function calculation(config, model, state) {
  const {worker, assignment} = selection(model, state);
  const info = worker.inner;
  let html = `<p class="selection-label">Worker T${worker.id} · Block ${tuple(worker.block)} · Thread ${tuple(worker.thread)}</p>`;
  if (config.hierarchy) html += `<div class="calculation-step"><span class="level">1</span><div><h3>Block → outer offset</h3>
    <p>IdxRange [${tuple(config.outerBegin)}, ${tuple(config.outerEnd)}), stride ${tuple(config.outerStride)}</p>
    <p>Group index ${tuple(worker.outer.workerIndex)} of ${tuple(worker.outer.workerCount)}</p>
    <code>${worker.outer.traversal} traversal · ${worker.outer.layout} layout</code></div></div>`;
  const range = config.hierarchy ? innerRange(config)
    : {begin: config.directBegin, end: config.directEnd, stride: config.stride};
  html += `<div class="calculation-step"><span class="level">${config.hierarchy ? 2 : 1}</span><div><h3>${workers[config.worker]} → ${config.hierarchy ? "relative inner index" : "range index"}</h3>
    <p>IdxRange [${tuple(range.begin)}, ${tuple(range.end)}), stride ${tuple(range.stride)}</p>
    <p>Group index ${tuple(info.workerIndex)} of ${tuple(info.workerCount)}</p>
    <code>${info.traversal} traversal · ${info.layout} layout</code></div></div>`;
  if (assignment) {
    html += `<div class="formula">${config.hierarchy
      ? `global = outerIdx + innerIdx<br><strong>${tuple(assignment.index)} = ${tuple(assignment.origin)} + ${tuple(assignment.local)}</strong>`
      : `Returned index<br><strong>${tuple(assignment.index)}</strong>`}</div>
      <p>Outer iteration <strong>${assignment.outerIteration}</strong> · Inner iteration <strong>${assignment.innerIteration}</strong></p>
      <p>${assignment.inBounds ? `Linear offset (origin-zero buffer): <strong>${assignment.linear}</strong>` : "<strong>Outside the logical range — rejected by the bounds guard.</strong>"}</p>`;
  } else html += "<p>This worker has no assignments in the selected iteration.</p>";
  return html + '<p class="hint">Worker identities, effective policies, returned indices, composition, and linear offsets above come from C++. Intermediate iterator arithmetic is not yet traced.</p>';
}

function assignments(model, state) {
  const list = assignmentsFor(model, state);
  return `<p class="hint">Selected worker · ${list.length} assignments (including guarded indices)</p><div class="table-scroll"><table><thead><tr><th>Outer</th><th>Inner</th><th>Origin</th><th>Local</th><th>Global</th><th>Guard</th></tr></thead><tbody>${list.map(a =>
    `<tr><td>${a.outerIteration}</td><td>${a.innerIteration}</td><td>${tuple(a.origin)}</td><td>${tuple(a.local)}</td><td>${tuple(a.index)}</td><td>${a.inBounds ? "inside" : "outside"}</td></tr>`).join("")}</tbody></table></div>`;
}

function diagnostics(config, model) {
  const {expected, covered, repeated, guarded, idle} = model.diagnostics;
  return `<div class="metrics"><div><strong>${covered}/${expected}</strong><span>desired domain indices covered</span></div>
    <div><strong>${repeated}</strong><span>indices with repeated assignments</span></div>
    <div><strong>${guarded}</strong><span>assignments rejected by guard</span></div>
    <div><strong>${idle}</strong><span>idle workers</span></div></div>
    <p>${expected - covered} target indices have no assignment, including stride-skipped points and mapping gaps. Hatched cells are uncovered; white assigned cells have not yet been visited in playback.</p>
    <p>Repeated assignments can be intentional: multiple physical workers may share a worker-group identity. Coverage alone does not prove a kernel is race-free.</p>
    <p><strong>Context:</strong> ${apis[config.api]} API tags in synthetic worker contexts; no device kernel is executed.</p>
    ${config.spec ? `<p class="assumption">FrameSpec logical capacity ${tuple(model.trace.frameCapacity)}. ${resolutionDescription(config, model.trace)} Derived blocks ${tuple(model.trace.blocks)}, threads/block ${tuple(model.trace.threads)}.</p>` : ""}
    <p class="hint">No occupancy, timing, synchronization, memory-race, or hardware scheduling simulation. Warp and boundary overloads are not implemented.</p>`;
}

export function initializeInspector(getView) {
  let previousModel, active = "calculation";
  const rendered = new Map();
  function render() {
    const {config, model, state} = getView();
    if (model !== previousModel) {
      for (const id of ["calculation", "code", "table", "diagnostics"]) $(id).replaceChildren();
      previousModel = model;
      rendered.clear();
    }
    if (!model || !$("assignment-details").open) return;
    const key = active === "calculation" ? `${state.selectedWorker}:${state.position}`
      : active === "table" ? `${state.selectedWorker}:${state.allIterations ? "all" : state.position}` : "trace";
    if (rendered.get(active) === key) return;
    if (active === "code") $(active).textContent = model.code;
    else $(active).innerHTML = active === "calculation" ? calculation(config, model, state)
      : active === "table" ? assignments(model, state) : diagnostics(config, model);
    rendered.set(active, key);
  }
  $("assignment-details").addEventListener("toggle", render);
  for (const tab of document.querySelectorAll("[data-tab]")) tab.addEventListener("click", () => {
    active = tab.dataset.tab;
    for (const other of document.querySelectorAll("[data-tab]")) {
      const selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      $(other.dataset.tab).hidden = !selected;
    }
    render();
  });
  return render;
}

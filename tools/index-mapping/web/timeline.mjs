// SPDX-License-Identifier: MPL-2.0
import {tuple} from "./model.mjs";
import {timelineWindow} from "./visualization.mjs";

function focusSelector(button) {
  if (button.hasAttribute("data-event-worker"))
    return `[data-event-worker="${button.dataset.eventWorker}"][data-event-step="${button.dataset.eventStep}"]`;
  for (const attribute of ["data-step", "data-select-worker", "data-collapse-block"])
    if (button.hasAttribute(attribute)) return `[${attribute}="${button.getAttribute(attribute)}"]`;
  return null;
}

export function initializeTimeline(container) {
  let previousModel, structure, previousWorker;
  return function render(model, config, state) {
    const {timeline, palette, blocks} = model;
    const {position, selectedWorker, collapsedBlocks} = state;
    const {start, end} = timelineWindow(timeline.steps.length, position);
    const steps = timeline.steps.slice(start, end);
    if (!steps.length) { container.innerHTML = '<p class="hint">All workers are idle.</p>'; return; }
    const selectedBlock = model.workersById.get(selectedWorker)?.block.join(",") ?? null;
    const scopedBlock = state.timelineScope === "block" ? selectedBlock : null;
    const key = JSON.stringify([start, end, scopedBlock, [...collapsedBlocks]]);
    const selectionChanged = previousWorker !== selectedWorker || previousModel !== model;
    const scrollLeft = container.scrollLeft, scrollTop = container.scrollTop;
    if (previousModel !== model || structure !== key) {
      const active = document.activeElement;
      const focused = container.contains(active) && active.matches("button") ? focusSelector(active) : null;
      let html = `<table class="timeline-table"><caption>Steps ${start + 1}–${end} of ${timeline.steps.length}. Outer / inner loop iterations; blank cells indicate no assignment.</caption><thead><tr><th scope="col">Outer loop</th>`;
      for (let i = 0; i < steps.length;) {
        let j = i + 1;
        while (j < steps.length && steps[j].outerIteration === steps[i].outerIteration) ++j;
        html += `<th class="outer-group" scope="colgroup" colspan="${j - i}">${config.hierarchy ? steps[i].outerIteration : "Direct range"}</th>`;
        i = j;
      }
      html += '</tr><tr><th scope="col">Thread / inner loop</th>';
      steps.forEach((step, i) => {
        html += `<th scope="col"><button type="button" data-step="${start + i}" aria-label="Go to outer ${step.outerIteration}, inner ${step.innerIteration}">${step.innerIteration}</button></th>`;
      });
      html += "</tr></thead>";
      for (const [block, workers] of blocks) {
        if (scopedBlock !== null && block !== scopedBlock) continue;
        const collapsed = collapsedBlocks.has(block);
        html += `<tbody><tr class="timeline-block" style="--block-color:${palette.get(workers[0].id).block}"><th colspan="${steps.length + 1}" scope="rowgroup"><button type="button" data-collapse-block="${block}" aria-expanded="${!collapsed}">${collapsed ? "▸" : "▾"} Block ${tuple(workers[0].block)} · ${workers.length} threads</button></th></tr>`;
        if (!collapsed) for (const worker of workers) {
          const color = palette.get(worker.id);
          html += `<tr data-timeline-worker="${worker.id}"><th scope="row"><button type="button" data-select-worker="${worker.id}" style="--thread-color:${color.current}">Thread ${tuple(worker.thread)} <small></small></button></th>`;
          steps.forEach((step, i) => {
            const n = start + i, a = timeline.byWorker.get(worker.id).get(n);
            html += `<td data-column="${n}">`;
            if (a) html += `<button type="button" data-event-worker="${worker.id}" data-event-step="${n}" data-in-bounds="${a.inBounds}"
              class="timeline-event ${a.inBounds ? "" : "guarded"}"
              style="--event-current:${color.current};--event-past:${color.past};--event-ink:${color.ink}"
              aria-label="Worker T${worker.id}, index ${tuple(a.index)}, outer ${a.outerIteration}, inner ${a.innerIteration}${a.inBounds ? "" : ", skipped: rejected by guard; body not executed"}"
              title="${tuple(a.index)} = ${tuple(a.origin)} + ${tuple(a.local)}${a.inBounds ? "" : " · skipped: rejected by guard; body not executed"}"><span class="timeline-index">${tuple(a.index)}</span>${a.inBounds ? "" : '<span class="skipped-label" aria-hidden="true">⊘ Skipped</span>'}</button>`;
            else html += '<span class="idle-event" aria-label="No assignment">—</span>';
            html += "</td>";
          });
          html += "</tr>";
        }
        html += "</tbody>";
      }
      container.innerHTML = html + "</table>";
      structure = key;
      previousModel = model;
      if (focused) (container.querySelector(focused) ?? container).focus({preventScroll: true});
    }
    for (const button of container.querySelectorAll("[data-step]")) {
      const current = Number(button.dataset.step) === position;
      button.closest("th").classList.toggle("cursor-column", current);
      if (current) button.setAttribute("aria-current", "step"); else button.removeAttribute("aria-current");
    }
    for (const cell of container.querySelectorAll("[data-column]"))
      cell.classList.toggle("cursor-column", Number(cell.dataset.column) === position);
    for (const row of container.querySelectorAll("[data-timeline-worker]")) {
      const id = Number(row.dataset.timelineWorker), selected = id === selectedWorker;
      row.classList.toggle("selected-row", selected);
      const button = row.querySelector("[data-select-worker]");
      if (button.getAttribute("aria-pressed") !== String(selected)) {
        button.setAttribute("aria-pressed", String(selected));
        button.title = selected ? "Click again to deselect this thread" : "Select this thread";
        button.querySelector("small").innerHTML = `T${id}${selected ? ' · <span class="selected-badge">Selected</span>' : ""}`;
      }
    }
    for (const button of container.querySelectorAll("[data-event-worker]")) {
      const step = Number(button.dataset.eventStep);
      const phase = step < position ? "past" : step === position ? "current" : "future";
      button.dataset.phase = phase;
      for (const name of ["past", "current", "future"]) button.classList.toggle(name, name === phase);
      button.classList.toggle("selected-event", Number(button.dataset.eventWorker) === selectedWorker && step === position);
      button.classList.toggle("hide-history", phase === "past" && !state.keepHistory);
    }
    container.scrollLeft = scrollLeft;
    container.scrollTop = scrollTop;
    const column = container.querySelector('[aria-current="step"]').closest("th").getBoundingClientRect();
    const viewport = container.getBoundingClientRect();
    const rowHeaderWidth = container.querySelector('th[scope="row"]')?.getBoundingClientRect().width ?? 150;
    if (column.left < viewport.left + rowHeaderWidth)
      container.scrollLeft -= viewport.left + rowHeaderWidth - column.left;
    else if (column.right > viewport.left + container.clientWidth)
      container.scrollLeft += column.right - viewport.left - container.clientWidth;
    // Also reveal the row after expanding a previously hidden timeline.
    if (selectionChanged || container.dataset.revealSelection === "true") {
      const row = container.querySelector(`[data-timeline-worker="${selectedWorker}"]`)?.getBoundingClientRect();
      const headerBottom = viewport.top + container.querySelector("thead").getBoundingClientRect().height;
      if (row?.top < headerBottom) container.scrollTop -= headerBottom - row.top;
      else if (row?.bottom > viewport.top + container.clientHeight)
        container.scrollTop += row.bottom - viewport.top - container.clientHeight;
    }
    delete container.dataset.revealSelection;
    previousWorker = selectedWorker;
  };
}

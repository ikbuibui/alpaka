// SPDX-License-Identifier: MPL-2.0
// Trace-derived data and exploration transitions. Neither depends on the DOM.
import {eligible, ownersOf, snippet} from "./model.mjs";
import {buildTimeline, iterationKey, workerPalette} from "./visualization.mjs";

export function createTraceModel(trace, config) {
  const owners = ownersOf(trace), timeline = buildTimeline(trace);
  const workersById = new Map(trace.workers.map(worker => [worker.id, worker]));
  const blocks = new Map(), maxInner = new Map();
  for (const worker of trace.workers) {
    const key = worker.block.join(",");
    if (!blocks.has(key)) blocks.set(key, []);
    blocks.get(key).push(worker);
  }
  for (const step of timeline.steps)
    maxInner.set(step.outerIteration, Math.max(maxInner.get(step.outerIteration) ?? 0, step.innerIteration));
  let expected = 0;
  const rows = config.dim === 2 ? config.end[0] : 1, cols = config.end.at(-1);
  for (let row = 0; row < rows; ++row) for (let col = 0; col < cols; ++col)
    if (eligible(config, config.dim === 2 ? [row, col] : [col])) ++expected;
  return {trace, owners, timeline, workersById, blocks, maxInner, palette: workerPalette(trace),
    code: snippet(config, trace), diagnostics: {
      expected, covered: owners.size,
      repeated: [...owners.values()].filter(entries => entries.length > 1).length,
      guarded: trace.workers.reduce((sum, worker) => sum + worker.assignments.filter(a => !a.inBounds).length, 0),
      idle: trace.workers.filter(worker => !worker.assignments.length).length,
    }};
}

export const createExploration = () => ({selectedWorker: 0, position: 0, preview: null,
  allIterations: true, keepHistory: true, showComposition: true, labels: "index", timelineScope: "all",
  collapsedBlocks: new Set()});

export function selection(model, state) {
  const worker = model.workersById.get(state.selectedWorker);
  const step = model.timeline.steps[state.position];
  const assignment = model.timeline.byWorker.get(state.selectedWorker)?.get(state.position) ?? null;
  return {worker, assignment, outerIteration: step?.outerIteration ?? 0, innerIteration: step?.innerIteration ?? 0};
}

export function assignmentsFor(model, state) {
  const {worker, assignment} = selection(model, state);
  return state.allIterations ? worker?.assignments ?? [] : assignment ? [assignment] : [];
}

export function selectWorker(model, state, id) {
  if (id !== null && !model.workersById.has(id)) return;
  state.selectedWorker = id;
  state.preview = null;
}

export function toggleWorker(model, state, id) {
  selectWorker(model, state, state.selectedWorker === id ? null : id);
}

export function setStep(model, state, position) {
  state.position = Math.max(0, Math.min(position, Math.max(0, model.timeline.steps.length - 1)));
  state.preview = null;
}

export function selectAssignment(model, state, entry) {
  selectWorker(model, state, entry.worker.id);
  setStep(model, state, model.timeline.positions.get(iterationKey(entry.assignment)));
}

export function setIteration(model, state, name, value) {
  let {outerIteration, innerIteration} = selection(model, state);
  if (name === "outer") outerIteration = Math.min(value, model.timeline.steps.at(-1)?.outerIteration ?? 0);
  else innerIteration = value;
  innerIteration = Math.min(innerIteration, model.maxInner.get(outerIteration) ?? 0);
  setStep(model, state, model.timeline.positions.get(`${outerIteration}:${innerIteration}`) ?? 0);
}

export function resetTraceSelection(model, state) {
  if (state.selectedWorker !== null && !model.workersById.has(state.selectedWorker))
    state.selectedWorker = model.trace.workers[0].id;
  setStep(model, state, 0);
}

// SPDX-License-Identifier: MPL-2.0
import assert from "node:assert/strict";
import {test} from "node:test";
import {defaults} from "./web/model.mjs";
import {assignmentsFor, createExploration, createTraceModel, resetTraceSelection, selectAssignment,
  selection, selectWorker, setIteration, setStep, toggleWorker} from "./web/exploration.mjs";

const assignment = (outer, inner, index, inBounds = true) => ({outerIteration: outer,
  innerIteration: inner, index: [index], origin: [outer * 8], local: [index - outer * 8], inBounds});
const worker = (id, block, assignments) => ({id, block: [block], thread: [0], assignments,
  inner: {traversal: "tiled"}, outer: {traversal: "tiled"}});
function fixture() {
  const trace = {workers: [
    worker(0, 0, [assignment(0, 0, 0), assignment(0, 1, 4), assignment(1, 0, 8)]),
    worker(7, 1, [assignment(0, 0, 0), assignment(0, 1, 16, false)]),
    worker(9, 2, []),
  ]};
  return {model: createTraceModel(trace, defaults(1)), state: createExploration()};
}

test("selection derives from one shared cursor and worker ID, including idle steps", () => {
  const {model, state} = fixture();
  setStep(model, state, 1);
  state.preview = {worker: model.trace.workers[0], assignment: model.trace.workers[0].assignments[0]};
  selectWorker(model, state, 7);
  assert.equal(state.position, 1);
  assert.equal(state.preview, null);
  assert.equal(selection(model, state).assignment.inBounds, false);
  setStep(model, state, 2);
  assert.equal(selection(model, state).assignment, null);
  assert.equal(selection(model, state).worker.id, 7);
  selectWorker(model, state, 9);
  assert.equal(selection(model, state).assignment, null);
  assert.equal(state.position, 2);
});

test("worker toggles clear selection and previews without changing the cursor or trace", () => {
  const {model, state} = fixture();
  const before = structuredClone(model);
  setStep(model, state, 1);
  state.preview = model.owners.get("4")[0];
  toggleWorker(model, state, 0);
  assert.equal(state.selectedWorker, null);
  assert.equal(state.preview, null);
  assert.equal(state.position, 1);
  assert.equal(selection(model, state).worker, undefined);
  assert.equal(selection(model, state).assignment, null);
  assert.deepEqual(assignmentsFor(model, state), []);
  state.allIterations = false;
  assert.deepEqual(assignmentsFor(model, state), []);
  setStep(model, state, 2);
  assert.equal(state.selectedWorker, null);
  resetTraceSelection(model, state);
  assert.equal(state.selectedWorker, null);
  assert.deepEqual(model, before);
  toggleWorker(model, state, 7);
  assert.equal(state.selectedWorker, 7);
  toggleWorker(model, state, 0);
  assert.equal(state.selectedWorker, 0);
  const entry = model.owners.get("4")[0];
  selectAssignment(model, state, entry);
  selectAssignment(model, state, entry);
  assert.equal(state.selectedWorker, 0); // Event selection is not a toggle.
  assert.equal(selection(model, state).assignment, entry.assignment);
});

test("assignment selection and iteration edits use the same cursor, clamping shorter inner loops", () => {
  const {model, state} = fixture();
  const entry = model.owners.get("4")[0];
  selectAssignment(model, state, entry);
  assert.equal(selection(model, state).assignment, entry.assignment);
  setIteration(model, state, "outer", 1);
  assert.equal(state.position, 2);
  assert.equal(selection(model, state).innerIteration, 0);
  setIteration(model, state, "inner", 100);
  assert.equal(state.position, 2);
  setStep(model, state, -1);
  assert.equal(state.position, 0);
  setStep(model, state, 100);
  assert.equal(state.position, 2);
});

test("outline scope never changes selection, and trace-wide data remains unchanged", () => {
  const {model, state} = fixture();
  const before = structuredClone(model);
  assert.equal(assignmentsFor(model, state).length, 3);
  state.allIterations = false;
  setStep(model, state, 1);
  assert.deepEqual(assignmentsFor(model, state), [model.trace.workers[0].assignments[1]]);
  assert.equal(selection(model, state).assignment.index[0], 4);
  assert.deepEqual(model, before);
  assert.deepEqual(model.diagnostics, {expected: 16, covered: 3, repeated: 1, guarded: 1, idle: 1});
});

test("trace replacement preserves an existing worker but clears stale previews and positions", () => {
  const {model, state} = fixture();
  selectWorker(model, state, 7);
  setStep(model, state, 2);
  state.preview = model.owners.get("4")[0];
  resetTraceSelection(model, state);
  assert.equal(state.selectedWorker, 7);
  assert.equal(state.position, 0);
  assert.equal(state.preview, null);
  const replacement = createTraceModel({workers: [worker(12, 0, [])]}, defaults(1));
  resetTraceSelection(replacement, state);
  assert.equal(state.selectedWorker, 12);
  setIteration(replacement, state, "outer", 100);
  assert.equal(state.position, 0);
  assert.equal(selection(replacement, state).assignment, null);
});

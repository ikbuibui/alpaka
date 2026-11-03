// SPDX-License-Identifier: MPL-2.0
import assert from "node:assert/strict";
import {test} from "node:test";
import {assignmentPhase, buildTimeline, timelineWindow, visibleVisits, workerPalette} from "./web/visualization.mjs";

const assignment = (outer, inner, index, inBounds = true) => ({outerIteration: outer,
  innerIteration: inner, index: [index], origin: [0], local: [index], inBounds});
const worker = (id, block, assignments) => ({id, block: [block], thread: [id % 4], assignments});
const fixture = () => ({workers: [
  worker(0, 0, [assignment(0, 0, 0), assignment(0, 1, 4), assignment(1, 0, 16), assignment(1, 1, 20)]),
  worker(1, 0, [assignment(0, 0, 1), assignment(1, 0, 17)]),
  worker(4, 1, [assignment(0, 0, 8), assignment(0, 1, 12), assignment(1, 0, 24), assignment(1, 1, 28, false)]),
  worker(5, 1, []),
]});

test("timeline aligns nested loop pairs without packing exhausted workers into gaps", () => {
  const trace = fixture(), timeline = buildTimeline(trace);
  assert.deepEqual(timeline.steps.map(s => s.key), ["0:0", "0:1", "1:0", "1:1"]);
  assert.deepEqual(timeline.steps[1].events.map(e => e.assignment.index), [[4], [12]]);
  assert.equal(timeline.byWorker.get(1).get(1), undefined);
  assert.equal(timeline.byWorker.get(1).get(2), trace.workers[1].assignments[1]);
  assert.equal(timeline.byWorker.get(5).size, 0);
  assert.equal(timeline.steps[3].events[1].assignment.inBounds, false);
});

test("alignment is numeric, independent of worker enumeration order", () => {
  const trace = {workers: [worker(0, 0, [assignment(10, 0, 10), assignment(2, 1, 2)]),
    worker(1, 0, [assignment(0, 0, 0)])]};
  assert.deepEqual(buildTimeline(trace).steps.map(s => s.key), ["0:0", "2:1", "10:0"]);
});

test("history is derived from the cursor and reverses correctly when scrubbing", () => {
  const trace = fixture(), timeline = buildTimeline(trace);
  const entries = trace.workers.flatMap(worker => worker.assignments.map(assignment => ({worker, assignment})));
  assert.equal(visibleVisits(entries, timeline, 1, true).length, 5);
  assert.equal(visibleVisits(entries, timeline, 1, false).length, 2);
  assert.equal(visibleVisits(entries, timeline, 0, true).length, 3);
  assert.equal(visibleVisits(entries, timeline, 3, true).length, 9); // Guard never colors data.
  assert.equal(assignmentPhase(trace.workers[0].assignments[2], timeline, 1), "future");
  assert.equal(assignmentPhase(trace.workers[0].assignments[1], timeline, 1), "current");
  assert.equal(assignmentPhase(trace.workers[0].assignments[0], timeline, 1), "past");
});

test("repeated ownership is preserved, including previous visits to the same index", () => {
  const trace = {workers: [worker(0, 0, [assignment(0, 0, 0), assignment(0, 1, 0)]),
    worker(1, 0, [assignment(0, 1, 0)])]};
  const timeline = buildTimeline(trace);
  const entries = trace.workers.flatMap(worker => worker.assignments.map(assignment => ({worker, assignment})));
  assert.equal(visibleVisits(entries, timeline, 1, true).length, 3);
  assert.equal(visibleVisits(entries, timeline, 1, false).length, 2);
  assert.equal(timeline.steps[1].events.length, 2);
});

test("palette shares block colors, differentiates threads, and has fixed history colors", () => {
  const trace = fixture(), palette = workerPalette(trace);
  assert.equal(palette.get(0).block, palette.get(1).block);
  assert.notEqual(palette.get(0).block, palette.get(4).block);
  assert.notEqual(palette.get(0).current, palette.get(1).current);
  assert.notEqual(palette.get(0).current, palette.get(0).past);
  assert.deepEqual(workerPalette(trace), palette);
  assert(["#000", "#fff"].includes(palette.get(0).ink));
});

test("categorical block families stay fixed for ten blocks and deliberately cycle", () => {
  const trace = {workers: Array.from({length: 25}, (_, i) => worker(i, i, []))};
  const palette = workerPalette(trace);
  const colors = [...palette.values()].map(p => p.block);
  assert.deepEqual(colors.slice(0, 10), ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]);
  assert.equal(new Set(colors.slice(0, 10)).size, 10);
  for (let i = 10; i < colors.length; ++i) assert.deepEqual(palette.get(i), palette.get(i % 10));
  assert.deepEqual([...workerPalette({workers: trace.workers.slice(0, 6)}).values()],
    [...palette.values()].slice(0, 6)); // Adding blocks never recolors existing families.
});

test("all ten families retain the thread-lightness ramp and contrasting label ink", () => {
  const trace = {workers: Array.from({length: 40}, (_, id) => worker(id, Math.floor(id / 4), []))};
  const palette = workerPalette(trace);
  for (const [id, color] of palette) {
    const [, h, s, l] = color.current.match(/^hsl\((\S+) (\S+)% (\S+)%\)$/).map(Number);
    assert.equal(l, 42 + (id % 4) * 7);
    assert(color.past.endsWith(` ${79 + (id % 4) * 3}%)`));
    // Independently check the selected black/white ink against WCAG relative luminance.
    const a = s / 100 * Math.min(l / 100, 1 - l / 100);
    const rgb = [0, 8, 4].map(n => {
      const k = (n + h / 30) % 12;
      const v = l / 100 - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
      return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
    });
    const luminance = rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722;
    const contrast = color.ink === "#000" ? (luminance + .05) / .05 : 1.05 / (luminance + .05);
    assert(contrast >= 4.5, `${id}: ${color.current} with ${color.ink}`);
  }
});

test("windowing bounds the DOM while keeping every cursor accessible", () => {
  assert.deepEqual(timelineWindow(0, 0), {start: 0, end: 0});
  assert.deepEqual(timelineWindow(4, 2), {start: 0, end: 4});
  for (const cursor of [0, 1, 23, 16000, 32767]) {
    const {start, end} = timelineWindow(32768, cursor);
    assert.equal(end - start, 24);
    assert(start <= cursor && cursor < end);
  }
});

test("empty ranges still retain idle worker rows in the model", () => {
  const timeline = buildTimeline({workers: [worker(0, 0, [])]});
  assert.equal(timeline.steps.length, 0);
  assert.equal(timeline.positions.size, 0);
  assert.equal(timeline.byWorker.get(0).size, 0);
});

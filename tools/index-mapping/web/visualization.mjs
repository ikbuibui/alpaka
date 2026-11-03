// SPDX-License-Identifier: MPL-2.0
// Presentation only: align C++ trace records, never reconstruct index arithmetic.
export const iterationKey = a => `${a.outerIteration}:${a.innerIteration}`;

export function buildTimeline(trace) {
  const groups = new Map();
  for (const worker of trace.workers) for (const assignment of worker.assignments) {
    const key = iterationKey(assignment);
    if (!groups.has(key)) groups.set(key, {key, outerIteration: assignment.outerIteration,
      innerIteration: assignment.innerIteration, events: []});
    groups.get(key).events.push({worker, assignment});
  }
  const steps = [...groups.values()].sort((a, b) => a.outerIteration - b.outerIteration
    || a.innerIteration - b.innerIteration);
  const positions = new Map(steps.map((step, i) => [step.key, i]));
  const byWorker = new Map(trace.workers.map(worker => [worker.id, new Map()]));
  for (const [i, step] of steps.entries()) for (const {worker, assignment} of step.events)
    byWorker.get(worker.id).set(i, assignment);
  return {steps, positions, byWorker};
}

export function assignmentPhase(assignment, timeline, cursor) {
  const position = timeline.positions.get(iterationKey(assignment));
  return position < cursor ? "past" : position === cursor ? "current" : "future";
}

// Every visit remains inspectable; simultaneous/repeated ownership is never overwritten.
export function visibleVisits(entries, timeline, cursor, keepHistory) {
  return entries.filter(({assignment}) => assignment.inBounds &&
    (assignmentPhase(assignment, timeline, cursor) === "current"
      || (keepHistory && assignmentPhase(assignment, timeline, cursor) === "past")))
    .sort((a, b) => timeline.positions.get(iterationKey(a.assignment)) - timeline.positions.get(iterationKey(b.assignment)));
}

function contrastInk(hue, saturation, lightness) {
  // Choose the higher-contrast text color using sRGB relative luminance.
  const l = lightness / 100, a = saturation / 100 * Math.min(l, 1 - l);
  const rgb = [0, 8, 4].map(n => {
    const k = (n + hue / 30) % 12;
    const v = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
  });
  const luminance = rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722;
  return luminance > .179 ? "#000" : "#fff";
}

export function workerPalette(trace) {
  const blocks = new Map();
  const palette = new Map();
  for (const worker of trace.workers) {
    const key = worker.block.join(",");
    if (!blocks.has(key)) blocks.set(key, []);
    blocks.get(key).push(worker);
  }
  const hues = [220, 30, 160, 285, 65, 190];
  let b = 0;
  for (const workers of blocks.values()) {
    const hue = hues[b] ?? (b * 137.508) % 360;
    workers.forEach((worker, t) => {
      const lightness = 42 + (t % 4) * 7;
      const threadHue = hue + Math.floor(t / 4) % 3 * 5;
      palette.set(worker.id, {
        block: `hsl(${hue} 54% 43%)`,
        current: `hsl(${threadHue} 65% ${lightness}%)`,
        ink: contrastInk(threadHue, 65, lightness),
        past: `hsl(${threadHue} 24% ${79 + (t % 4) * 3}%)`,
      });
    });
    ++b;
  }
  return palette;
}

export function timelineWindow(length, cursor, size = 24) {
  const start = Math.max(0, Math.min(cursor - Math.floor(size / 2), length - size));
  return {start, end: Math.min(length, start + size)};
}

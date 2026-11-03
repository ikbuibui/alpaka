// SPDX-License-Identifier: MPL-2.0
// Configuration and presentation helpers only. Index mapping lives in engine.cpp.
export const workers = ["threadsInGrid", "linearThreadsInGrid", "blocksInGrid", "linearBlocksInGrid",
  "threadsInBlock", "linearThreadsInBlock", "allThreads"];
export const traversals = ["default", "flat", "tiled"];
export const layouts = ["optimized", "strided", "contiguous"];
export const apis = ["host", "cuda", "hip", "oneApi"];
export const executors = ["anyExecutor", "cpuSerial", "cpuOmpBlocks", "cpuTbbBlocks", "gpuCuda", "gpuHip", "oneApi"];
export const fields = ["blocks", "threads", "frames", "tile", "begin", "end", "stride"];
export const scalars = ["dim", "spec", "api", "executor", "hierarchy", "worker", "traversal", "layout",
  "outerTraversal", "outerLayout", "outerWorker"];

export function defaults(dim = 2) {
  return {dim, spec: 0, api: 1, executor: 0, hierarchy: 1, worker: 4,
    traversal: 0, layout: 0, outerTraversal: 0, outerLayout: 0, outerWorker: 0,
    blocks: dim === 2 ? [2, 2] : [2], threads: dim === 2 ? [1, 4] : [4],
    frames: dim === 2 ? [2, 2] : [2], tile: dim === 2 ? [2, 4] : [8],
    begin: dim === 2 ? [0, 0] : [0], end: dim === 2 ? [4, 8] : [16],
    stride: dim === 2 ? [1, 1] : [1]};
}

export function serialize(config) {
  const values = [1, ...scalars.map(key => config[key])];
  for (const key of fields) {
    if (!Array.isArray(config[key]) || config[key].length !== config.dim) throw Error(`Invalid ${key} dimensions`);
    values.push(...config[key]);
  }
  if (!values.every(Number.isSafeInteger)) throw Error("All parameters must be integers");
  return values.join(" ");
}

export function fromHash(hash) {
  if (!hash) return defaults();
  if (hash.length > 8192) throw Error("Configuration link is too large");
  const result = JSON.parse(decodeURIComponent(hash.replace(/^#/, "")));
  if (!result || ![1, 2].includes(result.dim)) throw Error("Unsupported dimensionality");
  serialize(result);
  return Object.fromEntries([...scalars, ...fields].map(key => [key, result[key]]));
}

export const tuple = value => `(${value.join(", ")})`;
export const vec = value => `Vec{${value.map(v => `${v}u`).join(", ")}}`;

export function ownersOf(trace) {
  const owners = new Map();
  for (const worker of trace.workers) for (const assignment of worker.assignments) {
    if (!assignment.inBounds) continue;
    const key = assignment.index.join(",");
    if (!owners.has(key)) owners.set(key, []);
    owners.get(key).push({worker, assignment});
  }
  return owners;
}

export function eligible(config, index) {
  return index.every((value, d) => value >= config.begin[d] && value < config.end[d]
    && (config.hierarchy || (value - config.begin[d]) % config.stride[d] === 0));
}

export function snippet(c, trace) {
  const range = `IdxRange{${vec(c.begin)}, ${vec(c.end)}, ${vec(c.stride)}}`;
  const tileRange = `IdxRange{${vec(c.begin)}, ${vec(c.end)}, tileExtent}`;
  const policyArgs = (traversal, layout, effective) => {
    if (!traversal && !layout) return "";
    const traverse = traversal ? traversals[traversal] : effective;
    return `, onAcc::traverse::${traverse}, onAcc::layout::${layouts[layout]}`;
  };
  const call = (group, range, traversal, layout, effective) =>
    `onAcc::makeIdxMap(acc, onAcc::worker::${group},\n        ${range}${policyArgs(traversal, layout, effective)})`;
  const spec = c.spec
    ? `onHost::FrameSpec{${vec(c.frames)}, ${vec(c.tile)}, exec::${executors[c.executor]}}`
    : `onHost::ThreadSpec{${vec(c.blocks)}, ${vec(c.threads)}, exec::${executors[c.executor]}}`;
  const header = `// Snippets, not a complete runnable program.\n// using namespace alpaka; API: api::${apis[c.api]}\nauto launchSpec = ${spec};\n`
    + (c.spec ? `// Browser trace assumes blocks=${tuple(c.blocks)}, threads=${tuple(c.threads)}.\n// Actual backend FrameSpec resolution is NOT modeled.\n` : "")
    + "\n// Inside the kernel, with acc supplied by alpaka:\n";
  if (!c.hierarchy) return header + `for (auto idx : ${call(workers[c.worker], range, c.traversal, c.layout, trace.workers[0].inner.traversal)})\n{\n    // Process idx.\n}`;
  const guard = c.end.map((value, d) => `globalIdx[${d}u] < ${value}u`).join(" && ");
  return header + `auto tileExtent = ${vec(c.tile)};\nfor (auto tileStart : ${call(c.outerWorker ? "linearBlocksInGrid" : "blocksInGrid", tileRange, c.outerTraversal, c.outerLayout, trace.workers[0].outer.traversal)})\n{\n    for (auto localIdx : ${call(workers[c.worker], "IdxRange{tileExtent}", c.traversal, c.layout, trace.workers[0].inner.traversal)})\n    {\n        auto globalIdx = tileStart + localIdx;\n        if (${guard})\n        {\n            // Process globalIdx.\n        }\n    }\n}`;
}

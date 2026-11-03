// SPDX-License-Identifier: MPL-2.0
// Configuration and presentation helpers only. Index mapping lives in engine.cpp.
export const workers = ["threadsInGrid", "linearThreadsInGrid", "blocksInGrid", "linearBlocksInGrid",
  "threadsInBlock", "linearThreadsInBlock", "allThreads"];
export const traversals = ["default", "flat", "tiled"];
export const layouts = ["optimized", "strided", "contiguous"];
export const apis = ["host", "cuda", "hip", "oneApi"];
export const executors = ["anyExecutor", "cpuSerial", "cpuOmpBlocks", "cpuTbbBlocks", "gpuCuda", "gpuHip", "oneApi"];
export const fields = ["blocks", "threads", "frames", "tile", "begin", "end", "stride", "frameExtent",
  "outerBegin", "outerEnd", "outerStride", "innerBegin", "innerEnd", "innerStride", "directBegin", "directEnd"];
export const scalars = ["dim", "spec", "api", "executor", "hierarchy", "worker", "traversal", "layout",
  "outerTraversal", "outerLayout", "outerWorker", "maxThreadsPerBlock", "frameType", "innerRangeMode",
  "outerDomainBounds", "directDomainBounds", "tileOuterStride"];

export function defaults(dim = 2) {
  return {dim, spec: 0, api: 1, executor: 0, hierarchy: 1, worker: 4,
    traversal: 0, layout: 0, outerTraversal: 0, outerLayout: 0, outerWorker: 0,
    blocks: dim === 2 ? [2, 2] : [2], threads: dim === 2 ? [1, 4] : [4],
    frames: dim === 2 ? [2, 2] : [2], frameExtent: dim === 2 ? [2, 4] : [8],
    tile: dim === 2 ? [2, 4] : [8], maxThreadsPerBlock: 1024, frameType: 0, innerRangeMode: 0,
    outerBegin: Array(dim).fill(0), outerEnd: dim === 2 ? [4, 8] : [16],
    outerStride: dim === 2 ? [2, 4] : [8], outerDomainBounds: 0, directDomainBounds: 0, tileOuterStride: 1,
    directBegin: Array(dim).fill(0), directEnd: dim === 2 ? [4, 8] : [16],
    innerBegin: Array(dim).fill(0), innerEnd: dim === 2 ? [2, 4] : [8], innerStride: Array(dim).fill(1),
    begin: dim === 2 ? [0, 0] : [0], end: dim === 2 ? [4, 8] : [16],
    stride: dim === 2 ? [1, 1] : [1]};
}

export function linkedRanges(config) {
  if (!config.hierarchy) return config.directDomainBounds
    ? {...config, directBegin: [...config.begin], directEnd: [...config.end]} : config;
  return {...config,
    ...(config.outerDomainBounds ? {outerBegin: [...config.begin], outerEnd: [...config.end]} : {}),
    ...(config.tileOuterStride ? {tile: [...config.outerStride]} : {})};
}

export function serialize(config) {
  config = linkedRanges(config);
  const values = [6, ...scalars.map(key => config[key])];
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
  const frameExtent = result.frameExtent ?? result.tile;
  const tile = !result.outerBegin && result.spec && (result.coupleTile ?? 1) ? frameExtent : result.tile;
  const migrated = {...result, frameExtent: [...frameExtent], tile: [...tile],
    maxThreadsPerBlock: result.maxThreadsPerBlock ?? 1024, frameType: result.frameType ?? 0,
    innerRangeMode: result.innerRangeMode ?? 0,
    outerDomainBounds: result.outerDomainBounds ?? 0, directDomainBounds: result.directDomainBounds ?? 0,
    tileOuterStride: result.tileOuterStride ?? 0,
    directBegin: result.directBegin ?? [...result.begin], directEnd: result.directEnd ?? [...result.end],
    outerBegin: result.outerBegin ?? [...result.begin], outerEnd: result.outerEnd ?? [...result.end],
    outerStride: result.hierarchy && result.outerTileStride ? [...tile] : result.outerStride ?? [...tile],
    innerBegin: result.innerBegin ?? Array(result.dim).fill(0),
    innerEnd: result.innerEnd ?? [...tile], innerStride: result.innerStride ?? Array(result.dim).fill(1)};
  const linked = linkedRanges(migrated);
  serialize(linked);
  return Object.fromEntries([...scalars, ...fields].map(key => [key, linked[key]]));
}

export const tuple = value => `(${value.join(", ")})`;
export const vec = value => `Vec{${value.map(v => `${v}u`).join(", ")}}`;
export const cvec = value => `CVec<uint32_t, ${value.map(v => `${v}u`).join(", ")}>{}`;

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
  return index.every((value, d) => value >= config.begin[d] && value < config.end[d]);
}

export function resolutionDescription(c, trace) {
  if (!c.spec || !trace?.blocks) return "";
  const executor = executors[trace.effectiveExecutor];
  const selection = c.executor ? executor : `anyExecutor → ${executor} (profile default)`;
  const type = c.frameType ? "Compile-time CVec frame extents" : "Runtime Vec frame extents";
  const rule = trace.effectiveExecutor === 1 ? "cpuSerial collapses blocks and threads to one per dimension."
    : trace.effectiveExecutor <= 3 ? "Blocks = frame count; one thread/block in every dimension."
    : c.frameType
      ? `Compile-time thread limit = ${trace.threadLimit}; halve dimensions in alpaka's fixed back-and-forth order. Runtime device limit is not used.${c.api === 3 ? " oneAPI bound is a profile assumption matching the current headers." : ""}`
      : `Assumed maxThreadsPerBlock = ${trace.maxThreadsPerBlock}; repeatedly halve the largest frame-extent component until it fits.`;
  return `Resolved for this device profile: ${selection}. ${type}. ${rule} No hardware queried; native defaults depend on the device/build. No kernel resource, occupancy, or scheduling simulation.`;
}

export function innerRange(c) {
  return c.innerRangeMode ? {begin: c.innerBegin, end: c.innerEnd, stride: c.innerStride}
    : {begin: Array(c.dim).fill(0), end: c.tile, stride: Array(c.dim).fill(1)};
}

export function snippet(c, trace) {
  c = linkedRanges(c);
  const range = `IdxRange{${vec(c.directBegin)}, ${vec(c.directEnd)}, ${vec(c.stride)}}`;
  const outerRange = `IdxRange{${vec(c.outerBegin)}, ${vec(c.outerEnd)}, ${vec(c.outerStride)}}`;
  const localRange = c.innerRangeMode
    ? `IdxRange{${vec(c.innerBegin)}, ${vec(c.innerEnd)}, ${vec(c.innerStride)}}` : "IdxRange{tileExtent}";
  const policyArgs = (traversal, layout, effective) => {
    if (!traversal && !layout) return "";
    const traverse = traversal ? traversals[traversal] : effective;
    return `, onAcc::traverse::${traverse}, onAcc::layout::${layouts[layout]}`;
  };
  const call = (group, range, traversal, layout, effective) =>
    `onAcc::makeIdxMap(acc, onAcc::worker::${group},\n        ${range}${policyArgs(traversal, layout, effective)})`;
  const spec = c.spec
    ? `onHost::FrameSpec{${vec(c.frames)}, ${(c.frameType ? cvec : vec)(c.frameExtent)}, exec::${executors[c.executor]}}`
    : `onHost::ThreadSpec{${vec(c.blocks)}, ${vec(c.threads)}, exec::${executors[c.executor]}}`;
  const header = `// Snippets, not a complete runnable program.\n// using namespace alpaka; API: api::${apis[c.api]}\nauto launchSpec = ${spec};\n`
    + (c.spec ? `// ${resolutionDescription(c, trace)}\n// Derived ThreadSpec: blocks=${tuple(trace.blocks)}, threads=${tuple(trace.threads)}.\n` : "")
    + "\n// Inside the kernel, with acc supplied by alpaka:\n";
  if (!c.hierarchy) {
    const directGuard = c.end.map((value, d) => `${c.begin[d]}u <= idx[${d}u] && idx[${d}u] < ${value}u`).join(" && ");
    return header + `for (auto idx : ${call(workers[c.worker], range, c.traversal, c.layout, trace.workers[0].inner.traversal)})\n{\n    if (${directGuard})\n    {\n        // Process idx.\n    }\n}`;
  }
  const guard = c.end.map((value, d) => `${c.begin[d]}u <= globalIdx[${d}u] && globalIdx[${d}u] < ${value}u`).join(" && ");
  return header + (c.innerRangeMode ? "" : `auto tileExtent = ${vec(c.tile)};\n`)
    + `for (auto outerIdx : ${call(c.outerWorker ? "linearBlocksInGrid" : "blocksInGrid", outerRange, c.outerTraversal, c.outerLayout, trace.workers[0].outer.traversal)})\n{\n    for (auto innerIdx : ${call(workers[c.worker], localRange, c.traversal, c.layout, trace.workers[0].inner.traversal)})\n    {\n        auto globalIdx = outerIdx + innerIdx;\n        if (${guard})\n        {\n            // Process globalIdx.\n        }\n    }\n}`;
}

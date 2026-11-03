// SPDX-License-Identifier: MPL-2.0
// Native/WASM conformance, plus independent coverage, ownership and sequence assertions.
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {existsSync, readFileSync, readdirSync} from "node:fs";
import {spawnSync} from "node:child_process";
import {fileURLToPath} from "node:url";
import {defaults, eligible, ownersOf, serialize} from "./web/model.mjs";

const here = fileURLToPath(new URL(".", import.meta.url));
const root = fileURLToPath(new URL("../../", import.meta.url));
const build = `${here}build/`;
const manifest = JSON.parse(readFileSync(`${build}manifest.json`));
const digest = createHash("sha256");
function headers(dir) {
  return readdirSync(dir, {withFileTypes: true}).flatMap(entry => entry.isDirectory()
    ? headers(`${dir}/${entry.name}`) : entry.name.endsWith(".hpp") ? [`${dir}/${entry.name}`] : []);
}
for (const path of [...headers(`${root}include`).sort(), `${here}engine.cpp`, `${here}build.py`, `${here}validate_backend.cpp`]) {
  digest.update(path.slice(root.length));
  digest.update(readFileSync(path));
}
assert.equal(digest.digest("hex"), manifest.sourceHash, "Stale engine: rebuild after changing alpaka headers or adapter sources");
for (const file of readdirSync(`${here}web`)) {
  assert(readFileSync(`${build}site/${file}`).equals(readFileSync(`${here}web/${file}`)), `Stale site asset: ${file}; rebuild`);
}
let wasm;
if (manifest.wasm) {
  const {default: createEngine} = await import("./build/site/engine.mjs");
  wasm = await createEngine();
} else console.warn("Native-only run: WASM conformance is NOT being tested.");

const cases = [];
for (const dim of [1, 2]) for (const api of [0, 1, 2, 3]) for (const hierarchy of [0, 1])
  for (const traversal of [0, 1, 2]) for (const layout of [0, 1, 2]) {
    const c = defaults(dim);
    Object.assign(c, {api, hierarchy, traversal, layout, outerTraversal: traversal, outerLayout: layout,
      worker: hierarchy ? 4 : 0, spec: (api + traversal + layout) % 2});
    if (api === 0) c.threads = Array(dim).fill(1);
    c.begin = dim === 2 ? [1, 2] : [2];
    c.end = dim === 2 ? [5, 7] : [17];
    c.stride = dim === 2 ? [2, 3] : [3];
    cases.push({config: c, coverage: true});
  }
for (const dim of [1, 2]) for (const worker of [0, 1, 2, 3, 4, 5, 6])
  for (const layout of [0, 1, 2]) for (const traversal of [0, 1, 2]) {
    const c = {...defaults(dim), hierarchy: 0, worker, layout, traversal};
    const unsupported = dim === 2 && [1, 3, 5].includes(worker);
    cases.push({config: c, coverage: !unsupported, error: unsupported});
  }
for (const dim of [1, 2]) for (const layout of [1, 2]) for (const traversal of [1, 2]) {
  const c = defaults(dim);
  Object.assign(c, {hierarchy: 0, worker: 0, layout, traversal});
  c.blocks = Array(dim).fill(1);
  c.threads = dim === 2 ? [4, 4] : [16];
  c.end = Array(dim).fill(2);
  cases.push({config: c, coverage: true}); // More workers than elements.
}
for (const dim of [1, 2]) {
  const empty = defaults(dim);
  empty.begin = [...empty.end];
  cases.push({config: empty, coverage: true});
  const linearOuter = {...defaults(dim), outerWorker: 1};
  cases.push({config: linearOuter, coverage: dim === 1, error: dim === 2});
}
// Independent policies at the outer and inner levels; multiple outer iterations per block.
for (const dim of [1, 2]) for (const outerTraversal of [1, 2]) for (const outerLayout of [1, 2])
  for (const traversal of [1, 2]) for (const layout of [1, 2]) {
    const c = {...defaults(dim), outerTraversal, outerLayout, traversal, layout};
    c.blocks = Array(dim).fill(1);
    cases.push({config: c, coverage: true});
  }
const invalid = [
  {...defaults(), dim: 3}, {...defaults(), api: 5}, {...defaults(), executor: 6},
  {...defaults(), api: 0}, {...defaults(), worker: 9}, {...defaults(), traversal: 3},
  {...defaults(), layout: -1}, {...defaults(), blocks: [64, 64]},
  {...defaults(), threads: [0, 4]}, {...defaults(), tile: [0, 4]},
  {...defaults(), stride: [1, 0]}, {...defaults(), begin: [5, 0]},
  {...defaults(), end: [64, 64]}, {...defaults(), end: [4, 65]},
];
// Deliberately serialize malformed dimensionality through a raw request below instead.
for (const c of invalid.filter(c => c.dim !== 3)) cases.push({config: c, error: true});
const requests = cases.map(test => serialize(test.config));
requests.push("", "1 3 0 0 0 0 0 0 0 0 0 0", `${serialize(defaults())} trailing`, "x".repeat(2048));

function native(binary) {
  const result = spawnSync(binary, [], {input: requests.join("\n") + "\n", encoding: "utf8", maxBuffer: 100 * 1024 * 1024});
  assert.equal(result.status, 0, `${binary} failed: ${result.stderr}`);
  return result.stdout.trim().split("\n").map(line => JSON.parse(line));
}
const reference = native(`${build}mapping-native`);
assert.equal(reference.length, requests.length);
if (manifest.singleHeader) {
  const exported = readFileSync(`${build}godbolt-validation.cpp`, "utf8");
  assert(exported.includes(`#include <${manifest.godboltHeaderUrl}>`), "Godbolt export must import the configured HTTP header");
  assert(!exported.includes("#define ALPAKA_VERSION_MAJOR"), "Godbolt export must not embed an alpaka header");
  assert(exported.includes(manifest.build), "Godbolt export must identify its build");
  assert.equal(readFileSync(`${build}site/godbolt-validation.cpp`, "utf8"), exported, "Stale downloadable validator");
  assert(existsSync(`${build}mapping-single-header`));
  assert.deepEqual(native(`${build}mapping-single-header`), reference, "Godbolt single header differs from regular headers");
}
let assignmentCount = 0;
for (let i = 0; i < requests.length; ++i) {
  const actual = reference[i];
  if (wasm) {
    const browser = JSON.parse(wasm.ccall("mapping_trace", "string", ["string"], [requests[i]]));
    assert.deepEqual(browser, actual, `Native/WASM disagreement for request ${i}: ${requests[i]}`);
  }
  const test = cases[i];
  if (!test || test.error) { assert(actual.error, `Expected rejected configuration ${i}`); continue; }
  assert(!actual.error, `Unexpected rejection ${i}: ${actual.error}`);
  assert.equal(actual.build, manifest.build);
  const c = test.config;
  const owners = ownersOf(actual);
  const expected = new Set();
  const rows = c.dim === 2 ? c.end[0] : 1;
  for (let row = 0; row < rows; ++row) for (let col = 0; col < c.end.at(-1); ++col) {
    const index = c.dim === 2 ? [row, col] : [col];
    if (eligible(c, index)) expected.add(index.join(","));
  }
  if (test.coverage) assert.deepEqual(new Set(owners.keys()), expected, `Coverage mismatch in configuration ${i}`);
  if ((c.hierarchy && c.worker === 4) || (!c.hierarchy && c.worker === 0)) {
    for (const entries of owners.values()) assert.equal(entries.length, 1, `Duplicate work for configuration ${i}`);
  }
  for (const worker of actual.workers) for (const a of worker.assignments) {
    ++assignmentCount;
    assert.deepEqual(a.index, a.local.map((v, d) => v + a.origin[d]));
    assert.equal(a.inBounds, a.index.every((v, d) => v >= c.begin[d] && v < c.end[d]));
    assert.equal(a.linear, a.inBounds ? a.index.reduce((offset, v, d) => offset * c.end[d] + v, 0) : null);
  }
}

// Known independent oracle: 4 threads covering 16 elements, not an adapter-vs-itself check.
function evaluate(c) {
  const result = spawnSync(`${build}mapping-native`, [], {input: serialize(c) + "\n", encoding: "utf8"});
  assert.equal(result.status, 0);
  return JSON.parse(result.stdout);
}
const known = {...defaults(1), hierarchy: 0, worker: 0, blocks: [1], threads: [4], layout: 1};
assert.deepEqual(evaluate(known).workers.map(w => w.assignments.map(a => a.index[0])),
  [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]);
known.layout = 2;
assert.deepEqual(evaluate(known).workers.map(w => w.assignments.map(a => a.index[0])),
  [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]);
assert.equal(evaluate(defaults(1)).workers[0].inner.traversal, "tiled");
assert.equal(evaluate(defaults(2)).workers[0].inner.traversal, "flat");
const host = {...defaults(1), api: 0, threads: [1]};
assert.equal(evaluate(host).workers[0].inner.layout, "contiguous");
assert.equal(evaluate(defaults(1)).workers[0].inner.layout, "strided");
// This is a separate execution path: real alpaka queue/kernel launch, writes to device buffers,
// copy-back to host buffers, and only THEN comparison with the explorer's trace.
function compareBackend(binary, mode, allowUnavailable = false) {
  const result = spawnSync(binary, [mode, "--json"], {encoding: "utf8", maxBuffer: 50 * 1024 * 1024});
  if (allowUnavailable && result.status === 77) {
    console.warn(`NOT VALIDATED on CUDA hardware: ${result.stderr.trim()}`);
    return 0;
  }
  assert.equal(result.status, 0, `${binary} ${mode} failed: ${result.stderr}`);
  const launches = result.stdout.trim().split("\n").map(line => JSON.parse(line));
  assert.equal(launches.length, 31, "Unexpected real-backend validation coverage");
  const expectedRun = spawnSync(`${build}mapping-native`, [], {
    input: launches.map(launch => launch.request).join("\n") + "\n", encoding: "utf8", maxBuffer: 50 * 1024 * 1024,
  });
  assert.equal(expectedRun.status, 0, expectedRun.stderr);
  const expected = expectedRun.stdout.trim().split("\n").map(line => JSON.parse(line));
  const normalize = trace => trace.workers.map(({id, block, thread, assignments}) => ({id, block, thread, assignments}));
  for (let i = 0; i < launches.length; ++i) {
    assert.deepEqual(launches[i].workers, normalize(expected[i]), `Real-backend mismatch: ${launches[i].case}`);
    if (wasm) {
      const browser = JSON.parse(wasm.ccall("mapping_trace", "string", ["string"], [launches[i].request]));
      assert.deepEqual(launches[i].workers, normalize(browser), `Real backend vs WASM mismatch: ${launches[i].case}`);
    }
  }
  const corrupt = spawnSync(binary, [mode, "--inject-error"], {encoding: "utf8"});
  assert.equal(corrupt.status, 1, "Validator must reject a corrupted returned-buffer record");
  assert.match(corrupt.stderr, /UID or iteration differs/);
  console.log(`PASS: ${launches.length} real ${mode.slice(2).toUpperCase()} launches (${binary.split("/").at(-1)}); host buffers match explorer${wasm ? " and WASM" : ""}; injected corruption rejected.`);
  return launches.length;
}
compareBackend(`${build}validate-cpu`, "--cpu");
if (manifest.singleHeader) compareBackend(`${build}validate-godbolt-cpu`, "--cpu");
if (manifest.godboltHttpCompiled) {
  assert.equal(createHash("sha256").update(readFileSync(`${build}godbolt-header-http.hpp`)).digest("hex"),
    manifest.godboltHttpHeaderSha256, "Fetched Godbolt header changed since compilation");
  compareBackend(`${build}validate-godbolt-http-cpu`, "--cpu");
}
if (process.env.ALPAKA_REQUIRE_CUDA === "1") assert(manifest.cudaCompiled, "Rebuild with --cuda to require GPU validation");
if (manifest.cudaCompiled) {
  compareBackend(`${build}validate-cuda`, "--cpu");
  compareBackend(`${build}validate-cuda`, "--cuda", process.env.ALPAKA_REQUIRE_CUDA !== "1");
  if (manifest.singleHeader) {
    compareBackend(`${build}validate-godbolt-cuda`, "--cpu");
    compareBackend(`${build}validate-godbolt-cuda`, "--cuda", process.env.ALPAKA_REQUIRE_CUDA !== "1");
  }
  if (manifest.godboltHttpCompiled) {
    compareBackend(`${build}validate-godbolt-http-cuda`, "--cpu");
    compareBackend(`${build}validate-godbolt-http-cuda`, "--cuda", process.env.ALPAKA_REQUIRE_CUDA !== "1");
  }
}
console.log(`PASS: ${requests.length} configurations; ${assignmentCount} assignments; independent sequence/coverage checks`
  + `${wasm ? "; native/WASM exact equality" : ""}${manifest.singleHeader ? "; Godbolt header exact equality" : ""}.`);

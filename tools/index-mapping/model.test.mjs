// SPDX-License-Identifier: MPL-2.0
import assert from "node:assert/strict";
import {test} from "node:test";
import {defaults, fromHash, innerRange, linkedRanges, resolutionDescription, serialize, snippet} from "./web/model.mjs";

const hash = c => `#${encodeURIComponent(JSON.stringify(c))}`;
const trace = {blocks: [2, 2], threads: [2, 4], effectiveExecutor: 4, maxThreadsPerBlock: 8, threadLimit: 1024,
  workers: [{inner: {traversal: "flat"}, outer: {traversal: "flat"}}]};

test("new links preserve independent launch, outer range, and inner range controls", () => {
  const config = {...defaults(), spec: 1, frameExtent: [4, 8], tile: [2, 3], maxThreadsPerBlock: 8,
    outerDomainBounds: 0, tileOuterStride: 0, innerRangeMode: 1, outerBegin: [1, 2], outerEnd: [6, 8], outerStride: [2, 3],
    innerBegin: [1, 0], innerEnd: [4, 5], innerStride: [2, 2]};
  assert.deepEqual(fromHash(hash(config)), config);
  assert.equal(serialize(config).split(" ")[0], "6");
  assert(!("coupleTile" in fromHash(hash({...config, coupleTile: 1}))));
  assert.deepEqual(innerRange(config), {begin: [1, 0], end: [4, 5], stride: [2, 2]});
});

test("legacy links flatten coupling once and migrate tile iteration into independent ranges", () => {
  const legacy = {...defaults(), spec: 1, tile: [3, 5], frameExtent: [4, 8], coupleTile: 1};
  for (const key of ["outerBegin", "outerEnd", "outerStride", "innerBegin", "innerEnd", "innerStride", "innerRangeMode", "outerDomainBounds", "tileOuterStride", "directDomainBounds", "directBegin", "directEnd"])
    delete legacy[key];
  const migrated = fromHash(hash(legacy));
  assert.deepEqual(migrated.tile, [4, 8]);
  assert.deepEqual(migrated.outerStride, [4, 8]);
  assert.deepEqual(migrated.outerBegin, legacy.begin);
  assert.deepEqual(migrated.outerEnd, legacy.end);
  assert.equal(migrated.innerRangeMode, 0);
  assert(!("coupleTile" in migrated));
  migrated.frameExtent[0] = 16;
  assert.deepEqual(migrated.tile, [4, 8]);
  assert.deepEqual(migrated.outerStride, [4, 8]);
  assert.deepEqual(innerRange(migrated), {begin: [0, 0], end: [4, 8], stride: [1, 1]});
});

test("snippets distinguish launch frames, outer ranges, and inner tile shorthand", () => {
  const config = {...defaults(), spec: 1, frameExtent: [4, 8], tile: [2, 3], tileOuterStride: 0};
  const code = snippet(config, trace);
  assert.match(code, /FrameSpec\{Vec\{2u, 2u\}, Vec\{4u, 8u\}/);
  assert.match(code, /auto tileExtent = Vec\{2u, 3u\}/);
  assert.match(code, /IdxRange\{tileExtent\}/);
  assert.match(code, /IdxRange\{Vec\{0u, 0u\}, Vec\{4u, 8u\}, Vec\{2u, 4u\}\}/);
  assert.match(code, /Derived ThreadSpec: blocks=\(2, 2\), threads=\(2, 4\)/);
  assert.match(code, /anyExecutor → gpuCuda/);
  assert.match(code, /maxThreadsPerBlock = 8/);
  assert.match(code, /No hardware queried/);
  const compiled = snippet({...config, frameType: 1}, trace);
  assert.match(compiled, /CVec<uint32_t, 4u, 8u>\{\}/);
  assert.match(compiled, /Compile-time thread limit = 1024/);
  assert.match(compiled, /Runtime device limit is not used/);
  assert.match(resolutionDescription({...config, api: 0}, {...trace, effectiveExecutor: 1}), /cpuSerial collapses/);
});

test("explicit inner snippets include both full ranges, composition, and both guard bounds", () => {
  const config = {...defaults(), innerRangeMode: 1, outerDomainBounds: 0, tileOuterStride: 0,
    begin: [1, 2], outerBegin: [2, 1],
    outerEnd: [8, 7], outerStride: [3, 2], innerBegin: [1, 1], innerEnd: [3, 5], innerStride: [1, 2]};
  const code = snippet(config, trace);
  assert(!code.includes("tileExtent"));
  assert.match(code, /IdxRange\{Vec\{2u, 1u\}, Vec\{8u, 7u\}, Vec\{3u, 2u\}\}/);
  assert.match(code, /IdxRange\{Vec\{1u, 1u\}, Vec\{3u, 5u\}, Vec\{1u, 2u\}\}/);
  assert.match(code, /globalIdx = outerIdx \+ innerIdx/);
  assert.match(code, /1u <= globalIdx\[0u\]/);
  assert.match(code, /2u <= globalIdx\[1u\]/);
});


test("only tile follows outer stride by default; domain links are explicit and independent", () => {
  const config = {...defaults(), begin: [1, 2], end: [6, 9], outerStride: [3, 5]};
  assert.equal(config.outerDomainBounds, 0);
  assert.equal(config.directDomainBounds, 0);
  assert.equal(config.tileOuterStride, 1);
  const linked = linkedRanges(config);
  assert.deepEqual(linked.outerBegin, [0, 0]);
  assert.deepEqual(linked.outerEnd, [4, 8]);
  assert.deepEqual(linked.tile, [3, 5]);
  assert.notEqual(linked.tile, config.outerStride);
  assert.deepEqual(fromHash(hash(config)), linked);
  assert.deepEqual(linkedRanges({...config, outerDomainBounds: 1}).outerBegin, [1, 2]);
  const direct = {...config, hierarchy: 0, directDomainBounds: 1};
  assert.deepEqual(linkedRanges(direct).directBegin, [1, 2]);
  assert.deepEqual(linkedRanges(direct).directEnd, [6, 9]);
  const unlinked = {...linked, tileOuterStride: 0, outerStride: [1, 1]};
  assert.deepEqual(linkedRanges(unlinked).tile, [3, 5]);
  assert.deepEqual(fromHash(hash(unlinked)), unlinked);
});

test("legacy stride-to-tile link flattens once without reversing its old source", () => {
  const legacy = {...defaults(), outerDomainBounds: 1, outerTileStride: 1, tile: [3, 5],
    outerStride: [1, 1], begin: [1, 2], end: [6, 9]};
  for (const key of ["tileOuterStride", "directDomainBounds", "directBegin", "directEnd"]) delete legacy[key];
  const migrated = fromHash(hash(legacy));
  assert.deepEqual(migrated.outerStride, [3, 5]);
  assert.deepEqual(migrated.outerBegin, [1, 2]);
  assert.equal(migrated.tileOuterStride, 0);
  assert(!("outerTileStride" in migrated));
});

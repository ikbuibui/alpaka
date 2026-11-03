// SPDX-License-Identifier: MPL-2.0
import {expect, test} from "@playwright/test";
import {spawnSync} from "node:child_process";
import {fileURLToPath} from "node:url";
import {defaults, ownersOf, serialize} from "./web/model.mjs";

function reference(config) {
  const result = spawnSync(fileURLToPath(new URL("build/mapping-native", import.meta.url)), [],
    {input: serialize(config) + "\n", encoding: "utf8"});
  expect(result.status).toBe(0);
  return JSON.parse(result.stdout);
}

async function ready(page, hash = "") {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(`/${hash}`);
  await expect(page.locator("#status")).toContainText("Live C++");
  expect(errors).toEqual([]);
}

async function assertDrawing(page, config, selectedWorker = 0, iteration = null) {
  const trace = reference(config);
  expect(trace.error).toBeUndefined();
  const expectedOwners = ownersOf(trace);
  const actual = await page.locator("[data-cell]").evaluateAll(cells => cells.map(cell => ({
    index: cell.dataset.cell, owners: Number(cell.dataset.ownerCount), highlighted: cell.dataset.highlighted === "true",
  })));
  const assignments = trace.workers[selectedWorker].assignments.filter(a => a.inBounds && (!iteration
    || (a.outerIteration === iteration.outer && a.innerIteration === iteration.inner)));
  const highlighted = new Set(assignments.map(a => a.index.join(",")));
  expect(actual.length).toBe(config.end.reduce((a, b) => a * b, 1));
  for (const cell of actual) {
    expect(cell.owners, cell.index).toBe(new Set((expectedOwners.get(cell.index) ?? []).map(e => e.worker.id)).size);
    expect(cell.highlighted, cell.index).toBe(highlighted.has(cell.index));
  }
}

test("SVG ownership and worker selection agree with native C++", async ({page}) => {
  await ready(page);
  const config = defaults();
  await assertDrawing(page, config);
  await page.locator('[data-worker="7"]').click();
  await assertDrawing(page, config, 7);
  await expect(page.locator("#calculation")).toContainText("Worker T7");
  await page.locator("#all-iterations").uncheck();
  await page.locator("#inner-iteration").fill("1");
  await page.locator("#inner-iteration").dispatchEvent("change");
  await assertDrawing(page, config, 7, {outer: 0, inner: 1});
});

test("selecting a cell traces its owning worker and C++ index", async ({page}) => {
  await ready(page);
  await page.locator('[data-cell="3,6"]').click();
  const native = ownersOf(reference(defaults())).get("3,6")[0];
  await expect(page.locator(`[data-worker="${native.worker.id}"]`)).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#calculation")).toContainText("(3, 6)");
  await expect(page.locator("#calculation")).toContainText("30");
});

test("1D direct mapping, offsets and strides match native C++", async ({page}) => {
  await ready(page);
  await page.locator("#dim").selectOption("1");
  await page.locator("#preset").selectOption("direct");
  for (const [field, value] of [["begin", "2"], ["end", "17"], ["stride", "3"]]) {
    await page.locator(`[data-field="${field}"]`).fill(value);
    await page.locator(`[data-field="${field}"]`).dispatchEvent("change");
  }
  const config = {...defaults(1), hierarchy: 0, worker: 0, begin: [2], end: [17], stride: [3]};
  await assertDrawing(page, config);
  await page.locator("#layout").selectOption("2");
  config.layout = 2;
  await assertDrawing(page, config);
  await expect(page.locator("#calculation")).toContainText("contiguous layout");
});

test("partial tiles expose guarded assignments", async ({page}) => {
  await ready(page);
  await page.locator("#preset").selectOption("uneven");
  const config = {...defaults(), end: [3, 7]};
  await assertDrawing(page, config);
  await page.locator("#tab-diagnostics").click();
  await expect(page.locator("#diagnostics")).toContainText("21/21");
  await expect(page.locator("#diagnostics .metrics > div").nth(2).locator("strong")).not.toHaveText("0");
});

test("invalid configurations clear the previous result", async ({page}) => {
  await ready(page);
  await page.locator("#api").selectOption("0");
  await expect(page.locator("#status")).toHaveClass(/error/);
  await expect(page.locator("[data-cell]")).toHaveCount(0);
  await expect(page.locator("#calculation")).toBeEmpty();
  await page.locator('[data-field="threads"][data-dimension="1"]').fill("1");
  await page.locator('[data-field="threads"][data-dimension="1"]').dispatchEvent("change");
  await expect(page.locator("#status")).not.toHaveClass(/error/);
  await assertDrawing(page, {...defaults(), api: 0, threads: [1, 1]});
});

test("FrameSpec is visibly an assumption and snippets reflect configuration", async ({page}) => {
  await ready(page);
  await page.locator("#spec").selectOption("1");
  await expect(page.locator("#status")).toContainText("NOT resolved");
  await expect(page.locator("#frames-field")).toBeVisible();
  await page.locator("#tab-code").click();
  await expect(page.locator("#code")).toContainText("onHost::FrameSpec");
  await expect(page.locator("#code")).toContainText("Actual backend FrameSpec resolution is NOT modeled");
});

test("shared configuration loads without network services and narrow layouts remain usable", async ({page}) => {
  const config = {...defaults(1), hierarchy: 0, worker: 0, layout: 2};
  const requests = [];
  page.on("request", request => requests.push(request.url()));
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  await assertDrawing(page, config);
  expect(requests.every(url => url.startsWith("http://127.0.0.1:8765/"))).toBe(true);
  await page.setViewportSize({width: 390, height: 844});
  await expect(page.locator("#config-form")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("Godbolt validation download includes the configured HTTP header, not an embedded copy", async ({page}) => {
  await ready(page);
  const manifest = await (await page.request.get("/manifest.json")).json();
  test.skip(!manifest.singleHeader, "Build with --single-header to expose the download");
  await expect(page.locator("#download-validator")).toBeVisible();
  const response = await page.request.get("/godbolt-validation.cpp");
  expect(response.ok()).toBe(true);
  const source = await response.text();
  expect(source).toContain(manifest.build);
  expect(source).toContain("namespace backendValidation");
  expect(source).toContain("onHost::memcpy(queue, recordsH, recordsD)");
  expect(source).toContain(`#include <${manifest.godboltHeaderUrl}>`);
  expect(source).not.toContain("#define ALPAKA_VERSION_MAJOR");
  expect(source.length).toBeLessThan(200000);
});

test("composition stays on the map and shows traced 1D and 2D summands", async ({page}) => {
  const config = {...defaults(1), end: [32], layout: 1, outerLayout: 1};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  await page.locator('[data-cell="14"]').click();
  await page.mouse.move(0, 0);
  await expect(page.locator(".sum-equation")).toHaveAttribute("data-origin", "8");
  await expect(page.locator(".sum-equation")).toHaveAttribute("data-local", "6");
  await expect(page.locator(".sum-equation")).toHaveAttribute("data-global", "14");
  await expect(page.locator(".composition-overlay path")).toHaveCount(1);
  await expect(page.locator("#step-label")).toHaveText("2 / 4");
  await page.locator("#labels").selectOption("sum");
  await expect(page.locator('[data-cell="14"] > text').first()).toHaveText("8 + 6");
  await page.locator("#show-composition").uncheck();
  await expect(page.locator("#composition")).toBeHidden();
  await expect(page.locator(".composition-overlay")).toHaveCount(0);
  await page.locator("#show-composition").check();
  await expect(page.locator("#composition")).toBeVisible();
  await page.locator("#dim").selectOption("2");
  await page.locator('[data-cell="3,6"]').click();
  await page.mouse.move(0, 0);
  await expect(page.locator(".sum-equation")).toHaveAttribute("data-origin", "2,4");
  await expect(page.locator(".sum-equation")).toHaveAttribute("data-local", "1,2");
  await expect(page.locator(".local-inset svg")).toHaveAttribute("aria-label", "Local index (1, 2) in tile (2, 4)");
  await expect(page.locator('[data-worker="14"]')).toBeInViewport();
  await page.locator("#timeline").scrollIntoViewIfNeeded();
  await expect(page.locator('[data-timeline-worker="14"]')).toBeInViewport();
  await expect(page.locator('#timeline [aria-current="step"]')).toBeInViewport();
  await page.locator("#hierarchy").selectOption("0");
  await expect(page.locator("#composition")).toBeHidden();
  await expect(page.locator("#show-composition")).toBeDisabled();
  await expect(page.locator('#labels option[value="sum"]')).toHaveAttribute("disabled", "");
});

test("hover previews decomposition without changing the pinned selection or cursor", async ({page}) => {
  await ready(page);
  await page.locator('[data-cell="3,6"]').hover();
  await expect(page.locator("#composition")).toContainText("Preview");
  await expect(page.locator(".sum-equation")).toHaveAttribute("data-global", "3,6");
  await expect(page.locator('[data-worker="0"]')).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#step-label")).toHaveText("1 / 2");
  await page.mouse.move(0, 0);
  await expect(page.locator("#composition")).toContainText("Selected");
  await expect(page.locator(".sum-equation")).toHaveAttribute("data-global", "0,0");
  await page.locator('[data-worker="7"]').hover();
  await expect(page.locator("#composition")).toContainText("Preview");
  await expect(page.locator("#composition")).toContainText("Thread (0, 3)");
  await page.mouse.move(0, 0);
  await page.locator('[data-cell="3,6"]').focus();
  await page.keyboard.press("Enter");
  await expect(page.locator('[data-cell="3,6"]')).toBeFocused();
  await expect(page.locator(".sum-equation")).toHaveAttribute("data-global", "3,6");
  await page.locator('[data-worker="0"]').hover(); // Repainting the map must preserve keyboard focus.
  await expect(page.locator('[data-cell="3,6"]')).toBeFocused();
});

for (const [name, config] of [
  ["nested strided", {...defaults(1), end: [32], layout: 1, outerLayout: 1}],
  ["unequal block loop lengths", {...defaults(1), end: [40], layout: 1, outerLayout: 1}],
  ["2D nonzero partial contiguous", {...defaults(), blocks: [1, 1], begin: [1, 2], end: [5, 7], layout: 2, outerLayout: 2}],
  ["idle direct workers", {...defaults(1), hierarchy: 0, worker: 0, end: [2], layout: 1}],
]) test(`timeline events match native C++: ${name}`, async ({page}) => {
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  const trace = reference(config);
  const keys = [...new Set(trace.workers.flatMap(w => w.assignments.map(a => `${a.outerIteration}:${a.innerIteration}`)))];
  keys.sort((a, b) => {
    const [ao, ai] = a.split(":").map(Number), [bo, bi] = b.split(":").map(Number);
    return ao - bo || ai - bi;
  });
  const events = await page.locator("[data-event-worker]").evaluateAll(buttons => buttons.map(b => ({
    worker: Number(b.dataset.eventWorker), step: Number(b.dataset.eventStep),
    index: b.textContent, inBounds: b.dataset.inBounds === "true",
  })));
  expect(events.length).toBe(trace.workers.reduce((sum, w) => sum + w.assignments.length, 0));
  for (const event of events) {
    const assignment = trace.workers[event.worker].assignments.find(a => `${a.outerIteration}:${a.innerIteration}` === keys[event.step]);
    expect(event.index).toBe(`(${assignment.index.join(", ")})`);
    expect(event.inBounds).toBe(assignment.inBounds);
  }
  await expect(page.locator("[data-timeline-worker]")).toHaveCount(trace.workers.length);
  if (trace.workers.some(w => !w.assignments.length)) await expect(page.locator(".idle-event").first()).toBeVisible();
  await expect(page.locator(".timeline-note").first()).toContainText("not hardware lockstep");
});

test("stepping paints every worker's cohort and retains reversible, fixed-color history", async ({page}) => {
  const config = {...defaults(1), end: [32], layout: 1, outerLayout: 1};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  const cell = index => page.locator(`[data-cell="${index}"]`);
  for (const index of [0, 1, 2, 3, 8, 9, 10, 11]) await expect(cell(index)).toHaveAttribute("data-phase", "current");
  await expect(cell(14)).toHaveAttribute("data-phase", "future");
  await page.locator("#step-forward").click();
  for (const index of [4, 5, 6, 7, 12, 13, 14, 15]) await expect(cell(index)).toHaveAttribute("data-phase", "current");
  await expect(cell(0)).toHaveAttribute("data-phase", "past");
  const muted = await cell(0).locator("rect").first().getAttribute("fill");
  await page.locator("#step-forward").click();
  expect(await cell(0).locator("rect").first().getAttribute("fill")).toBe(muted);
  await page.locator("#keep-history").uncheck();
  await expect(cell(0)).toHaveAttribute("data-visit-count", "0");
  await expect(cell(0).locator("rect").first()).toHaveAttribute("fill", "#fff");
  await page.locator("#keep-history").check();
  expect(await cell(0).locator("rect").first().getAttribute("fill")).toBe(muted);
  await page.locator("#timeline-step").fill("0");
  await expect(cell(0)).toHaveAttribute("data-phase", "current");
  await expect(cell(4)).toHaveAttribute("data-phase", "future");
  await page.locator('[data-step="1"]').click();
  await expect(page.locator("#step-label")).toHaveText("2 / 4");
  await page.locator('[data-event-worker="6"][data-event-step="1"]').focus();
  await page.keyboard.press("Enter");
  await expect(page.locator('[data-worker="6"]')).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".sum-equation")).toHaveAttribute("data-global", "14");
  await expect(page.locator('[data-event-worker="6"][data-event-step="1"]')).toBeFocused();
});

test("guarded events remain inspectable but do not color processed cells", async ({page}) => {
  await ready(page);
  await page.locator("#preset").selectOption("uneven");
  const guarded = page.locator('[data-in-bounds="false"]').first();
  const worker = await guarded.getAttribute("data-event-worker");
  await guarded.click();
  await expect(page.locator(`[data-worker="${worker}"]`)).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#composition")).toContainText("rejected by guard");
  await expect(page.locator("#calculation")).toContainText("rejected by the bounds guard");
  await expect(page.locator(".timeline-event.selected-event")).toHaveClass(/guarded/);
  const native = reference({...defaults(), end: [3, 7]});
  const [outer, inner] = await page.locator("#timeline-step").getAttribute("aria-valuetext").then(text =>
    text.match(/outer (\d+), inner (\d+)/).slice(1).map(Number));
  const valid = new Set(native.workers.flatMap(w => w.assignments.filter(a => a.inBounds
    && a.outerIteration === outer && a.innerIteration === inner).map(a => a.index.join(","))));
  const painted = await page.locator('[data-cell][data-phase="current"]').evaluateAll(cells => cells.map(c => c.dataset.cell));
  expect(new Set(painted)).toEqual(valid);
});

test("repeated assignments retain multiple visits and cycle through their owners", async ({page}) => {
  const config = {...defaults(1), worker: 6};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  const entries = ownersOf(reference(config)).get("0");
  expect(entries.length).toBeGreaterThan(1);
  const cell = page.locator('[data-cell="0"]');
  await expect(cell).toHaveAttribute("data-visit-count", String(entries.length));
  await expect(cell.locator(".visit-count")).toHaveText(`${entries.length}×`);
  await cell.click();
  await expect(page.locator(`[data-worker="${entries[1].worker.id}"]`)).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#table")).toContainText("assignments");
});

test("block filtering and collapsing never change the shared iteration cursor", async ({page}) => {
  await ready(page);
  await expect(page.locator("[data-timeline-worker]")).toHaveCount(16);
  await page.locator('[data-collapse-block="0,0"]').click();
  await expect(page.locator('[data-collapse-block="0,0"]')).toHaveAttribute("aria-expanded", "false");
  await expect(page.locator("[data-timeline-worker]")).toHaveCount(12);
  await page.locator('[data-collapse-block="0,0"]').click();
  await page.locator('[data-worker="7"]').click();
  await page.locator("#timeline-scope").selectOption("block");
  await expect(page.locator("[data-timeline-worker]")).toHaveCount(4);
  await page.locator("#step-forward").click();
  await expect(page.locator("#step-label")).toHaveText("2 / 2");
  await page.locator("#timeline-scope").selectOption("all");
  await expect(page.locator("[data-timeline-worker]")).toHaveCount(16);
  await expect(page.locator("#step-label")).toHaveText("2 / 2");
});

test("playback pauses, stops at the end, and respects reduced motion", async ({page}) => {
  const config = {...defaults(1), end: [32]};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  await page.clock.install();
  await page.locator("#play").click();
  await expect(page.locator("#play")).toHaveText("Pause");
  await page.clock.runFor(900);
  await expect(page.locator("#step-label")).toHaveText("2 / 4");
  await page.locator("#play").click();
  await page.clock.runFor(1800);
  await expect(page.locator("#step-label")).toHaveText("2 / 4");
  await page.locator("#play").click();
  await page.clock.runFor(1800);
  await expect(page.locator("#step-label")).toHaveText("4 / 4");
  await expect(page.locator("#play")).toHaveAttribute("aria-pressed", "false");
  await page.locator("#play").click(); // Replay from the beginning.
  await expect(page.locator("#step-label")).toHaveText("1 / 4");
  await page.emulateMedia({reducedMotion: "reduce"});
  await expect(page.locator("#play")).toBeDisabled();
  await expect(page.locator("#play")).toHaveText("Play");
  await page.locator("#step-forward").click();
  await expect(page.locator("#step-label")).toHaveText("2 / 4");
});

test("empty ranges and invalid configurations clear timeline state safely", async ({page}) => {
  const config = {...defaults(1), begin: [16]};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  await expect(page.locator("#timeline")).toContainText("All workers are idle");
  await expect(page.locator("#step-label")).toHaveText("0 / 0");
  await expect(page.locator("#play")).toBeDisabled();
  await expect(page.locator("#timeline-step")).toBeDisabled();
  await page.locator("#api").selectOption("0");
  await expect(page.locator("#status")).toHaveClass(/error/);
  await expect(page.locator("#timeline")).toBeEmpty();
  await expect(page.locator("#composition")).toBeEmpty();
});

test("large timelines are windowed and hierarchical layouts remain narrow-screen safe", async ({page}) => {
  const config = {...defaults(1), blocks: [1], threads: [1], tile: [1], end: [64]};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  await expect(page.locator("[data-event-worker]")).toHaveCount(24);
  await page.locator("#timeline-step").fill("63");
  await expect(page.locator("#step-label")).toHaveText("64 / 64");
  await page.locator("#timeline").scrollIntoViewIfNeeded();
  await expect(page.locator('[data-event-step="63"]')).toBeInViewport();
  await expect(page.locator("[data-event-worker]")).toHaveCount(24);
  await page.setViewportSize({width: 390, height: 844});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator("#dim").selectOption("2");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("capture the standalone prototype", async ({page}, testInfo) => {
  await ready(page);
  await page.screenshot({path: testInfo.outputPath("explorer.png"), fullPage: true});
  await page.locator('[data-cell="3,6"]').click();
  await page.mouse.move(0, 0);
  await page.screenshot({path: testInfo.outputPath("composition-history.png"), fullPage: true});
});

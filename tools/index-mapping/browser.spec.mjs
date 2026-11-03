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

async function expandTimeline(page) {
  if (await page.locator("#expand-timeline").getAttribute("aria-expanded") === "false")
    await page.locator("#expand-timeline").click();
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

test("twelve blocks use ten stable color families and repeated colors stay selectable", async ({page}) => {
  const config = {...defaults(), blocks: [3, 4], end: [12, 16]};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  const colors = await page.locator(".block-group").evaluateAll(groups =>
    groups.map(g => g.style.getPropertyValue("--block-color")));
  expect(new Set(colors.slice(0, 10)).size).toBe(10);
  expect(colors[10]).toBe(colors[0]);
  expect(colors[11]).toBe(colors[1]);
  await page.locator('[data-worker="40"]').click();
  await assertDrawing(page, config, 40);
  await expect(page.locator('[data-worker="40"] .selected-badge')).toHaveText("Selected");
  await expect(page.locator('[data-worker="0"] .selected-badge')).toHaveCount(0);
  await expect(page.locator("#worker-selection")).toContainText("Selected: T40");
});

test("selected-thread rings span past, current and future without changing temporal fills", async ({page}) => {
  const config = {...defaults(), blocks: [3, 4], end: [12, 16]};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  await expandTimeline(page);
  await page.locator("#step-forward").click();
  await page.mouse.move(0, 0);
  await assertDrawing(page, config);
  const selected = page.locator('[data-highlighted="true"]');
  expect(new Set(await selected.evaluateAll(cells => cells.map(c => c.dataset.phase))))
    .toEqual(new Set(["past", "current", "future"]));
  await expect(selected.locator(".selection-ring")).toHaveCount(await selected.count());
  await expect(selected.locator(".selection-halo")).toHaveCount(await selected.count());
  await expect(page.locator('[data-highlighted="false"] .selection-ring')).toHaveCount(0);
  await expect(selected.locator(".selection-ring").first()).toHaveCSS("stroke", "rgb(23, 33, 44)");
  await expect(selected.locator(".selection-ring").first()).toHaveCSS("stroke-width", "3px");
  await expect(selected.locator(".selection-halo").first()).toHaveCSS("stroke", "rgb(255, 255, 255)");
  await expect(selected.locator(".selection-halo").first()).toHaveCSS("stroke-width", "7px");
  const fills = () => page.locator("[data-cell]").evaluateAll(cells => cells.map(c =>
    [c.dataset.cell, c.dataset.phase, c.querySelector(".cell-fill").getAttribute("fill")]));
  const before = await fills();
  const pinned = await page.locator(".assignment-marker").evaluate(marker => marker.parentElement.dataset.cell);
  await page.locator('[data-cell="0,1"]').hover();
  await expect(page.locator('[data-cell="0,1"] .preview-ring')).toHaveCount(1);
  await expect(page.locator('[data-cell="0,1"] .preview-ring')).toHaveCSS("stroke-dasharray", "3px, 2px");
  await expect(page.locator(`[data-cell="${pinned}"] .assignment-marker`)).toHaveCount(1);
  await assertDrawing(page, config);
  expect(await fills()).toEqual(before);
  await page.locator('[data-worker="7"]').click();
  await assertDrawing(page, config, 7);
  expect(await fills()).toEqual(before);
  await expect(page.locator('.selected-row [data-select-worker="7"]')).toContainText("Selected");
  await expect(page.locator('.selected-row [data-select-worker="7"]')).toHaveCSS("outline-style", "solid");
  await expect(page.locator('[data-worker="7"]')).toHaveCSS("outline-width", "2px");
  await page.locator("#keep-history").uncheck();
  await assertDrawing(page, config, 7); // Hiding history must not erase ownership rings.
  await expect(page.locator('[data-highlighted="true"] .selection-ring')).toHaveCount(
    reference(config).workers[7].assignments.filter(a => a.inBounds).length);
});

test("idle selected threads retain an explicit identity without inventing a footprint", async ({page}) => {
  const config = {...defaults(1), hierarchy: 0, worker: 0, end: [2], layout: 1};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  const idle = reference(config).workers.find(w => !w.assignments.length);
  expect(idle).toBeDefined();
  await page.locator(`[data-worker="${idle.id}"]`).click();
  await expect(page.locator("#worker-selection")).toContainText(`Selected: T${idle.id}`);
  await expect(page.locator("#worker-selection")).toContainText("Idle at this step");
  await expect(page.locator(".selection-ring")).toHaveCount(0);
  await expect(page.locator(".assignment-marker")).toHaveCount(0);
  await expect(page.locator(`[data-worker="${idle.id}"] .selected-badge`)).toBeVisible();
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

test("vector edits preserve focus when tabbing or clicking the next component", async ({page}) => {
  await ready(page);
  const row = page.locator('[data-field="begin"][data-dimension="0"]');
  const column = page.locator('[data-field="begin"][data-dimension="1"]');
  await row.fill("1");
  await page.keyboard.press("Tab");
  await expect(column).toBeFocused();
  await column.fill("2");
  await row.click();
  await expect(row).toBeFocused();
  await expect(column).toHaveValue("2");
  await assertDrawing(page, {...defaults(), begin: [1, 2]});
  await page.locator("#dim").selectOption("1");
  await expect(page.locator('[data-field="begin"]')).toHaveCount(1);
  await expect(page.locator('[data-field="begin"]')).toHaveAttribute("aria-label", "Range begin (inclusive), x");
});

test("partial tiles expose guarded assignments", async ({page}) => {
  await ready(page);
  await page.locator("#preset").selectOption("uneven");
  const config = {...defaults(), end: [3, 7]};
  await assertDrawing(page, config);
  await page.locator("#assignment-details > summary").click();
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
  await page.locator("#assignment-details > summary").click();
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
  await expect(page.locator("#config-form")).toBeHidden();
  await expect(page.locator("#domain-pane")).toBeVisible();
  await page.locator('.pane-toggles [data-toggle-pane="config"]').click();
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
  await expandTimeline(page);
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
  await expandTimeline(page);
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
  await expandTimeline(page);
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
  await expandTimeline(page);
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
  await expandTimeline(page);
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
  await expandTimeline(page);
  await page.locator("#step-forward").click();
  await expect(page.locator("#step-label")).toHaveText("2 / 4");
});

test("empty ranges and invalid configurations clear timeline state safely", async ({page}) => {
  const config = {...defaults(1), begin: [16]};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  await expandTimeline(page);
  await expect(page.locator("#timeline")).toContainText("All workers are idle");
  await expect(page.locator("#step-label")).toHaveText("0 / 0");
  await expect(page.locator("#play")).toBeDisabled();
  await expect(page.locator("#timeline-step")).toBeDisabled();
  await page.locator("#api").selectOption("0");
  await expect(page.locator("#status")).toHaveClass(/error/);
  await expect(page.locator("#timeline")).toBeEmpty();
  await expect(page.locator("#composition")).toBeHidden();
  await expect(page.locator("#composition-content")).toBeEmpty();
});

test("large timelines are windowed and hierarchical layouts remain narrow-screen safe", async ({page}) => {
  const config = {...defaults(1), blocks: [1], threads: [1], tile: [1], end: [64]};
  await ready(page, `#${encodeURIComponent(JSON.stringify(config))}`);
  await expandTimeline(page);
  await expect(page.locator("[data-event-worker]")).toHaveCount(24);
  await page.locator("#timeline-step").fill("63");
  await expect(page.locator("#step-label")).toHaveText("64 / 64");
  expect(await page.locator("#data").evaluate(data => {
    const viewport = data.getBoundingClientRect(), cell = data.querySelector('[data-cell="63"] > rect').getBoundingClientRect();
    return cell.left >= viewport.left && cell.right <= viewport.left + data.clientWidth;
  })).toBe(true); // Following the timeline must reveal the selected index in a wide domain.
  await page.locator("#timeline").scrollIntoViewIfNeeded();
  await expect(page.locator('[data-event-step="63"]')).toBeInViewport();
  await expect(page.locator("[data-event-worker]")).toHaveCount(24);
  await page.setViewportSize({width: 390, height: 844});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator("#dim").selectOption("2");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("wide desktops use the available width and keep the main view above details", async ({page}) => {
  await page.setViewportSize({width: 2560, height: 1080});
  await ready(page);
  const workspace = await page.locator("#workspace").boundingBox();
  const config = await page.locator("#config-pane").boundingBox();
  const workers = await page.locator("#worker-pane").boundingBox();
  const domain = await page.locator("#domain-pane").boundingBox();
  expect(workspace.width).toBe(2528);
  expect(config.width).toBe(240);
  expect(workers.width).toBe(200);
  expect(domain.width).toBeGreaterThan(2000);
  expect(domain.y).toBeLessThan(180);
  expect(domain.height).toBeGreaterThan(500);
  await expect(page.locator("#assignment-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#launch-summary")).toContainText("CUDA · blocks (2, 2) · threads/block (1, 4)");
  const fits = await page.locator("#data").evaluate(data => {
    const viewport = data.getBoundingClientRect();
    return [...data.querySelectorAll("[data-cell] > rect:first-of-type")].every(cell => {
      const rect = cell.getBoundingClientRect();
      return rect.top >= viewport.top && rect.bottom <= viewport.bottom
        && rect.left >= viewport.left && rect.right <= viewport.right;
    });
  });
  expect(fits).toBe(true);
  await expandTimeline(page);
  expect((await page.locator('.timeline-table th[scope="row"]').first().boundingBox()).width).toBeLessThanOrEqual(180);
});

test("short desktop panes retain a usable canvas when configuration is widened", async ({page}) => {
  await page.setViewportSize({width: 901, height: 768});
  await ready(page);
  const workspaceHeight = (await page.locator("#workspace").boundingBox()).height;
  const canvas = page.locator("#data");
  expect((await canvas.boundingBox()).height).toBeGreaterThanOrEqual(200);
  await page.locator("#config-splitter").focus();
  await page.keyboard.press("End");
  expect((await canvas.boundingBox()).height).toBeGreaterThanOrEqual(200);
  expect((await page.locator("#workspace").boundingBox()).height).toBe(workspaceHeight);
  expect(await page.locator("#domain-pane").evaluate(pane => pane.scrollHeight > pane.clientHeight)).toBe(true);
  await page.locator("#all-iterations").uncheck();
  await expect(page.locator("#inner-iteration")).toBeInViewport();
  await canvas.scrollIntoViewIfNeeded();
  await page.locator('[data-cell="0,0"]').click();
  await expect(page.locator("#worker-selection")).toContainText("Selected: T0");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("pane resizing works with keyboard and pointer, persists, and resets independently", async ({page}) => {
  await ready(page);
  const splitter = page.locator("#config-splitter");
  await splitter.focus();
  await page.keyboard.press("ArrowRight");
  await expect(splitter).toHaveAttribute("aria-valuenow", "256");
  expect((await page.locator("#config-pane").boundingBox()).width).toBe(256);
  await page.keyboard.press("Home");
  await expect(splitter).toHaveAttribute("aria-valuenow", "180");
  const handle = await splitter.boundingBox();
  await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
  await page.mouse.down();
  await page.mouse.move(handle.x + handle.width / 2 + 100, handle.y + handle.height / 2, {steps: 5});
  await page.mouse.up();
  await expect(splitter).toHaveAttribute("aria-valuenow", "280");
  await expect(page.locator("body")).not.toHaveClass(/resizing/);
  await page.locator("#workers-splitter").focus();
  await page.keyboard.press("Shift+ArrowRight");
  await expect(page.locator("#workers-splitter")).toHaveAttribute("aria-valuenow", "264");
  await page.reload();
  await expect(page.locator("#status")).toContainText("Live C++");
  await expect(splitter).toHaveAttribute("aria-valuenow", "280");
  await expect(page.locator("#workers-splitter")).toHaveAttribute("aria-valuenow", "264");
  await splitter.dblclick();
  await expect(splitter).toHaveAttribute("aria-valuenow", "240");
  await page.locator("#dim").selectOption("1");
  await page.locator("#reset-layout").click();
  await expect(page.locator("#workers-splitter")).toHaveAttribute("aria-valuenow", "200");
  await expect(page.locator("#dim")).toHaveValue("1"); // Layout reset must not reset the mapping.
});

test("collapsing side panes gives space to indices without losing shared selection", async ({page}) => {
  await ready(page);
  const initial = (await page.locator("#domain-pane").boundingBox()).width;
  await page.locator('.pane-toggles [data-toggle-pane="config"]').click();
  await expect(page.locator("#config-pane")).toBeHidden();
  await expect(page.locator("#config-splitter")).toBeHidden();
  await page.locator('.pane-toggles [data-toggle-pane="workers"]').click();
  await expect(page.locator("#worker-pane")).toBeHidden();
  expect((await page.locator("#domain-pane").boundingBox()).width).toBe(initial + 460);
  await page.locator('[data-cell="3,6"]').click();
  await expect(page.locator("#worker-selection")).toContainText("T14");
  await expect(page.locator(".sum-equation")).toHaveAttribute("data-global", "3,6");
  await page.locator('.pane-toggles [data-toggle-pane="workers"]').click();
  await expect(page.locator('[data-worker="14"]')).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator('[data-worker="14"]')).toBeInViewport();
  await page.reload();
  await expect(page.locator("#status")).toContainText("Live C++");
  await expect(page.locator("#config-pane")).toBeHidden();
  await expect(page.locator("#worker-pane")).toBeVisible();
  await page.locator("#reset-layout").click();
  await expect(page.locator("#config-pane")).toBeVisible();
});

test("responsive constraints preserve preferred widths across desktop and mobile", async ({page}) => {
  await ready(page);
  for (const id of ["config-splitter", "workers-splitter"]) {
    await page.locator(`#${id}`).focus();
    await page.keyboard.press("End");
  }
  for (const width of [1024, 950, 901]) {
    await page.setViewportSize({width, height: 1000});
    await expect.poll(async () => (await page.locator("#domain-pane").boundingBox()).width).toBeGreaterThanOrEqual(320);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await page.setViewportSize({width: 900, height: 1000});
  await expect(page.locator("#config-pane")).toBeHidden();
  await expect(page.locator("#worker-pane")).toBeHidden();
  await page.setViewportSize({width: 2560, height: 1080});
  await expect(page.locator("#config-pane")).toBeVisible();
  await expect(page.locator("#config-splitter")).toHaveAttribute("aria-valuenow", "420");
  await expect(page.locator("#workers-splitter")).toHaveAttribute("aria-valuenow", "400");
});

test("disabled or corrupt local storage cannot prevent the explorer from loading", async ({page}) => {
  await page.addInitScript(() => {
    Object.defineProperty(Storage.prototype, "getItem", {value() { throw new Error("Storage blocked"); }});
    Object.defineProperty(Storage.prototype, "setItem", {value() { throw new Error("Storage blocked"); }});
  });
  await ready(page);
  await page.locator("#config-splitter").focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#config-splitter")).toHaveAttribute("aria-valuenow", "256");
  await assertDrawing(page, defaults());
});

test("playback-only timeline expands without shrinking the workspace or changing details", async ({page}) => {
  await ready(page);
  const height = (await page.locator("#workspace").boundingBox()).height;
  await expect(page.locator("#timeline-details")).toBeHidden();
  await expect(page.locator("[data-event-worker]")).toHaveCount(0);
  expect((await page.locator("#timeline-panel").boundingBox()).height).toBeLessThan(65);
  await expect(page.locator("#step-back")).toBeHidden();
  await expect(page.locator("#keep-history")).toBeHidden();
  await page.locator("#timeline-step").fill("1");
  await expandTimeline(page);
  await expect(page.locator("#timeline")).toBeVisible();
  expect((await page.locator("#timeline").boundingBox()).height).toBeGreaterThan(400);
  await expect(page.locator("#step-label")).toHaveText("2 / 2");
  expect((await page.locator("#workspace").boundingBox()).height).toBe(height);
  await expect(page.locator("#assignment-details")).not.toHaveAttribute("open", "");
  await page.locator("#expand-timeline").click();
  await expect(page.locator("#timeline-details")).toBeHidden();
  expect((await page.locator("#timeline-panel").boundingBox()).height).toBeLessThan(65);
  await page.locator("#assignment-details > summary").click();
  await expect(page.locator("#calculation")).toBeVisible();
  expect((await page.locator("#workspace").boundingBox()).height).toBe(height);
});

test("mobile touch supports cell selection, configuration, workers, and internal scrolling", async ({browser}) => {
  const context = await browser.newContext({baseURL: "http://127.0.0.1:8765", viewport: {width: 390, height: 844},
    isMobile: true, hasTouch: true});
  try {
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await ready(page);
    await expect(page.locator("#config-pane")).toBeHidden();
    await expect(page.locator("#worker-pane")).toBeHidden();
    await expect(page.locator("#data")).toBeInViewport();
    await page.locator('[data-cell="1,2"]').tap();
    await expect(page.locator(".sum-equation")).toHaveAttribute("data-global", "1,2");
    await expect(page.locator("#step-label")).toHaveText("2 / 2");
    await page.locator('.pane-toggles [data-toggle-pane="workers"]').tap();
    await expect(page.locator("#worker-pane")).toBeVisible();
    await page.locator('[data-worker="7"]').tap();
    await expect(page.locator("#worker-selection")).toContainText("T7");
    await page.locator('.pane-toggles [data-toggle-pane="config"]').tap();
    await expect(page.locator("#worker-pane")).toBeHidden();
    await expect(page.locator("#config-pane")).toBeVisible();
    await page.locator("#spec").selectOption("1");
    await expect(page.locator("#launch-summary")).toContainText("assumed ThreadSpec");
    await expect(page.locator("#status")).toContainText("NOT resolved");
    await page.locator('#config-pane [data-toggle-pane="config"]').tap();
    await expect(page.locator("#config-pane")).toBeHidden();
    await page.locator("#data").evaluate(data => { data.scrollLeft = data.scrollWidth; });
    await page.locator('[data-cell="3,7"]').tap();
    await expect(page.locator(".sum-equation")).toHaveAttribute("data-global", "3,7");
    const cell = await page.locator('[data-cell="3,7"] > rect').first().boundingBox();
    expect(cell.width).toBeGreaterThanOrEqual(44);
    const control = await page.locator('.pane-toggles [data-toggle-pane="config"]').boundingBox();
    expect(control.height).toBeGreaterThanOrEqual(44);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(errors).toEqual([]);
  } finally { await context.close(); }
});

for (const width of [320, 390, 768, 900]) test(`map-first layout fits a ${width}px screen`, async ({page}) => {
  await page.setViewportSize({width, height: 844});
  await ready(page);
  await expect(page.locator("#config-pane")).toBeHidden();
  await expect(page.locator("#worker-pane")).toBeHidden();
  expect((await page.locator("#data").boundingBox()).y).toBeLessThan(450);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('.pane-toggles [data-toggle-pane="config"]').click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("automatic height fills the desktop viewport and grows beyond the old cap", async ({page}) => {
  await ready(page);
  await expect(page.locator("#workspace")).toHaveAttribute("data-height-mode", "fit");
  await expect(page.locator("#fit-height")).toBeDisabled();
  expect((await page.locator("#workspace").boundingBox()).height).toBeGreaterThan(750);
  await expect(page.locator("#playback-bar")).toBeInViewport();
  await expect(page.locator("#assignment-details > summary")).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight + 1)).toBe(true);
  await page.setViewportSize({width: 1440, height: 2160});
  await expect.poll(async () => (await page.locator("#workspace").boundingBox()).height).toBeGreaterThan(1800);
  const heights = await Promise.all(["config-pane", "worker-pane", "domain-pane"].map(async id =>
    (await page.locator(`#${id}`).boundingBox()).height));
  expect(new Set(heights).size).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight + 1)).toBe(true);
});

test("height fitting responds to wrapped launch assumptions rather than fixed offsets", async ({page}) => {
  await page.setViewportSize({width: 1040, height: 1000});
  await ready(page);
  await page.locator("#spec").selectOption("1");
  await expect(page.locator("#status")).toContainText("NOT resolved");
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollHeight <= innerHeight + 1)).toBe(true);
  await expect(page.locator("#assignment-details > summary")).toBeInViewport();
});

test("workspace height resizes by keyboard, persists, and returns to automatic fit", async ({page}) => {
  await ready(page);
  const height = (await page.locator("#workspace").boundingBox()).height;
  await page.locator("#height-splitter").focus();
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Shift+ArrowDown");
  await expect(page.locator("#workspace")).toHaveAttribute("data-height-mode", "manual");
  expect((await page.locator("#workspace").boundingBox()).height).toBe(height + 80);
  await expect(page.locator("#fit-height")).toBeEnabled();
  await page.reload();
  await expect(page.locator("#status")).toContainText("Live C++");
  expect((await page.locator("#workspace").boundingBox()).height).toBe(height + 80);
  await page.setViewportSize({width: 1440, height: 1400});
  expect((await page.locator("#workspace").boundingBox()).height).toBe(height + 80);
  await page.locator("#fit-height").click();
  await expect(page.locator("#workspace")).toHaveAttribute("data-height-mode", "fit");
  await expect.poll(async () => (await page.locator("#workspace").boundingBox()).height).toBeGreaterThan(height + 250);
  await page.locator("#height-splitter").focus();
  await page.keyboard.press("Home");
  await expect(page.locator("#height-splitter")).toHaveAttribute("aria-valuenow", "480");
  await page.locator("#reset-layout").click();
  await expect(page.locator("#workspace")).toHaveAttribute("data-height-mode", "fit");
});

test("dragging grows all three panes together and double-click restores fit", async ({page}) => {
  await ready(page);
  const before = (await page.locator("#workspace").boundingBox()).height;
  const handle = await page.locator("#height-splitter").boundingBox();
  const x = handle.x + handle.width / 2, y = handle.y + handle.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x, y + 100, {steps: 5});
  await page.mouse.up();
  expect((await page.locator("#workspace").boundingBox()).height).toBe(before + 100);
  for (const id of ["config-pane", "worker-pane", "domain-pane"])
    expect((await page.locator(`#${id}`).boundingBox()).height).toBe(before + 100);
  await expect(page.locator("body")).not.toHaveClass(/resizing-height/);
  await page.locator("#height-splitter").dblclick();
  await expect(page.locator("#workspace")).toHaveAttribute("data-height-mode", "fit");
  expect((await page.locator("#workspace").boundingBox()).height).toBe(before);
});

test("composition collapses independently, retains overlays, and remembers the compact equation", async ({page}) => {
  await ready(page);
  await page.locator('[data-cell="3,6"]').click();
  await page.mouse.move(0, 0);
  const canvasHeight = (await page.locator("#data").boundingBox()).height;
  const workspaceHeight = (await page.locator("#workspace").boundingBox()).height;
  await page.locator("#composition > summary").click();
  await expect(page.locator("#composition")).not.toHaveAttribute("open", "");
  await expect(page.locator("#composition-summary")).toHaveText("(3, 6) = (2, 4) + (1, 2)");
  await expect(page.locator(".sum-equation")).toBeHidden();
  await expect(page.locator(".local-inset")).toBeHidden();
  await expect(page.locator(".composition-overlay path")).toHaveCount(1);
  expect((await page.locator("#data").boundingBox()).height).toBeGreaterThan(canvasHeight + 80);
  expect((await page.locator("#workspace").boundingBox()).height).toBe(workspaceHeight);
  await page.locator("#timeline-step").fill("0");
  await expect(page.locator("#composition-summary")).toHaveText("(2, 6) = (2, 4) + (0, 2)");
  await page.locator("#show-composition").uncheck();
  await expect(page.locator("#composition")).toBeHidden();
  await page.locator("#show-composition").check();
  await expect(page.locator("#composition")).not.toHaveAttribute("open", "");
  await page.reload();
  await expect(page.locator("#status")).toContainText("Live C++");
  await expect(page.locator("#composition")).not.toHaveAttribute("open", "");
  await expect(page.locator("#composition-summary")).toContainText("(0, 0) = (0, 0) + (0, 0)");
  await page.locator("#hierarchy").selectOption("0");
  await expect(page.locator("#composition")).toBeHidden();
  await page.locator("#hierarchy").selectOption("1");
  await expect(page.locator("#composition")).not.toHaveAttribute("open", "");
  await page.locator("#reset-layout").click();
  await expect(page.locator("#composition")).toHaveAttribute("open", "");
});

test("closed timeline rebuilds current events on expansion after playback or mapping changes", async ({page}) => {
  await ready(page);
  await page.locator("#timeline-step").fill("1");
  await expect(page.locator("#timeline")).toBeEmpty();
  await expandTimeline(page);
  await expect(page.locator('.timeline-event.current')).toHaveCount(16);
  await page.locator("#expand-timeline").click();
  await page.locator("#preset").selectOption("uneven");
  await page.locator("#timeline-step").fill("1");
  await expandTimeline(page);
  await expect(page.locator('.timeline-event.current')).toHaveCount(16);
  await expect(page.locator('.timeline-event.current.guarded')).not.toHaveCount(0);
  await expect(page.locator("#step-label")).toHaveText("2 / 2");
});

test("mobile ignores desktop height and supports collapsing composition by touch", async ({browser}) => {
  const context = await browser.newContext({baseURL: "http://127.0.0.1:8765", viewport: {width: 390, height: 844},
    isMobile: true, hasTouch: true});
  try {
    await context.addInitScript(() => localStorage.setItem("alpaka-index-lab.layout.v1",
      JSON.stringify({height: 1600, compositionCollapsed: true})));
    const page = await context.newPage();
    await ready(page);
    await expect(page.locator("#workspace-resizebar")).toBeHidden();
    await expect(page.locator("#timeline-details")).toBeHidden();
    await expect(page.locator("#composition")).not.toHaveAttribute("open", "");
    expect((await page.locator("#data").boundingBox()).height).toBeLessThan(501);
    await page.locator("#composition > summary").tap();
    await expect(page.locator(".sum-equation")).toBeVisible();
    await page.locator("#composition > summary").tap();
    await page.locator("#expand-timeline").tap();
    await expect(page.locator("#timeline")).toBeVisible();
    await page.locator("#expand-timeline").tap();
    await expect(page.locator("#timeline-details")).toBeHidden();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.setViewportSize({width: 1440, height: 1100});
    await expect(page.locator("#workspace-resizebar")).toBeVisible();
    await expect(page.locator("#workspace")).toHaveAttribute("data-height-mode", "manual");
    expect((await page.locator("#workspace").boundingBox()).height).toBe(1600);
  } finally { await context.close(); }
});

test("capture wide and mobile workspaces", async ({browser}, testInfo) => {
  for (const [name, viewport, touch] of [
    ["wide", {width: 2560, height: 1080}, false], ["mobile", {width: 390, height: 844}, true],
  ]) {
    const context = await browser.newContext({baseURL: "http://127.0.0.1:8765", viewport, isMobile: touch, hasTouch: touch});
    try {
      const page = await context.newPage();
      await ready(page);
      await page.screenshot({path: testInfo.outputPath(`${name}.png`), fullPage: true});
    } finally { await context.close(); }
  }
});

test("capture the standalone prototype", async ({page}, testInfo) => {
  await ready(page);
  await page.screenshot({path: testInfo.outputPath("explorer.png"), fullPage: true});
  await page.locator('[data-cell="3,6"]').click();
  await page.mouse.move(0, 0);
  await page.screenshot({path: testInfo.outputPath("composition-history.png"), fullPage: true});
  await page.locator("#composition > summary").click();
  await page.screenshot({path: testInfo.outputPath("collapsed-composition.png"), fullPage: true});
});

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

test("capture the standalone prototype", async ({page}, testInfo) => {
  await ready(page);
  await page.screenshot({path: testInfo.outputPath("explorer.png"), fullPage: true});
});

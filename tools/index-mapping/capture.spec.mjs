// SPDX-License-Identifier: MPL-2.0
import {ready, test} from "./browser-fixtures.mjs";

for (const [name, viewport] of [
  ["desktop", {width: 1440, height: 1100}], ["wide", {width: 2560, height: 1080}],
]) test(`capture ${name}`, async ({page}, testInfo) => {
  await page.setViewportSize(viewport);
  await ready(page);
  await page.screenshot({path: testInfo.outputPath("explorer.png"), fullPage: true});
  await page.locator('[data-cell="3,6"]').click();
  await page.mouse.move(0, 0);
  await page.screenshot({path: testInfo.outputPath("composition-history.png"), fullPage: true});
  await page.locator("#composition > summary").click();
  await page.screenshot({path: testInfo.outputPath("collapsed-composition.png"), fullPage: true});
});

test("capture mobile", async ({touchPage: page}, testInfo) => {
  await ready(page);
  await page.screenshot({path: testInfo.outputPath("explorer.png"), fullPage: true});
});

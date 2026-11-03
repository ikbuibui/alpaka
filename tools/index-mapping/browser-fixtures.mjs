// SPDX-License-Identifier: MPL-2.0
import {expect, test as base} from "@playwright/test";
export {expect};

async function checkPageErrors(page, use) {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await use(page);
  expect(errors, "Uncaught browser errors during the test").toEqual([]);
}

export const test = base.extend({
  page: async ({page}, use) => checkPageErrors(page, use),
  touchPage: async ({browser, baseURL}, use) => {
    const context = await browser.newContext({baseURL, viewport: {width: 390, height: 844},
      isMobile: true, hasTouch: true});
    try { await checkPageErrors(await context.newPage(), use); }
    finally { await context.close(); }
  },
});

export async function ready(page, hash = "") {
  await page.goto(`/${hash}`);
  await expect(page.locator("#status")).toContainText("Live C++");
}

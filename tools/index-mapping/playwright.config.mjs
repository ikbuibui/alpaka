// SPDX-License-Identifier: MPL-2.0
import {defineConfig} from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "browser.spec.mjs",
  outputDir: "build/browser-results",
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8765",
    viewport: {width: 1440, height: 1100},
    launchOptions: process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {},
  },
  webServer: {
    command: "python3 -m http.server 8765 --bind 127.0.0.1 --directory build/site",
    url: "http://127.0.0.1:8765",
    reuseExistingServer: false,
  },
});

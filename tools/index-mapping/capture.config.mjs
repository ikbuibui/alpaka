// SPDX-License-Identifier: MPL-2.0
import config from "./playwright.config.mjs";

// Artifact generation is opt-in; screenshots without baselines are not tests.
export default {...config, testMatch: "capture.spec.mjs", outputDir: "build/screenshots"};

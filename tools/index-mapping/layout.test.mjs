// SPDX-License-Identifier: MPL-2.0
import assert from "node:assert/strict";
import {test} from "node:test";
import {defaultLayout, fitPaneWidths, fitWorkspaceHeight, readLayout} from "./web/layout.mjs";

test("layout defaults favor the domain and do not share mutable preferences", () => {
  assert.deepEqual(defaultLayout(), {config: 240, workers: 200, height: null, compositionCollapsed: false, collapsed: {config: false, workers: false}});
  const changed = defaultLayout();
  changed.collapsed.config = true;
  assert.equal(defaultLayout().collapsed.config, false);
});

test("stored layout validates values and tolerates missing or corrupt storage", () => {
  for (const raw of [null, "", "null", "false", "[]", "invalid", '{"config":"999"}'])
    assert.deepEqual(readLayout(raw), defaultLayout());
  assert.deepEqual(readLayout('{"config":999,"workers":1,"collapsed":{"config":true,"workers":"false"}}'),
    {...defaultLayout(), config: 420, workers: 140, collapsed: {config: true, workers: false}});
  const saved = {...defaultLayout(), config: 300, workers: 280, collapsed: {config: false, workers: true}};
  assert.deepEqual(readLayout(JSON.stringify(saved)), saved);
});

test("saved height and composition preferences migrate old layouts and validate new fields", () => {
  const old = {config: 300, workers: 180, collapsed: {config: false, workers: true}};
  assert.deepEqual(readLayout(JSON.stringify(old)), {...defaultLayout(), ...old});
  assert.deepEqual(readLayout('{"height":1200,"compositionCollapsed":true}'),
    {...defaultLayout(), height: 1200, compositionCollapsed: true});
  for (const height of [null, "1000", false])
    assert.equal(readLayout(JSON.stringify({height})).height, null);
  assert.equal(readLayout('{"height":-1}').height, 480);
  assert.equal(readLayout('{"height":999999}').height, 10000);
  assert.equal(readLayout('{"compositionCollapsed":"true"}').compositionCollapsed, false);
});

test("automatic height fits collapsed chrome without the former 850px cap", () => {
  assert.equal(fitWorkspaceHeight(1100, 130, 150), 820);
  assert.equal(fitWorkspaceHeight(2160, 130, 150), 1880);
  assert.equal(fitWorkspaceHeight(600, 200, 150), 480);
  assert.equal(fitWorkspaceHeight(1100, 130.4, 150.3), 819);
});

test("pane constraints leave a usable data domain at every desktop viewport", () => {
  for (const available of [869, 900, 1024, 1400, 2560]) for (const config of [180, 240, 420])
    for (const workers of [140, 200, 400]) for (const hideConfig of [false, true]) for (const hideWorkers of [false, true]) {
      const state = {config, workers, collapsed: {config: hideConfig, workers: hideWorkers}};
      const widths = fitPaneWidths(state, available);
      assert(widths.config >= 180 && widths.config <= 420);
      assert(widths.workers >= 140 && widths.workers <= 400);
      const used = (hideConfig ? 0 : widths.config + 10) + (hideWorkers ? 0 : widths.workers + 10);
      assert(available - used >= 320);
      assert.equal(state.config, config, "resizing the viewport must not overwrite saved width preferences");
      assert.equal(state.workers, workers);
    }
});

test("collapsing panes releases space without forgetting their preferred widths", () => {
  const state = {config: 420, workers: 400, collapsed: {config: true, workers: false}};
  assert.deepEqual(fitPaneWidths(state, 900), {config: 420, workers: 400});
  state.collapsed.config = false;
  assert.deepEqual(fitPaneWidths(state, 900), {config: 180, workers: 380});
});

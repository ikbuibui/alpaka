// SPDX-License-Identifier: MPL-2.0
// Workspace preferences are local presentation state, separate from shared mapping configurations.
export const layoutStorageKey = "alpaka-index-lab.layout.v1";
const bounds = {config: [180, 420], workers: [140, 400]};
const heightBounds = [480, 10000];
const clamp = (n, min, max) => Math.min(max, Math.max(min, n));
export const defaultLayout = () => ({config: 240, workers: 200, height: null,
  compositionCollapsed: false, showComposition: true, collapsed: {config: false, workers: false}});

export function fitWorkspaceHeight(viewportHeight, workspaceTop, reservedHeight) {
  return Math.max(heightBounds[0], Math.floor(viewportHeight - workspaceTop - reservedHeight));
}

export function readLayout(raw) {
  const state = defaultLayout();
  try {
    const saved = JSON.parse(raw);
    if (Number.isFinite(saved?.height)) state.height = clamp(saved.height, ...heightBounds);
    if (typeof saved?.compositionCollapsed === "boolean") state.compositionCollapsed = saved.compositionCollapsed;
    if (typeof saved?.showComposition === "boolean") state.showComposition = saved.showComposition;
    for (const key of Object.keys(bounds)) {
      if (Number.isFinite(saved?.[key])) state[key] = clamp(saved[key], ...bounds[key]);
      if (typeof saved?.collapsed?.[key] === "boolean") state.collapsed[key] = saved.collapsed[key];
    }
  } catch { /* Corrupt or unavailable preferences must not prevent exploring indices. */ }
  return state;
}

export function fitPaneWidths(state, available) {
  const widths = Object.fromEntries(Object.entries(bounds).map(([key, range]) => [key, clamp(state[key], ...range)]));
  const open = Object.keys(bounds).filter(key => !state.collapsed[key]);
  // Reserve 320px for data and 10px for each visible separator.
  let excess = open.reduce((sum, key) => sum + widths[key], 0) + open.length * 10 + 320 - available;
  for (const key of open) {
    const reduction = Math.min(Math.max(0, excess), widths[key] - bounds[key][0]);
    widths[key] -= reduction;
    excess -= reduction;
  }
  return widths;
}

// Regions opt in to automatic height reservation. Optional chrome markers identify
// the collapsed footprint; expanding content must not reduce the workspace height.
function reservedHeight(region) {
  const style = getComputedStyle(region);
  const chrome = [...region.querySelectorAll("[data-fit-chrome]")];
  const frame = ["marginTop", "marginBottom", "borderTopWidth", "borderBottomWidth", "paddingTop", "paddingBottom"]
    .reduce((sum, key) => sum + parseFloat(style[key]), 0);
  // Multiple chrome markers share a row, as in the footer.
  return chrome.length ? Math.max(...chrome.map(element => element.getBoundingClientRect().height)) + frame
    : region.getBoundingClientRect().height + parseFloat(style.marginTop) + parseFloat(style.marginBottom);
}

export function initializeLayout(onReveal = () => {}, onCompositionVisibility = () => {}) {
  const $ = id => document.getElementById(id);
  const workspace = $("workspace"), mobile = matchMedia("(max-width: 900px)");
  const regions = [...document.querySelectorAll("[data-fit-region]")];
  let state;
  try { state = readLayout(localStorage.getItem(layoutStorageKey)); } catch { state = defaultLayout(); }
  const mobileOpen = {config: false, workers: false};
  let drag = null;
  const isCollapsed = key => mobile.matches ? !mobileOpen[key] : state.collapsed[key];
  const persist = () => {
    try { localStorage.setItem(layoutStorageKey, JSON.stringify(state)); } catch { /* Storage can be disabled. */ }
  };
  const heightSeparator = $("height-splitter"), composition = $("composition");
  const compositionCheckbox = $("show-composition");
  function restoreComposition() {
    compositionCheckbox.checked = state.showComposition;
    onCompositionVisibility(state.showComposition);
    composition.open = !state.compositionCollapsed;
  }
  restoreComposition();
  compositionCheckbox.addEventListener("change", () => {
    state.showComposition = compositionCheckbox.checked;
    persist();
  });
  composition.addEventListener("toggle", () => {
    state.compositionCollapsed = !composition.open;
    persist();
  });
  function applyHeight() {
    $("workspace-resizebar").hidden = mobile.matches;
    if (mobile.matches) return; // Keep natural scrolling and the saved desktop preference.
    const reserved = regions.reduce((sum, region) => sum + reservedHeight(region), 0)
      + parseFloat(getComputedStyle(workspace.parentElement).paddingBottom);
    const top = workspace.getBoundingClientRect().top + window.scrollY;
    const height = state.height ?? fitWorkspaceHeight(window.innerHeight, top, reserved);
    workspace.style.setProperty("--workspace-height", `${height}px`);
    workspace.dataset.heightMode = state.height === null ? "fit" : "manual";
    heightSeparator.setAttribute("aria-valuenow", Math.round(height));
    heightSeparator.setAttribute("aria-valuemax", Math.max(heightBounds[1], height));
    heightSeparator.setAttribute("aria-valuetext", `${Math.round(height)} pixels, ${state.height === null ? "automatic fit" : "manual height"}`);
    $("fit-height").disabled = state.height === null;
  }
  function apply() {
    const available = mobile.matches ? 900 : workspace.clientWidth;
    const widths = fitPaneWidths(state, available);
    workspace.style.setProperty("--config-width", `${widths.config}px`);
    workspace.style.setProperty("--worker-width", `${widths.workers}px`);
    for (const key of Object.keys(bounds)) {
      const collapsed = isCollapsed(key);
      workspace.dataset[`${key}Collapsed`] = collapsed;
      $(key === "config" ? "config-pane" : "worker-pane").hidden = collapsed;
      const separator = $(`${key}-splitter`);
      separator.hidden = mobile.matches || collapsed;
      separator.setAttribute("aria-valuenow", Math.round(widths[key]));
      const other = key === "config" ? "workers" : "config";
      const max = Math.max(bounds[key][0], Math.min(bounds[key][1], available - 320
        - (isCollapsed(other) ? 10 : widths[other] + 20)));
      separator.setAttribute("aria-valuemax", Math.floor(max));
      for (const button of document.querySelectorAll(`[data-toggle-pane="${key}"]`))
        button.setAttribute("aria-expanded", String(!collapsed));
    }
    applyHeight();
  }
  function endDrag() {
    if (!drag) return;
    const {separator, pointerId} = drag;
    drag = null;
    if (separator.hasPointerCapture(pointerId)) separator.releasePointerCapture(pointerId);
    document.body.classList.remove("resizing", "resizing-height");
    persist();
  }
  function bindResize(separator, {axis, min, change, reset}) {
    const coordinate = axis === "x" ? "clientX" : "clientY";
    const negative = axis === "x" ? "ArrowLeft" : "ArrowUp";
    const positive = axis === "x" ? "ArrowRight" : "ArrowDown";
    const resize = value => change(clamp(value, min, Number(separator.getAttribute("aria-valuemax"))));
    separator.addEventListener("pointerdown", event => {
      if (drag || mobile.matches || event.button !== 0) return;
      event.preventDefault();
      separator.focus();
      drag = {separator, pointerId: event.pointerId, start: event[coordinate],
        value: Number(separator.getAttribute("aria-valuenow"))};
      separator.setPointerCapture(event.pointerId);
      document.body.classList.add(axis === "x" ? "resizing" : "resizing-height");
    });
    separator.addEventListener("pointermove", event => {
      if (drag?.separator === separator && drag.pointerId === event.pointerId)
        resize(drag.value + event[coordinate] - drag.start);
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) separator.addEventListener(name, endDrag);
    separator.addEventListener("keydown", event => {
      if (![negative, positive, "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const value = Number(separator.getAttribute("aria-valuenow"));
      resize(event.key === "Home" ? min : event.key === "End" ? Number(separator.getAttribute("aria-valuemax"))
        : value + (event.key === negative ? -1 : 1) * (event.shiftKey ? 64 : 16));
      persist();
    });
    separator.addEventListener("dblclick", reset);
  }
  for (const button of document.querySelectorAll("[data-toggle-pane]")) button.addEventListener("click", () => {
    endDrag();
    const key = button.dataset.togglePane;
    if (mobile.matches) {
      const open = !mobileOpen[key];
      mobileOpen.config = mobileOpen.workers = false;
      mobileOpen[key] = open;
    } else {
      state.collapsed[key] = !state.collapsed[key];
      persist();
    }
    apply();
    document.querySelector(`.pane-toggles [data-toggle-pane="${key}"]`).focus({preventScroll: true});
    onReveal();
  });
  for (const separator of document.querySelectorAll("[data-resize-pane]")) {
    const key = separator.dataset.resizePane;
    bindResize(separator, {axis: "x", min: bounds[key][0],
      change: value => { state[key] = value; apply(); },
      reset: () => { endDrag(); state[key] = defaultLayout()[key]; apply(); persist(); }});
  }
  function fitHeight() {
    endDrag();
    state.height = null;
    persist();
    applyHeight();
  }
  bindResize(heightSeparator, {axis: "y", min: heightBounds[0],
    change: value => { state.height = value; applyHeight(); }, reset: fitHeight});
  $("fit-height").addEventListener("click", fitHeight);
  $("reset-layout").addEventListener("click", () => {
    endDrag();
    state = defaultLayout();
    restoreComposition();
    mobileOpen.config = mobileOpen.workers = false;
    persist();
    apply();
    onReveal();
  });
  const responsive = () => { endDrag(); apply(); };
  mobile.addEventListener("change", responsive);
  window.addEventListener("resize", responsive);
  let pendingHeight = 0;
  const heightObserver = new ResizeObserver(() => {
    if (!pendingHeight) pendingHeight = requestAnimationFrame(() => { pendingHeight = 0; applyHeight(); });
  });
  for (const element of document.querySelectorAll("[data-fit-region], [data-fit-chrome], [data-fit-above]"))
    heightObserver.observe(element);
  let previousWidth = workspace.clientWidth;
  new ResizeObserver(() => {
    if (workspace.clientWidth !== previousWidth) { previousWidth = workspace.clientWidth; apply(); }
  }).observe(workspace);
  apply();
}

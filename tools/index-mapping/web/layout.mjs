// SPDX-License-Identifier: MPL-2.0
// Workspace preferences are local presentation state, separate from shared mapping configurations.
export const layoutStorageKey = "alpaka-index-lab.layout.v1";
const bounds = {config: [180, 420], workers: [140, 400]};
const heightBounds = [480, 10000];
const clamp = (n, min, max) => Math.min(max, Math.max(min, n));
export const defaultLayout = () => ({config: 240, workers: 200, height: null,
  compositionCollapsed: false, collapsed: {config: false, workers: false}});

export function fitWorkspaceHeight(viewportHeight, workspaceTop, reservedHeight) {
  return Math.max(heightBounds[0], Math.floor(viewportHeight - workspaceTop - reservedHeight));
}

export function readLayout(raw) {
  const state = defaultLayout();
  try {
    const saved = JSON.parse(raw);
    if (Number.isFinite(saved?.height)) state.height = clamp(saved.height, ...heightBounds);
    if (typeof saved?.compositionCollapsed === "boolean") state.compositionCollapsed = saved.compositionCollapsed;
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

export function initializeLayout(onReveal = () => {}) {
  const $ = id => document.getElementById(id);
  const workspace = $("workspace");
  const mobile = matchMedia("(max-width: 900px)");
  let state;
  try { state = readLayout(localStorage.getItem(layoutStorageKey)); } catch { state = defaultLayout(); }
  const mobileOpen = {config: false, workers: false};
  let drag = null;
  const isCollapsed = key => mobile.matches ? !mobileOpen[key] : state.collapsed[key];
  const persist = () => {
    try { localStorage.setItem(layoutStorageKey, JSON.stringify(state)); } catch { /* Storage can be disabled. */ }
  };
  const heightSeparator = $("height-splitter");
  const composition = $("composition");
  composition.open = !state.compositionCollapsed;
  composition.addEventListener("toggle", () => {
    state.compositionCollapsed = !composition.open;
    persist();
  });
  const outerHeight = element => {
    const style = getComputedStyle(element);
    return element.getBoundingClientRect().height + parseFloat(style.marginTop) + parseFloat(style.marginBottom);
  };
  function applyHeight() {
    $("workspace-resizebar").hidden = mobile.matches;
    if (mobile.matches) return; // Keep natural page scrolling and the saved desktop preference.
    const timeline = $("timeline-panel"), assignment = $("assignment-details");
    const borders = element => {
      const style = getComputedStyle(element);
      return parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth);
    };
    const margins = element => {
      const style = getComputedStyle(element);
      return parseFloat(style.marginTop) + parseFloat(style.marginBottom);
    };
    // Only reserve collapsed chrome. Expanded timeline/details must not shrink the main view.
    const reserved = outerHeight($("workspace-resizebar")) + outerHeight($("playback-bar")) + borders(timeline) + margins(timeline)
      + outerHeight(assignment.querySelector(":scope > summary")) + borders(assignment) + margins(assignment)
      + Math.max(outerHeight(document.querySelector("footer > span")), outerHeight(document.querySelector("footer > details > summary")))
      + margins(document.querySelector("footer")) + parseFloat(getComputedStyle(document.querySelector("main")).paddingBottom);
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
    // Hiding a pane must not leave keyboard focus inside hidden content.
    document.querySelector(`.pane-toggles [data-toggle-pane="${key}"]`).focus({preventScroll: true});
    onReveal();
  });
  for (const separator of document.querySelectorAll("[data-resize-pane]")) {
    const key = separator.dataset.resizePane;
    const resize = value => {
      state[key] = clamp(value, bounds[key][0], Number(separator.getAttribute("aria-valuemax")));
      apply();
    };
    separator.addEventListener("pointerdown", event => {
      if (drag || mobile.matches || event.button !== 0) return;
      event.preventDefault();
      separator.focus();
      drag = {separator, pointerId: event.pointerId, start: event.clientX,
        width: Number(separator.getAttribute("aria-valuenow"))};
      separator.setPointerCapture(event.pointerId);
      document.body.classList.add("resizing");
    });
    separator.addEventListener("pointermove", event => {
      if (drag?.separator === separator && drag.pointerId === event.pointerId)
        resize(drag.width + event.clientX - drag.start);
    });
    separator.addEventListener("pointerup", endDrag);
    separator.addEventListener("pointercancel", endDrag);
    separator.addEventListener("lostpointercapture", endDrag);
    separator.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const width = Number(separator.getAttribute("aria-valuenow"));
      resize(event.key === "Home" ? bounds[key][0] : event.key === "End"
        ? Number(separator.getAttribute("aria-valuemax")) : width + (event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 64 : 16));
      persist();
    });
    separator.addEventListener("dblclick", () => { resize(defaultLayout()[key]); persist(); });
  }
  function resizeHeight(value) {
    state.height = clamp(value, heightBounds[0], Number(heightSeparator.getAttribute("aria-valuemax")));
    applyHeight();
  }
  function fitHeight() {
    endDrag();
    state.height = null;
    persist();
    applyHeight();
  }
  heightSeparator.addEventListener("pointerdown", event => {
    if (drag || mobile.matches || event.button !== 0) return;
    event.preventDefault();
    heightSeparator.focus();
    drag = {separator: heightSeparator, pointerId: event.pointerId, start: event.clientY,
      height: workspace.getBoundingClientRect().height};
    heightSeparator.setPointerCapture(event.pointerId);
    document.body.classList.add("resizing-height");
  });
  heightSeparator.addEventListener("pointermove", event => {
    if (drag?.separator === heightSeparator && drag.pointerId === event.pointerId)
      resizeHeight(drag.height + event.clientY - drag.start);
  });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) heightSeparator.addEventListener(name, endDrag);
  heightSeparator.addEventListener("keydown", event => {
    if (!["ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const height = workspace.getBoundingClientRect().height;
    resizeHeight(event.key === "Home" ? heightBounds[0] : event.key === "End" ? heightBounds[1]
      : height + (event.key === "ArrowUp" ? -1 : 1) * (event.shiftKey ? 64 : 16));
    persist();
  });
  heightSeparator.addEventListener("dblclick", fitHeight);
  $("fit-height").addEventListener("click", fitHeight);
  $("reset-layout").addEventListener("click", () => {
    endDrag();
    state = defaultLayout();
    composition.open = true;
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
  for (const selector of [".masthead", "#workspace-toolbar", "#status", "#workspace-resizebar", "#playback-bar",
    "#assignment-details > summary", "footer > span", "footer > details > summary"])
    heightObserver.observe(document.querySelector(selector));
  let previousWidth = workspace.clientWidth;
  new ResizeObserver(() => {
    if (workspace.clientWidth !== previousWidth) { previousWidth = workspace.clientWidth; apply(); }
  }).observe(workspace);
  apply();
}

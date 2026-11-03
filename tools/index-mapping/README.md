# Standalone index-mapping explorer

A 1D/2D prototype, independent of Sphinx, Read the Docs, and the main CMake build.
The browser executes alpaka's **actual C++ `makeIdxMap` implementation**, compiled
from this checkout to WebAssembly. JavaScript only configures and presents traces.

## Build and try

Requirements: a C++20 compiler, Python 3.10+, Node.js 20+, and Emscripten.
Emscripten **4.0.19** is the tested SDK version. No CUDA/HIP/SYCL toolkit or GPU is
required. No npm dependencies are required to build or view the explorer.

If needed, install Emscripten **outside the repository**:

```sh
git clone --depth 1 --branch 4.0.19 https://github.com/emscripten-core/emsdk.git /tmp/emsdk
/tmp/emsdk/emsdk install 4.0.19
/tmp/emsdk/emsdk activate 4.0.19
source /tmp/emsdk/emsdk_env.sh
```

From the alpaka repository root:

```sh
python3 tools/index-mapping/build.py
node tools/index-mapping/test.mjs
python3 -m http.server 8000 --bind 127.0.0.1 --directory tools/index-mapping/build/site
```

Open **http://127.0.0.1:8000**. Use HTTP, not `file://`, to load the WASM module.
Once built, everything runs locally; no requests go to Godbolt or another service.
Copy the configuration link to reproduce a configuration on the same build.

Options:

```sh
# Explicit compiler paths (also configurable through CXX and EMXX):
python3 tools/index-mapping/build.py --cxx g++ --emxx /path/to/em++

# Also build a native single-header reference and export an HTTP-include Godbolt validator:
python3 tools/index-mapping/build.py --single-header
node tools/index-mapping/test.mjs

# Use a particular published single header (for example a PR documentation build):
python3 tools/index-mapping/build.py --single-header --godbolt-header-url https://example.org/alpaka.hpp

# Also fetch that URL and compile/test the header Godbolt will actually import:
python3 tools/index-mapping/build.py --single-header --validate-http-header
node tools/index-mapping/test.mjs

# Work on native adapter tests without Emscripten (no usable browser engine):
python3 tools/index-mapping/build.py --native-only
node tools/index-mapping/test.mjs
```

`--single-header` generates a fresh header using `script/create-single-header.sh`;
that script needs network access to its amalgamation tool. No header is vendored.
The full header pulls in host system APIs unavailable in WASM, so the WASM build
uses a smaller set of regular headers. The conformance tests compare both routes.
Standard-library SIMD, TBB, and hwloc are disabled for this scalar mapping adapter.
Emscripten currently warns about unused 32-bit `size_t` hashing in `UniqueId.hpp`;
the adapter does not instantiate or call `uniqueId`.

## What works

- Explicit `ThreadSpec` extents, 1D and 2D.
- `FrameSpec` inputs **with a separately supplied, explicitly assumed ThreadSpec**.
- Host/CPU, CUDA/NVIDIA, HIP/AMD and oneAPI/Intel-GPU API/device tags.
- Compatible executor selection; scalar host thread-count validation.
- Direct ranges and two nested block → tile → element mappings.
- Named grid/block/thread worker groups, `allThreads`, and 1D linearized groups.
- Default/flat/tiled traversal and optimized/strided/contiguous layout, separately
  configurable at each hierarchy level.
- Nonzero range beginnings, strides in direct mode, partial tiles, bounds guards.
- Linked worker/data views, per-level iterations, ownership and coverage,
  effective policies, C++ snippets, and configuration links.
- In-map hierarchical index composition, a zero-based local-tile inset in 2D,
  and optional origin-plus-local cell labels.
- An all-block logical iteration timeline with shared selection, scrubbing,
  playback, and persistent desaturated ownership history.

For hierarchical mappings, the outer range stride is the tile extent and the
inner stride is one. Both are visible in the UI; the direct-mode stride is disabled.
Linearized worker groups need a **1D range** in the current adapter. They are not
silently mapped into a 2D range.

## Workspace layout

The worker hierarchy and index domain are the main view. The page uses the available
screen width with small fixed gutters (16px on desktop, 8px on smaller screens),
not an article-style maximum width. The grid preserves its aspect ratio rather than
stretching coordinates into wide rectangles.

- **Desktop, above 900px:** configuration, workers, and data are three adjacent panes.
  Configuration starts at 240px, workers at 200px; the domain receives the remaining
  width. The first two panes scroll independently. Drag their separators to resize,
  or focus a separator and use Left/Right (Shift for larger steps), Home, or End.
  Double-click resets that pane's width. Constraints reserve at least 320px for data.
- The desktop workspace automatically fills the remaining window height, reserving
  only the playback strip, collapsed assignment-details header, and footer. It has
  no fixed 850px cap. Drag the horizontal separator below the panes to adjust their
  shared height, or use Up/Down (Shift for larger steps), Home, or End. Manual heights
  are saved locally. **Fit height** or double-clicking the separator restores automatic
  fitting. A 480px minimum keeps short windows usable; larger manual heights scroll
  the page. Expanded timeline/details content never shrinks the main workspace.
  In short or narrow desktop panes, the domain panel scrolls vertically to keep
  at least 200px of canvas available alongside the composition and controls.
- **Configure / Workers** toggle the corresponding pane. Collapsing either gives its
  space to the data domain; the selected worker's identity stays visible in the
  domain header. Widths, height mode, and desktop visibility are saved locally when storage is
  available. **Reset layout** restores defaults without changing the mapping.
  Shrinking the viewport does not overwrite preferred desktop widths.
- **Phones and tablets, up to 900px:** the grid is shown first, with configuration
  and workers initially collapsed. Their toolbar buttons open inline panels above
  the grid, one at a time. Touch controls and cells have larger targets; tap to select
  and swipe inside the grid for large domains. Hover previews are only enabled on
  hover-capable devices. Wide grids and timelines scroll internally, not the page
  horizontally. Selecting an assignment pans its data cell into view without moving
  the whole page. Desktop height preferences and the horizontal resize handle are
  not applied on mobile; the page retains natural vertical scrolling.
- The toolbar keeps examples, dimensions, and launch assumptions visible. Detailed
  settings remain in the configuration pane. The timeline starts as a playback-only
  strip: **Play/Pause**, the logical-step slider/counter, and **Expand timeline**.
  Expanding reveals the worker rows, previous/next buttons, history/filter controls,
  and interpretation notes below the strip; collapsing hides them without changing
  the logical cursor or canvas height. Timeline rows are rendered only when expanded.
  Calculation, snippets, assignment tables, and diagnostics live in the initially
  collapsed **Assignment details** section. Only its visible tab is rendered; unchanged
  snippets, diagnostics, and all-assignment tables are reused while stepping.
  Build provenance is also collapsible.

Layout preferences are separate from shared C++ mapping configurations. Storage
failures or malformed saved preferences fall back safely; the explorer still works.

## Reading the linked visualizations

**Composition stays on the data map.** With **Composition** enabled, selecting
an assignment outlines its tile in the block color, marks the tile origin, and
routes a local-offset arrow along cell corners to the selected index. The equation
strip shows `global = tile origin + local index` using the actual C++ trace values.
These are the values returned by the two loops, **not raw block and thread IDs**.
In 2D, a zero-based tile inset and row/column displacement explain the local index.
Choose **Labels → Origin + local** to show summands in cells (two lines in 2D).
Hover a worker or data cell to preview; click to pin the assignment. Composition
and sum labels are disabled for direct mappings, which have no outer tile loop.

The **Index composition** disclosure independently collapses the detailed equation,
explanation, and local inset to a compact equation row. Its locally remembered state
survives selection, playback, and configuration changes. Collapsing frees canvas
height while leaving the tile-origin marker and offset arrow enabled; the separate
**Composition** checkbox still switches the whole composition visualization off.
**Reset layout** restores expanded composition and automatic workspace height.

**The logical iteration timeline is not a device scheduler.** Each column aligns
one `(outer iteration, inner iteration)` pair across all workers, ordered outer
first, then inner. Direct mappings have only inner iterations. Blocks and threads
are not asserted to execute together: a column is a logical iteration cohort,
not hardware lockstep, elapsed time, or a barrier. Missing assignments remain `—`;
a worker's next outer iteration is never packed into an exhausted inner-loop gap.

- Block headers, tile outlines, and worker buttons use ten fixed D3 Category10
  (classic Tableau) color families, intentionally cycling after ten blocks instead
  of generating near-colliding hues. Thread lightness retains its four-step ramp.
  This is not a guarantee of ten distinguishable colors for every form of color
  vision: explicit block/thread IDs and selection remain necessary, especially
  when families repeat.
- Current assignments are vivid. Earlier assignments retain fixed, desaturated
  ownership colors; future map cells are neutral. **Keep history** controls past
  colors, without deleting trace records. Scrubbing backward recomputes history
  from the cursor rather than leaving stale colors behind.
- Use the slider, step buttons, or column headers to move every worker to the same
  logical step. Click an event to select its worker, index, and composition.
  Worker selection never filters other workers' current assignments out of the map.
  **All assignments of selected worker** controls only that worker's outline overlay
  (enabled initially, including past, current, and future indices). A dark ring with
  a white halo identifies selected-thread cells without changing their temporal
  fills. A small triangle identifies the pinned assignment; a dashed inset ring is
  only a hover preview and never replaces selection. Worker buttons and timeline
  rows carry matching dark emphasis and an explicit **Selected** label. The domain
  header keeps the selected identity visible and reports idle steps.
- When the timeline is expanded, all blocks are shown initially, with collapsible groups and an optional
  **Selected block** timeline filter. The logical cursor remains global when filtered.
  Timelines show at most 24 columns around the cursor; the slider reaches every step.
- Guard-rejected events are crossed out and remain selectable, but never color
  processed data cells. Repeated assignments retain a count and ownership stripes
  (up to eight recent visits); click a map cell repeatedly to inspect its visits.
  The timeline and Assignments table retain all events, not just the last owner.
- Playback advances at a fixed presentation cadence, **not simulated execution time**.
  It stops at the end, on manual stepping/selection, configuration changes, or when
  the page is hidden. Autoplay is disabled for reduced-motion preferences; manual
  stepping and scrubbing remain available.

## Accuracy boundaries

**This is an index-iteration explorer, not a backend/device emulator.**

`engine.cpp` constructs real `onAcc::Acc` objects containing alpaka's generic
index layers and real API/device tags. Alpaka computes worker identities, index
sequences, default policy selection, and linear offsets. The adapter also
composes the two nested ranges and records a bounds guard. These composition
rules describe the selected kernel preset, not an implicit feature of `FrameSpec`.

The browser never launches native GPU kernels. Its executor selection validates
compatibility but is not a launch of that executor. The separate validator below
does launch real kernels. No timing, occupancy, synchronization or race
analysis is provided. Iteration numbers are not a hardware scheduling order.

**FrameSpec auto-resolution is deliberately not implemented.** Frame count and
frame extent describe logical capacity; neither guarantees a block/thread shape.
The user supplies the assumed worker extents, and the UI labels this prominently.
Frame count is not an implicit bound on `makeIdxMap`: the explicit logical range
still determines the work. Different launch shapes can cover the same range.

Not yet supported: warp/lane groups, boundary overloads, custom worker groups,
selectable iterator dimensions, arbitrary hierarchy trees, or device-dependent
launch-resolution heuristics. C++ snippets are illustrative fragments, not full
runnable Godbolt programs. Intermediate iterator arithmetic is not yet traced.

## Real CPU/CUDA buffer validation and Godbolt

`validate_backend.cpp` launches **real alpaka kernels**, not synthetic worker
contexts. Each worker writes its UID, block/thread coordinates, iteration numbers,
local/global indices and bounds results to its **own trace region**. Counts and
unwritten guard records detect missing workers and trace overflow. CUDA writes
an allocated device buffer; `onHost::memcpy` copies it into a host buffer and
`onHost::wait` completes before any host validation reads it.

Separate trace regions are important: an element-only owner buffer could hide
duplicate assignments through last-writer-wins overwrites. The host checks ordered
per-worker traces against the explorer adapter and checks coverage independently.
`--inject-error` deliberately corrupts a returned record; the tests require this
to fail, proving the validator actually inspects the copied-back data.

The suite has **31 launches per backend** across 1D/2D direct and hierarchical
mapping, default and explicit policies, nonzero starts, strides, partial tiles,
empty ranges, idle workers, repeated worker identities and duplicate ownership.
CPU uses `cpuSerial` (one thread per block); CUDA uses `gpuCuda` with multiple
threads per block. This tests `ThreadSpec`, **not FrameSpec auto-resolution**.
It is a preset suite, not an export of the widget's currently selected configuration.

```sh
# The regular build always compiles the CPU validator:
./tools/index-mapping/build/validate-cpu --cpu

# Optional CUDA compile (also compiles a locally resolved Godbolt export for CUDA):
python3 tools/index-mapping/build.py --single-header --cuda --cxx g++ --nvcc /path/to/nvcc
./tools/index-mapping/build/validate-cuda --cuda

# On a GPU runner, make unavailable CUDA hardware a test failure rather than a skip:
ALPAKA_REQUIRE_CUDA=1 node tools/index-mapping/test.mjs
```

Exit codes: **0 = passed, 1 = validation/runtime failure, 2 = bad arguments,
77 = backend unavailable (NOT validated)**. A CUDA compiler alone does not prove
GPU execution. Without a GPU, `node test.mjs` explicitly reports the CUDA check as
not validated; it still verifies the CUDA-compiled program's CPU path.

With `--single-header`, open **Download CPU/CUDA validator** in the explorer or use
`tools/index-mapping/build/godbolt-validation.cpp`. It contains the explorer adapter
and real-launch validator, and imports alpaka using Compiler Explorer's HTTP include:

```cpp
#include <https://alpaka.readthedocs.io/en/latest/alpaka.hpp>
```

The header is **not embedded**. Set `--godbolt-header-url` to a public single-header
URL for a specific version or PR documentation build. The explorer and normal
native tests still use this checkout; **the Godbolt program uses the published
URL, which can differ from this checkout and cannot include uncommitted changes**.
Prefer a versioned URL for reproducibility. Paste/open the downloaded source in
Compiler Explorer:

| Target | Compiler | Flags |
| --- | --- | --- |
| CPU | GCC or Clang with C++20 | `-std=c++20 -O1 -pthread` |
| CUDA | NVCC with C++20 | `-x cu -std=c++20 -O1 --expt-relaxed-constexpr --extended-lambda` |

Add a program executor in Godbolt. The default is CPU for GCC/Clang and CUDA for
NVCC; program arguments `--cpu` or `--cuda` select explicitly, and `--json` emits
machine-readable copied-back traces. HTTP includes are Compiler Explorer preprocessing,
not a standard C++ compiler feature; native compilers need the include resolved first. CUDA execution needs a GPU-enabled executor;
an assembly/compiler pane alone is only a compile check. An unavailable GPU must
not be interpreted as a pass. Edit the dimensions near `runCase` in the exported
source to try additional cases. Keep traces within the stated prototype limits.

For local testing, the build resolves the HTTP include in a separate
`build/godbolt-validation-local.cpp` using this checkout's freshly generated
single header. This checks the export body against the tree, **not the URL**.
To additionally check the actual published header:

```sh
python3 tools/index-mapping/build.py --single-header --validate-http-header
# Add --cuda --nvcc /path/to/nvcc for CUDA compilation too.
node tools/index-mapping/test.mjs
```

This fetches the configured URL, records its SHA-256 in the manifest, builds
`build/godbolt-validation-http.cpp`, and compares actual host-buffer traces with
the explorer. A published-header mismatch or compilation failure is not silently
replaced with the local header. It is opt-in because a published version can
legitimately lag unmerged alpaka changes; the regular CI remains tree-local.
CUDA execution still requires hardware even when the HTTP-header CUDA build succeeds.

## Frontend structure

The frontend has three separate kinds of data:

- `web/model.mjs` defines shared mapping configuration and serialization. C++ remains
  the authority for worker identities, index sequences, and effective policies.
- `web/exploration.mjs` builds a trace model once per engine result: ownership,
  timeline, worker/block lookups, diagnostics, and snippets. Exploration state has
  one logical-step cursor and a selected worker ID; the selected assignment and
  outer/inner coordinates are derived, never synchronized by a renderer.
- `web/app.mjs` connects controls and playback to those transitions and renders the
  linked domain/worker views. `web/timeline.mjs` and `web/inspector.mjs` own their DOM
  updates. Worker buttons and map-cell groups remain mounted until the trace changes;
  timeline events remain mounted until its window, filter, or collapsed groups change.
  Rendering does not change exploration state. Presentation failures are reported
  separately from rejected engine requests and do not discard the valid trace.

`web/layout.mjs` owns local workspace preferences, not exploration state. Both resize
axes share pointer/keyboard handling. HTML `data-fit-region` markers declare the
regions reserved below the desktop workspace; optional `data-fit-chrome` descendants
identify their collapsed footprint. Height measurement and observation use the same
regions; `data-fit-above` marks content whose size changes can move the workspace.
Neither depends on footer or inspector descendant selectors.

## Verification and freshness

```sh
node tools/index-mapping/test.mjs
npm --prefix tools/index-mapping ci
cd tools/index-mapping
npx playwright install chromium
npm test
# Optional: regenerate desktop/wide/mobile screenshots (not visual regression tests).
npm run capture
```

`npm run test:unit` and `npm run test:browser` run the frontend layers separately.
Engine/backend checks remain `npm run test:engine`; they require built artifacts.

For an installed Chromium, omit the install command and use
`CHROMIUM_PATH=/usr/bin/chromium npm test` instead.

- Engine tests compare complete native and WASM JSON traces, and optionally the
  freshly generated Godbolt-header executable, including index sequence order.
- Independent known-answer, coverage, ownership, linearization, empty-range,
  invalid-input and uneven-tile tests supplement cross-compilation comparisons.
- Real-backend tests run the buffer-writing CPU suite and compare its actual
  copied-back records with both the native explorer and WASM. With `--single-header`
  they also execute the locally resolved Godbolt source. With `--validate-http-header`
  they additionally test the actual downloaded URL header and record its hash. Optional CUDA builds run
  the same suite on a GPU when available; strict GPU execution can be required.
- Pure exploration, visualization, and layout tests check selection transitions,
  cursor/iteration clamping, trace replacement, nested-loop alignment, sparse/idle
  workers, history, repeated ownership, deterministic palettes, bounded timeline
  windows, saved-preference validation, and pane-width constraints.
- Browser tests compare rendered ownership/highlighting and timeline events with
  native C++ traces; test composition, hover previews, shared selection, reversible
  history, playback, reduced motion, filtering, range editing, guards, errors,
  shared configurations, pane resizing/persistence, storage failures, full-width
  desktops, vertical fit/drag sizing, composition disclosure persistence, 320–900px
  layouts, and mobile touch selection. They also check DOM/focus preservation during
  playback and lazy inspector rendering. Fixtures assert uncaught page errors through
  teardown, including touch contexts. Browser results are saved under
  `build/browser-results/`; optional captures go to `build/screenshots/`.
- Source fingerprints cover the alpaka headers, adapter, and build script; tests
  reject stale engines and stale copied UI assets. Every response carries the
  build identity, which the browser checks against its manifest.
- `.github/workflows/index-mapping.yml` rebuilds and runs these checks when alpaka
  headers or the prototype change, without building Sphinx.

Native-versus-WASM equality alone validates compilation and serialization, not
backend execution. The separate buffer suite adds real-backend evidence for the
covered presets. It is not exhaustive proof: FrameSpec resolution, warp behavior,
other CPU executors, and other device backends remain outside this validation.
Changes in covered alpaka semantics should fail the independent assertions and
require an explicit review rather than regenerating expected results blindly.

All generated artifacts, SDK-independent binaries, and npm dependencies remain
under ignored directories. The source revision and source hash appear in the UI.
Rebuild after editing sources; do not reuse a WASM module from another revision.

## Files and later documentation integration

- `engine.cpp`: bounded request validation, worker contexts, actual alpaka iteration,
  and a JSON trace returned through one C ABI function.
- `build.py`: native/WASM/single-header builds, provenance, and static site assembly.
- `validate_backend.cpp`: real CPU/CUDA kernel launches, per-worker buffer records,
  copy-back checks, independent coverage checks, and HTTP-include Godbolt export entry point.
- `web/`: dependency-free configuration, exploration state, scoped rendering, and layout.
- `test.mjs`: conformance, independent semantic checks, and downloadable-export checks.
- `exploration.test.mjs`: selection transitions and trace-derived model invariants.
- `visualization.test.mjs`: dependency-free tests for timeline and color/history helpers.
- `layout.test.mjs`: saved-layout migration/validation, width constraints, and automatic-height tests.
- `browser.spec.mjs`: rendered-output and interaction checks using Playwright.
- `browser-fixtures.mjs`: shared desktop/touch fixtures and runtime-error checks.
- `capture.spec.mjs`: opt-in screenshot generation, separate from the test suite.

The generated `build/site/` is an ordinary static site. We can review and iterate
here first, then embed the same assets or mount the viewer inside Sphinx. No Sphinx
configuration or existing documentation page has been changed.

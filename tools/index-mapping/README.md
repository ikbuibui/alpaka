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

For hierarchical mappings, the outer range stride is the tile extent and the
inner stride is one. Both are visible in the UI; the direct-mode stride is disabled.
Linearized worker groups need a **1D range** in the current adapter. They are not
silently mapped into a 2D range.

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

## Verification and freshness

```sh
node tools/index-mapping/test.mjs
npm --prefix tools/index-mapping ci
cd tools/index-mapping
npx playwright install chromium
npm test
```

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
- Browser tests compare rendered ownership/highlighting with native C++ traces,
  test selection, iteration filtering, range editing, guards, errors, shared
  configurations, and narrow layouts. A screenshot is saved under
  `build/browser-results/`.
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
- `web/`: dependency-free controls, SVG rendering, and trace inspection.
- `test.mjs`: conformance and independent semantic checks.
- `browser.spec.mjs`: rendered-output checks using Playwright.

The generated `build/site/` is an ordinary static site. We can review and iterate
here first, then embed the same assets or mount the viewer inside Sphinx. No Sphinx
configuration or existing documentation page has been changed.

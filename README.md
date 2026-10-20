# Mandelbrot Formula Explorer

A browser-based Mandelbrot and custom fractal formula explorer focused on deep zoom rendering, interactive experimentation, and algorithm research.

This project extends the original work by Bert Baron with advanced rendering modes, custom iteration formulas, Buddhabrot support, and optional WebGPU acceleration.

## Highlights

- Deep zoom support with automatic algorithm switching (Float64, perturbation, extended-float perturbation)
- Custom fractal formulas such as `z*z + c`, `sin(z) + c`, and other parser-supported expressions
- Optional WebGPU renderers for standard Mandelbrot and custom formulas
- Experimental Buddhabrot rendering (CPU workers and WebGPU path)
- Parallel CPU rendering with tile-based Web Workers
- High-precision reference-point math using BigInt fixed-point arithmetic
- Interactive controls for palettes, supersampling, orbit visualization, coordinates, and image export

## Requirements

- A modern browser with support for:
	- ES modules
	- BigInt
	- Web Workers
- WebGPU is optional (used only when enabled and available)
- Node.js is recommended for running a local development server

## Quick Start

1. Clone this repository.
2. Start the local server:

```bash
node server.js
```

3. Open:

```text
http://localhost:3030
```

## Core Rendering Strategy

The app dynamically switches algorithms based on zoom level and resolution.

- Up to about `1e13`: Float64 Mandelbrot (`mandelbrotFloat.mjs`)
- Up to about `1e300`: Perturbation + BigInt fixed-point reference (`mandelbrotPerturbation.mjs`)
- Beyond about `1e300`: Perturbation + extended float exponent (`mandelbrotPerturbationExtFloat.mjs`)

At very high resolutions, switching can happen at different zoom levels to reduce visible artifacts.

## Main Features

### Fractal Setup

- Fractal Type selector (including presets)
- Julia Set toggle with dedicated reset button
- Custom Iteration Function input with parser-supported math functions
- z0 real and imaginary inputs for initial value control
- Max iterations control

### Rendering Quality and Display

- Fractal GPU toggle for WebGPU acceleration
- Hi-DPI toggle
- Smooth coloring toggle and Escape Radius input
- Supersampling selector (OFF, 2x2, 4x4, 8x8, 16x16, 32x32)
- Palette selector plus palette density and palette rotation controls with reset buttons

### Orbit and Detail Overlays

- Orbit overlay toggle
- Orbit drawing mode selector (Lines+Dots, Lines, Dots)
- Orbit point gradient toggle based on iteration ratio
- Hover detail popup toggle

### Orbit Trap Controls

- Orbit Trap settings panel shown when the Orbit Trap palette is selected
- Shape selector: ring, cross, point, line, parabola, triangle, square, bitmap
- Data mode selector: closest, farthest, average, first capture, TIA, N-th step
- Size, angle, color pattern, center position, threshold, start iteration controls
- Optional bitmap file input with live preview for bitmap-based trapping

### Buddhabrot Controls

- Buddhabrot mode selector (Buddhabrot / Anti-Buddhabrot)
- Buddhabrot view toggle
- Always-on Realtime Navigation: drag, wheel/trackpad zoom, or two-finger pinch on a rendered main or Julia Buddhabrot
- Continuous toggle: accumulate samples until Stop (CPU updates live; GPU display updates every second)
- Sample count input (computed sample total in Continuous mode)
- Sampling range selector: Current View (default) or Initial View; initial sampling uses the selected preset's reset position/zoom while orbit hits map to the current view, on both CPU and GPU
- Buddhabrot GPU toggle
- Buddhabrot palette selector
- Band mode selector for both Buddhabrot and Anti-Buddhabrot (per-trajectory / per-point; defaults to per-trajectory)
- Brightness and gamma controls with reset buttons
- CPU render speed delay slider with reset button
- CPU points-per-batch slider (default 64, maximum is min(Samples, 1024), or 1024 in Continuous mode) for concurrent trajectories across all workers; each advances one step per render delay, with reset button
- Render and Stop actions

Enable **Continuous**, then select **Render Buddhabrot** to keep adding samples
without a finite limit. Changing Continuous does not restart or stop the current
render; the selection applies to the next Render, navigation or render-setting
change. CPU results update about every 100ms at full speed (or each configured
delayed presentation); GPU density is accumulated and displayed every second.
Finite and Continuous CPU rendering share the same 100ms full-speed display
cadence and always publish remaining density at completion. Configured delayed
presentation is preserved. Both modes retain one calculation job, reusing workers
and GPU pipelines. Samples becomes read-only and shows the computed total about every 500ms,
with a final update on Stop. Render time shows the elapsed duration when a
Continuous render is stopped. Turning Continuous off restores the prior finite
sample limit. Render, navigation and render-setting changes start a fresh density
map and reset the current render's count. Reset All Settings turns Continuous off.

While Buddhabrot is shown (including a stopped image), changing Palette, Buddhabrot Mode,
Band Mode or Sampling Range immediately starts a new render. Changing Samples
only changes the limit for the next Render or navigation; it does not restart
the current render. These controls do not start Buddhabrot
when the ordinary fractal view is shown.

In finite mode, Realtime Navigation renders during interaction with the configured Samples,
canvas resolution, supersampling, iterations, CPU delay and points per batch.
There is no reduced-quality interaction mode. Only the latest waiting view is
kept; running frames are reprojected to the current pan and zoom when displayed.
Wheel zoom anchors to the pointer; pinch zoom anchors to the fingers' midpoint
and also supports panning. Release (or a pause in wheel input) renders the final
view with the same settings. High-quality settings can increase update latency.
Stop freezes the current image. Clicking without moving does not stop or restart
the renderer. Both normal fractal and Buddhabrot canvases show a grabbing cursor
while dragging. With Orbit enabled, a stationary click or single-finger tap pins
the orbit; clicking/tapping its marker again releases it, including on Julia
Buddhabrot. Drags and pinches do not toggle orbit pins.

Interaction reference: [Buddhabrot Explorer](https://heledron.com/misc/buddhabrot/).
Navigation reuses this project's CPU/WebGPU renderers.

### Navigation, Export, and Session Flow

- Fullscreen toggle
- Reset All Settings action
- Save Image action
- Jump To favorites selector
- GPU-only prepared animation with zoom-speed and frame-rate sliders

Enable **Animation**, apply coordinates, then select **Prepare**.
Preparation renders the entire path at the current resolution, iterations, smooth
coloring, and supersampling settings. The percentage reaches 100% after all frames
have been saved; **Play** then starts at the initial view, pans to the target, and
zooms in. Playback uses the saved images without recalculating the fractal.

**Stop** cancels preparation or holds the current playback frame. A subsequent
**Play** starts again from the beginning. Changing rendering settings, speed, frame
rate, coordinates, or canvas resolution requires preparation again. Unapplied
coordinate text is not used as the target. Animation OFF hides its settings and Stop.

**Auto adjust iterations & density** is OFF by default. When enabled, **Minimum
iterations** defaults to 1000 if left empty; the maximum is the applied target's
Max iterations. A minimum at or above the target makes iterations constant, even
when the target is below 1000. Palette density uses the analyzed starting-depth
value during the initial pan and ends at the target value (including negative
values). Iterations increase monotonically and density moves in one direction
within the range between 0 and the target; the final frame uses the exact target
settings. Equal start/target zoom uses the target settings
throughout; zoom-out paths use the same start-to-end adjustment rules.

Preparation first analyzes 17 evenly spaced logarithmic zoom depths at the target
center, with a probe image of at most 192 pixels on its long edge and smoothing and
supersampling disabled only for analysis. Escaped-iteration quantiles estimate an
iteration budget with 10% headroom and palette frequency matching. At least 32
escaped samples and a nonzero distribution width are needed; insufficient data
uses a smooth depth-based fallback. Fixed endpoints, monotonic regression and
cubic interpolation prevent parameter reversals and overshoot. This approximates
detail and color-band density within the allowed ranges; it cannot guarantee
identical appearance or preserve every tiny feature. Stripe, Grid and Orbit Trap
keep their existing density-independent coloring. Full frames keep the configured
resolution, smoothing and supersampling. Analysis occupies the first 10% of
progress and can be stopped; 100% still requires every frame to be saved.

Changing automatic adjustment or its minimum invalidates preparation. Animation
Settings reset restores automatic adjustment to OFF and clears the minimum.
Target iteration/density controls retain their configured values during analysis
and playback.

Animation requires WebGPU and browser temporary storage (OPFS and Web Locks).
It turns GPU on automatically and does not fall back to CPU rendering. Julia,
Buddhabrot, and settings unsupported by the GPU cannot be prepared. Lossless PNG
frames are stored temporarily with a limit of 2 GiB or 80% of estimated free quota,
whichever is smaller. Buffer limits and a 128 MiB decoded-image budget are checked
without reducing quality. Temporary frames are deleted on invalidation; abandoned
sessions are cleaned on the next visit without deleting another tab's active data.

Animation checks: `node --test test/animationTests.mjs`. With the local server
running, open `test/animationBrowserTests.html` and `test/gpuResourceTests.html`
in a WebGPU-capable browser for GPU and storage integration tests.

### Coordinates and Diagnostics

- Manual X (real), Y (imaginary), and Zoom text inputs
- Apply Coordinates and Reset Coordinates actions
- Render time display

## Key Files

- `index.html`, `index.js`, `style.css`: UI and application orchestration
- `worker.js`, `workerLoader.mjs`, `workerContext.mjs`: parallel rendering infrastructure
- `fxp.mjs`: BigInt fixed-point arithmetic utilities
- `sharedCalculations.mjs`: shared high-precision Mandelbrot calculations
- `referencePointProvider.mjs`: perturbation reference-point caching
- `mandelbrot*.mjs`: Mandelbrot renderers (CPU/WebGPU/custom)
- `palette.js`, `buddhaPalettes.mjs`: color systems
- `functionPresets.mjs`: built-in function presets

## Rendering resources

CPU Workers match the browser's reported logical core count and send up to four
16×16 tiles per message. Expensive deep-zoom, high-iteration and supersampled
tasks remain separate to keep progressive rendering responsive. Orbit Trap
configuration and bitmap data are sent only when changed; angle and bitmap-size
calculations are cached. Custom expressions allocate iteration history only when
they use `zAt`, `zDelay` or `delayZ`. With supersampling OFF, expensive custom
expressions reuse matching iteration steps for Orbit Trap; divergent trajectories
fall back to the original evaluator, and cached steps are limited to 4096.

Detailed CPU/GPU timing diagnostics are removed. Rendering duration display,
animation pacing, stop polling and UI throttling use `Date.now()`.

Custom renderers reuse buffers for equal pixel counts and smooth settings,
combine readbacks when the device buffer limit allows it, and retain up to four
compiled pipeline variants. Shader output and all palette channels are preserved. Standard Mandelbrot
also reuses equal-sized buffers across renders and supersampling changes. Output
channels are cleared once at the start of a new render; intermediate perturbation
passes keep their accumulated results. Resizing replaces only affected buffers.

## Testing

CPU rendering regressions: `node --test test/cpuRenderingTests.mjs`.
Complex formula and escape-rule regressions: `node --test test/complexIterationTests.mjs`.
Open `test/complexIterationBrowserTests.html` for actual CPU/WebGPU escape
classification, Orbit Trap and Buddhabrot orbit-location checks. Custom formulas
use the selected escape radius with Smooth both OFF and ON; smoothing never
changes an escaped point into an interior point. Non-finite results and
components with magnitude at least `1e20` terminate the orbit as escaped.
Buddhabrot/Anti-Buddhabrot automatically align orbit hits with the ordinary
parameter-plane axes for formulas of the form `f(z) + a*c + b`: hits are displayed
at `(z-b)/a`. This aligns rotations, translations and scaling such as `c*i`,
`c+i` and `2*c`. Iteration and escape tests retain the original coordinates.
Julia and nonlinear, mixed or singular parameter terms retain raw orbit coordinates.
Run `node --test test/buddhabrotProjectionTests.mjs` for projection regressions.
With the local server running, open `test/cpuWorkerTests.html` for Worker batch,
bitmap-cache update, cancellation and subsequent-job checks.

Buddhabrot CPU checks: `node test/buddhabrotWorkerTests.mjs`,
`node test/buddhabrotRunnerTests.mjs`, `node test/buddhabrotConvergenceTests.mjs`,
`node test/buddhabrotRealtimeTests.mjs`, `node test/buddhabrotSettingsTests.mjs`
and `node test/buddhabrotContinuousTests.mjs`.
The worker tests accept `BUDDHA_REFERENCE_WORKER=/path/to/old-worker.mjs` to
compare seeded orbit hits, colors and immediate RGB density with an earlier implementation.
Run `node test/buddhabrotCpuBenchmark.mjs` for a seeded single-worker Node benchmark
(640×360, 1,000 iterations, Delay=0, two warmups plus three measured runs).
It also accepts `BUDDHA_REFERENCE_WORKER`, `BUDDHA_BENCH_SAMPLES` (default 10,000)
and `BUDDHA_BENCH_REPEATS` (default 3). `BUDDHA_BENCH_CASES` filters case names
with a regular expression. Reference/current runs alternate to reduce timing bias.
These times exclude browser canvas coloring.
Open `test/buddhabrotPerformanceBrowserTests.html` for fixed-seed GPU timing
(seven measured runs after two warmups). It uses GPU timestamps when supported,
and reports end-to-end sampling/readback time separately. An optional
`?reference=./reference.mjs` compares a local earlier module and verifies exact
density parity; its relative imports must resolve from the reference location.
Use `&compareDensity=false` only when comparing with a version whose intended
display coordinates differ, such as the version before parameter-plane alignment.
Immediate CPU sampling yields at sample checkpoints when its 8 ms time slice
expires; delayed rendering still advances each active trajectory once per Delay.

Browser-based tests are available under the `test/` directory.

1. Start the local server:

```bash
node server.js
```

2. Open:

```text
http://localhost:3030/test/test.html
```

## Performance Notes

- CPU mode scales with worker count and tile size.
- WebGPU can significantly improve throughput at deeper zoom levels, but smoothness may vary by device.
- BigInt operations are intentionally concentrated in high-precision reference calculations to keep hot loops fast.

## Attribution

This project is based on [bertbaron/mandelbrot](https://github.com/bertbaron/mandelbrot) by Bert Baron and includes substantial enhancements.

## License

GPL-3.0. See [LICENSE](LICENSE) for details.

GPU resource regression tests require a WebGPU-capable browser and the local server:

```text
http://localhost:3030/test/gpuResourceTests.html
```

They cover buffer reuse, output stability after setting changes, odd image sizes,
readback buffer limits, mapping failure recovery, and repeated rendering. Shader
outputs are also compared with frozen pre-optimization generators, including
reference-limit ties, custom formulas using iteration numbers and orbit history,
smooth coloring, and supersampling.

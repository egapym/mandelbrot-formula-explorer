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
- Sample count input
- Buddhabrot GPU toggle
- Buddhabrot palette selector
- Brightness and gamma controls with reset buttons
- CPU render speed delay slider with reset button
- Render and Stop actions
- Hi-res scale selector (1x to 5x) and Save Hi-Res Image action

### Navigation, Export, and Session Flow

- Fullscreen toggle
- Reset All Settings action
- Save Image action
- Jump To favorites selector
- Animation toggle and stop action

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

## Testing

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

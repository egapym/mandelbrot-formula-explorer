import assert from 'node:assert/strict'
import { CpuDensityColorMap } from '../buddhabrotCpuColorMap.mjs'

// Independent full-image reference matching the existing CPU display formula.
function reference(r, g, b, brightness, gamma) {
  const pixels = new Uint8ClampedArray(r.length * 4)
  let max = 0
  for (let i = 0; i < r.length; i++) max = Math.max(max, r[i] + g[i] + b[i])
  if (max === 0) {
    for (let i = 3; i < pixels.length; i += 4) pixels[i] = 255
    return pixels
  }
  const invLogDenom = (1 / Math.log10(1 + max)) * 2
  const denomScale = 1 + Math.log10(1 + max) / 4
  for (let i = 0; i < r.length; i++) {
    for (const [channel, buf] of [r, g, b].entries()) {
      const linear = Math.log10(1 + buf[i]) * invLogDenom
      pixels[i * 4 + channel] = (255 * Math.min(1, ((linear * brightness) / denomScale) ** gamma)) | 0
    }
    pixels[i * 4 + 3] = 255
  }
  return pixels
}

const width = 17, height = 13, size = width * height
const r = new Float32Array(size), g = new Float32Array(size), b = new Float32Array(size)
const pixels = new Uint8ClampedArray(size * 4)
const mapper = new CpuDensityColorMap(r, g, b, width, height, pixels)
let checks = 0
function check(brightness, gamma, chunks) {
  const previous = pixels.slice()
  const update = mapper.update(brightness, gamma, chunks)
  assert.deepEqual(pixels, reference(r, g, b, brightness, gamma), 'Incremental RGB differs from full mapping')
  for (let i = 0; i < size; i++) {
    if (previous.subarray(i * 4, i * 4 + 4).some((value, channel) => value !== pixels[i * 4 + channel])) {
      assert.ok(i % width >= update.x && i % width < update.x + update.width)
      assert.ok(Math.floor(i / width) >= update.y && Math.floor(i / width) < update.y + update.height)
    }
  }
  checks++
  return update
}
check(1.2, 4.8)
r[0] = 20
check(1.2, 4.8, [{ x: 0, y: 0, indices: [0] }])
g[width + 3] = 0.25
const small = check(1.2, 4.8, [{ x: 3, y: 1, indices: [0] }])
assert.equal(small.full, false)
assert.deepEqual([small.x, small.y, small.width, small.height], [3, 1, 1, 1])

for (let step = 0; step < 200; step++) {
  const i = (step * 37) % size, j = (step * 53 + 11) % size
  r[i] += 0.125
  g[j] += 0.75
  b[i] += 0.5
  // Coalesced workers must both contribute their dirty indices.
  check(step < 100 ? 1.2 : 2.1, step < 150 ? 4.8 : 0.1, [
    { x: 0, y: 0, indices: [i] }, { x: 0, y: 0, indices: [j, i] },
  ])
}
r[0] = 100
assert.equal(check(2.1, 0.1, [{ x: 0, y: 0, indices: [0] }]).full, true, 'New maximum did not recolor existing pixels')
// Dense chunks, silent buffer resets, display changes, and resumed rendering.
r[5] += 1
check(0.8, 20, [{ x: 0, y: 0, w: width, h: height }])
r.fill(0); g.fill(0); b.fill(0)
check(1.2, 4.8)
b[4] = 0.5
check(1.2, 4.8, [{ x: 0, y: 0, indices: [4] }])
console.log(`PASS: ${checks} exact RGBA comparisons, sparse bounds, normalization, settings, multi-worker chunks and reset`)

// Compare the expensive full-image formula with the incremental implementation
// under identical sparse updates. Timings are diagnostic, never pass criteria.
const benchSize = 640 * 360
const br = new Float32Array(benchSize), bg = new Float32Array(benchSize), bb = new Float32Array(benchSize)
br[0] = 100
const benchMapper = new CpuDensityColorMap(br, bg, bb, 640, 360, new Uint8ClampedArray(benchSize * 4))
benchMapper.update(1.2, 4.8)
const steps = 100
const started = performance.now()
for (let step = 1; step <= steps; step++) {
  br[step] = 0.5
  benchMapper.update(1.2, 4.8, [{ x: 0, y: 0, indices: [step] }])
}
const incrementalMs = performance.now() - started
br.fill(0)
br[0] = 100
const fullStarted = performance.now()
for (let step = 1; step <= steps; step++) {
  br[step] = 0.5
  reference(br, bg, bb, 1.2, 4.8)
}
const fullMs = performance.now() - fullStarted
assert.deepEqual(benchMapper.pixels, reference(br, bg, bb, 1.2, 4.8))
console.log(`Color mapping only, 640x360, Batch 1, ${steps} steps: full=${fullMs.toFixed(1)}ms incremental=${incrementalMs.toFixed(1)}ms`)

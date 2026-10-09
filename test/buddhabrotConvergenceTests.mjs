import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

// Execute the scalar integer operations extracted from the actual WGSL, rather
// than maintaining a second implementation of the optimized state machine.
const source = readFileSync(new URL('../buddhabrotWebGPU.mjs', import.meta.url), 'utf8')
const extract = (name) => {
  const block = source.split(`// convergence ${name} begin`)[1]?.split(`// convergence ${name} end`)[0]
  assert.ok(block, `Missing convergence ${name} block`)
  return block.replace(/:\s*i32/g, '').replace(/\b(\d+)u\b/g, '$1').replace(/i32\(oi\)/g, 'oi')
}
const createTracker = new Function(`${extract('state')}
  return (currentPixelIdx, oi, iter) => {
    ${extract('update').replace(/break;/g, 'return true;')}
    return false;
  }`)

const cpuSource = readFileSync(new URL('../buddhabrotWorker.mjs', import.meta.url), 'utf8')
const extractCpu = (name) => {
  const block = cpuSource.split(`// convergence ${name} begin`)[1]?.split(`// convergence ${name} end`)[0]
  assert.ok(block, `Missing CPU convergence ${name} block`)
  return block
}
const createCpuTracker = new Function(`${extractCpu('state')}
  return (currentPixelIdx, k, trajLen) => {
    const stepped = false;
    ${extractCpu('update').replace('if (stepped) yield false', '').replace(/\breturn\b/g, 'return true')}
    return false;
  }`)

let checks = 0
function verify(pixels) {
  const track = createTracker()
  const trackCpu = createCpuTracker()
  for (let i = 0; i < pixels.length; i++) {
    const expected = i >= 200 && i % 10 === 0 &&
      new Set(pixels.slice(Math.max(0, i - 49), i + 1).filter((p) => p >= 0)).size <= 2
    assert.equal(track(pixels[i], i, pixels.length), expected, `Window ending at ${i}`)
    assert.equal(trackCpu(pixels[i], i, pixels.length), expected, `CPU window ending at ${i}`)
    checks++
  }
}

for (const length of [1, 199, 200, 201, 210, 211, 1000]) {
  for (const period of [1, 2, 3, 49, 50, 51]) {
    verify(Array.from({ length }, (_, i) => i % period))
  }
  verify(Array(length).fill(-1))
}
// A third distinct pixel exactly inside/outside the 50-point window, followed
// by two-pixel cycles or entirely offscreen points.
for (let third = 149; third <= 211; third++) {
  for (const outside of [false, true]) {
    verify(Array.from({ length: 300 }, (_, i) => i === third ? 2 : outside ? -1 : i % 2))
  }
}
let seed = 0x12345678
const rand = () => {
  seed ^= seed << 13
  seed ^= seed >>> 17
  seed ^= seed << 5
  return seed >>> 0
}
for (let trial = 0; trial < 2000; trial++) {
  let pixel = -1
  verify(Array.from({ length: 1000 }, (_, i) => {
    if (i % 75 === 0 || rand() % 8 === 0) pixel = rand() % (trial % 7 + 1) - 1
    return pixel
  }))
}
console.log(`PASS: ${checks} CPU/GPU convergence decisions match the 50-point Set oracle`)

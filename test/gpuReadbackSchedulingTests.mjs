import assert from 'node:assert/strict'
import { MandelbrotWebGPU } from '../mandelbrotWebGPU.mjs'
import * as fxp from '../fxp.mjs'

// Exercise the real orchestration with deterministic clocks and GPU pass data.
// Actual buffer mapping and shader parity are covered by gpuResourceTests.html.
const task = {
  w: 2, h: 1, frameWidth: 2, frameHeight: 1, xOffset: 0, yOffset: 0,
  frameTopLeft: [fxp.fromNumber(-2), fxp.fromNumber(-1)],
  frameBottomRight: [fxp.fromNumber(1), fxp.fromNumber(1)],
  precision: 64, maxIter: 100, smooth: true, escapeRadius: 4, readEscapeZ: false,
}
const originalNow = Date.now
for (const mode of ['fast', 'visible', 'stop-pass', 'stop-reference', 'final-only', 'final-only-stop']) {
  let clock = 0, passes = 0, reads = 0, stopped = false
  const updates = []
  Date.now = () => clock
  const renderer = Object.create(MandelbrotWebGPU.prototype)
  renderer.p = { onGpuUpdate: (data) => updates.push(data) }
  renderer.max_iter = 100
  renderer.currentTask = 'job'
  renderer.updateCache = () => {}
  renderer.referencePoints = [{ rr: 0n, ri: 0n, size: 2 }, { rr: 0n, ri: 0n, size: 2 }]
  renderer.getInitialIndices = () => new Uint32Array([0, 1])
  renderer.shouldStop = () => stopped
  renderer.calculate_reference = async () => {
    stopped = true
    return { rr: 0n, ri: 0n }
  }
  if (mode === 'stop-reference') renderer.referencePoints.pop()
  renderer.perturbationPass = async (data) => {
    assert.equal(data.deferReadback, true, 'Pass eagerly read back full pixels')
    passes++
    clock = mode === 'visible' ? passes * 101 : 0
    stopped = mode === 'stop-pass' || mode === 'final-only-stop'
    return { indices: new Uint32Array(passes === 1 ? [1] : []) }
  }
  renderer.mandelbrotPipeline = {
    readResults: async (options) => {
      reads++
      assert.equal(options.readEscapeZ, false)
      return { values: new Int32Array([4, 7]), smooth: new Uint8Array([5, 6]),
        signs: new Int8Array([1, 2]), zreal: null, zimag: null }
    },
    finish: async () => {},
  }
  const result = await renderer.calculate(2, 1, false, {
    ...task, jobToken: 'job', finalOnly: mode.startsWith('final-only'),
  })
  assert.equal(reads, mode === 'visible' ? 2 : mode === 'final-only-stop' ? 0 : 1, mode)
  if (mode === 'final-only-stop') {
    assert.equal(updates.length, 0)
    assert.equal(result.error, 'Stopped')
  } else {
    assert.equal(updates.at(-1).isFinished, true)
    assert.deepEqual(Array.from(updates.at(-1).values), [4, 7], 'Final/stopped pixels were lost')
    assert.equal(updates.length, mode === 'visible' ? 2 : 1)
  }
}
Date.now = originalNow
console.log('PASS: deferred GPU readback, visible update cadence, completion, both Stop paths and final-only cancellation')

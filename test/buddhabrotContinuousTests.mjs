import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import { BuddhabrotContinuousRunner as Runner } from '../buddhabrotContinuous.mjs'

const fake = (gpu = false) => ({
  width: 2, height: 2, starts: [], stops: 0,
  ...(gpu ? { devicePromise: Promise.resolve({}) } : {}),
  start(params) {
    this.starts.push(params)
    for (const key of ['densityR', 'densityG', 'densityB']) this[key] = new Float32Array(4)
  },
  stop() { this.stops++ }, terminate() {},
})
const params = { width: 2, height: 2, samples: 1, maxIter: 1000, mode: 'antibuddha', gpu: false }
for (const gpu of [false, true]) {
  const native = fake(gpu)
  let chunks = 0, completions = 0
  const runner = new Runner({ runner: native, onChunk: () => chunks++, onComplete: () => completions++ })
  await runner.start({ ...params, gpu })
  assert.equal(native.starts[0].samples, Infinity)
  assert.equal(native.starts[0].continuous, true)
  assert.equal(native.starts[0].mode, 'antibuddha')
  assert.equal(runner.densityR, native.densityR, 'Reuse native accumulation without copies')
  native.densityR[0] = 7
  native.onProgress({ delta: 90000 })
  native.onChunk({ continuous: true })
  assert.equal(runner.totalSamples, 90000, 'Do not truncate progress at the old batch limit')
  assert.equal(runner.densityR[0], 7)
  assert.equal(chunks, 1)
  assert.equal(native.starts.length, 1, 'Keep one calculation job alive')
  runner.stop()
  if (gpu) {
    assert.equal(completions, 0, 'Wait for GPU final readback')
    native.densityR[0] = 9
    native.onProgress({ delta: 100 })
    native.onComplete({ aborted: true })
    native.onComplete({})
    assert.equal(runner.totalSamples, 90100)
    assert.equal(runner.densityR[0], 9)
  }
  await Promise.resolve()
  assert.equal(completions, 1, 'Stop must complete exactly once')
  native.onProgress({ delta: 1000 })
  assert.equal(runner.totalSamples, gpu ? 90100 : 90000)
}
const fallback = fake()
const noDevice = fake(true)
noDevice.devicePromise = Promise.resolve(null)
const fallbackLoop = new Runner({ runner: noDevice, createCpuRunner: () => fallback })
await fallbackLoop.start(params)
assert.equal(fallbackLoop.useGpu, false)
assert.equal(fallback.starts.length, 1)
fallbackLoop.terminate()
fallback.onComplete({})
console.log('PASS: one native continuous job, density aliases, unbounded totals, Stop races and CPU fallback')

const appSource = readFileSync(new URL('../index.js', import.meta.url), 'utf8')
let now = 0, sampleText = '1234'
const writes = []
const input = {
  readOnly: false,
  get value() { return sampleText },
  set value(value) { sampleText = value; writes.push({ value, time: now }) },
}
const ui = {
  DOM: { buddha: { continuous: { checked: true }, iterations: input } },
  performance: { now: () => now },
  syncBuddhaPointBatchSize() {},
}
runInNewContext(appSource.slice(appSource.indexOf('let buddhaFiniteSampleValue = null'),
  appSource.indexOf('function wrapContinuousBuddhaRunner(')), ui)
ui.syncBuddhaContinuousModeUi()
assert.equal(input.readOnly, true)
assert.equal(input.value, '0')
now = 499
ui.updateBuddhaContinuousSamples(100)
assert.equal(input.value, '0')
now = 500
ui.updateBuddhaContinuousSamples(200)
assert.equal(input.value, '200')
now = 600
ui.updateBuddhaContinuousSamples(300)
assert.equal(input.value, '200')
runInNewContext('updateBuddhaContinuousSamples(buddhaContinuousSampleCount, true)', ui)
assert.equal(input.value, '300', 'Stop must display the latest sample count immediately')
ui.DOM.buddha.continuous.checked = false
ui.syncBuddhaContinuousModeUi()
assert.equal(input.value, '1234', 'Restore the finite limit instead of using the computed total')
assert.equal(input.readOnly, false)
ui.updateBuddhaContinuousSamples(999, true)
assert.equal(input.value, '1234', 'Late counters must not overwrite a restored finite limit')
console.log('PASS: 500ms Samples cadence, final counter flush and finite-limit restoration')

let resolveCpu
const lateDevice = fake(true)
lateDevice.devicePromise = Promise.resolve(null)
const lateCpu = fake()
const lateLoop = new Runner({ runner: lateDevice, createCpuRunner: () => new Promise((resolve) => { resolveCpu = resolve }) })
const initializing = lateLoop.start(params)
for (let i = 0; i < 8 && !resolveCpu; i++) await Promise.resolve()
assert.equal(typeof resolveCpu, 'function')
lateLoop.terminate()
resolveCpu(lateCpu)
await initializing
assert.equal(lateCpu.stops, 1)
assert.equal(lateCpu.starts.length, 0, 'Stop/terminate during fallback initialization must not start workers later')
console.log('PASS: cancellation during asynchronous CPU fallback initialization')

let resolveDevice
let earlyCompletions = 0
const pendingGpu = fake(true)
pendingGpu.devicePromise = new Promise(resolve => { resolveDevice = resolve })
const pendingLoop = new Runner({ runner: pendingGpu, onComplete: () => earlyCompletions++ })
const pendingStart = pendingLoop.start(params)
pendingLoop.stop()
await Promise.resolve()
assert.equal(earlyCompletions, 1, 'Stop during GPU initialization must complete')
resolveDevice({})
await pendingStart
assert.equal(pendingGpu.starts.length, 0, 'A stopped GPU initialization must not dispatch later')

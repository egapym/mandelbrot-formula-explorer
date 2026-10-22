import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import { BuddhabrotRealtimeRenderer, getBuddhabrotFrameTransform } from '../buddhabrotRealtime.mjs'

const flush = async () => { for (let i = 0; i < 8; i++) await Promise.resolve() }
// Check reprojection numerically, including zoom-out and a moved pinch midpoint.
for (const [width, height] of [[1600, 900], [390, 844]]) {
  for (const from of [
    { center: { x: -0.5, y: 0 }, zoom: 1 },
    { center: { x: 0.2, y: -0.3 }, zoom: 12 },
  ]) {
    for (const to of [
      from,
      { center: { x: -0.4, y: 0.25 }, zoom: 0.1 },
      { center: { x: -0.1, y: 0.2 }, zoom: 24 },
    ]) {
      const t = getBuddhabrotFrameTransform(width, height, from, to)
      for (const point of [{ x: 0, y: 0 }, { x: -0.75, y: 0.3 }]) {
        const before = {
          x: width / 2 + (point.x - from.center.x) * from.zoom * width / 4,
          y: height / 2 + (point.y - from.center.y) * from.zoom * width / 4,
        }
        const expected = {
          x: width / 2 + (point.x - to.center.x) * to.zoom * width / 4,
          y: height / 2 + (point.y - to.center.y) * to.zoom * width / 4,
        }
        assert.ok(Math.abs(before.x * t.scale + t.x - expected.x) < 1e-8)
        assert.ok(Math.abs(before.y * t.scale + t.y - expected.y) < 1e-8)
      }
    }
  }
}

for (const gpuStyle of [false, true]) {
  const starts = [], draws = [], errors = []
  let finishStart, released = 0, stopped = 0, creates = 0
  const runner = {
    start(params) {
      starts.push(params)
      if (gpuStyle) return new Promise((resolve) => { finishStart = resolve })
      return Promise.resolve()
    },
    stop() { stopped++ },
    terminate() { released++ },
  }
  const preview = new BuddhabrotRealtimeRenderer({
    createRunner: async () => { creates++; return runner },
    draw: (_, params) => draws.push(params),
    onError: (error) => errors.push(error),
  })
  preview.request(1)
  await flush()
  preview.request(2)
  preview.request(3)
  await flush()
  assert.deepEqual(starts, [1], 'Never overlap rendering jobs')
  runner.onChunk()
  runner.onComplete({})
  finishStart?.()
  await flush()
  assert.deepEqual(starts, [1, 3], 'Coalesce intermediate pointer moves')
  assert.equal(creates, 1, 'Reuse runner across frames')
  assert.deepEqual(draws, [1, 1])
  preview.cancel()
  runner.onComplete({})
  runner.onChunk()
  preview.request(4)
  finishStart?.()
  await flush()
  assert.deepEqual(starts, [1, 3])
  assert.deepEqual(draws, [1, 1], 'No callbacks after cancel')
  assert.equal(released, 1)
  assert.equal(stopped, 1, 'Abort in-flight GPU work even without terminate support')
  assert.deepEqual(errors, [])
}

let resolveCreate, released = 0
const canceled = new BuddhabrotRealtimeRenderer({
  createRunner: () => new Promise((resolve) => { resolveCreate = resolve }),
  draw: () => assert.fail('Canceled initialization drew a frame'),
  onError: () => assert.fail('Canceled initialization reported error'),
})
canceled.request(1)
canceled.cancel()
resolveCreate({ terminate() { released++ }, start() { assert.fail('Canceled runner started') } })
await flush()
assert.equal(released, 1)

const errors = []
const failing = new BuddhabrotRealtimeRenderer({
  createRunner: async () => { throw new Error('unavailable') },
  draw: () => assert.fail('Failed runner drew a frame'),
  onError: (error) => errors.push(error.message),
})
failing.request(1)
await flush()
assert.deepEqual(errors, ['unavailable'])
assert.equal(failing.closed, true)
console.log('Buddhabrot realtime rendering tests passed')

// Exercise the real UI request builder: a saved CPU snapshot must not send
// gpu:false to a newly created GPU runner after the toggle is switched on.
const appSource = readFileSync(new URL('../index.js', import.meta.url), 'utf8')
const requestHandler = appSource.slice(
  appSource.indexOf('function requestRealtimeBuddhaFrame(drag) {'),
  appSource.indexOf('function transformRealtimeBuddhaView('),
)
for (const gpu of [true, false]) {
  for (const kind of ['main', 'julia']) {
    let requested
    const context = {
      lastBuddhaRenderParams: { gpu: !gpu, samplingZoom: 1, maxIter: 1000 },
      fractal: { max_iter: 2500 },
      applyFixedBuddhaSamples: params => params,
      DOM: { buddha: { gpu: { checked: gpu }, iterations: { value: '8192' }, renderSpeed: { value: '0' } } },
      getCpuBuddhabrotRenderDelay: () => 0,
      syncBuddhaPointBatchSize: () => 64,
      updateCoordinateInputs() {}, showZoomFactor() {}, refreshCoordinateGrid() {},
      drag: {
        kind, canvas: { width: 800, height: 600 },
        renderer: {
          supersampling: 2, max_iter: 1000,
          center: [{ toNumber: () => -0.5 }, { toNumber: () => 0 }],
          zoom: { toNumber: () => 1 },
        },
        rendering: { request: (params) => { requested = params } },
      },
    }
    runInNewContext(`${requestHandler}\nrequestRealtimeBuddhaFrame(drag)`, context)
    assert.equal(requested.gpu, gpu, 'Navigation must use the current GPU toggle, not the saved render snapshot')
    assert.equal(requested.samples, 32768)
    assert.equal(requested.width, 800)
    assert.equal(requested.supersampling, 2)
    assert.equal(requested.maxIter, 2500, `${kind} navigation must use the latest Max iterations`)
  }
}
console.log('PASS: CPU/GPU switches update navigation requests without reducing quality')

// A settings change is deferred until the next render request, including a
// source refresh with the same Julia c. Do not mutate the active density job.
let refreshFrame
let refreshed = 0
const settingsContext = {
  fractal: { max_iter: 1000, initPallete() {} },
  DOM: { iterations: { value: '1000' } },
  buddhaRunner: { maxIter: 1000 },
  updatePermalink() {},
  juliaState: { active: true },
  isBuddhabrotViewShownOnJulia: () => true,
  getCurrentJuliaCFromMain: () => ({ re: -0.5, im: 0 }),
  lastBuddhaRenderParams: { juliaRe: -0.5, juliaIm: 0, maxIter: 1000 },
  requestAnimationFrame: (callback) => { refreshFrame = callback; return 1 },
  startBuddhaRender: (options) => {
    assert.equal(options.preserveSourceRender, true)
    refreshed++
    return Promise.resolve()
  },
  buddhaRenderRequestGeneration: 0,
  console,
}
runInNewContext(appSource.slice(
  appSource.indexOf('function setIterations(value) {'),
  appSource.indexOf('/**\n * 入力欄の座標', appSource.indexOf('function setIterations(value) {')),
) + appSource.slice(
  appSource.indexOf('let juliaBuddhabrotSourceRedrawFrame = null'),
  appSource.indexOf('/**\n * 現在の Mandelbrot 中心', appSource.indexOf('let juliaBuddhabrotSourceRedrawFrame = null')),
), settingsContext)
settingsContext.scheduleJuliaBuddhabrotSourceRedraw()
refreshFrame()
assert.equal(refreshed, 0, 'Unchanged settings and Julia c must skip duplicate refreshes')
assert.equal(settingsContext.setIterations(2500), true)
assert.equal(refreshed, 0, 'Changing Max iterations must not restart Buddhabrot on its own')
assert.equal(settingsContext.buddhaRunner.maxIter, 1000, 'Active job must retain its original iteration limit')
settingsContext.scheduleJuliaBuddhabrotSourceRedraw()
refreshFrame()
assert.equal(refreshed, 1, 'The next source refresh must apply Max iterations even with the same Julia c')
console.log('PASS: deferred Max iterations changes refresh Julia without mixing active sampling settings')

const continuousViews = [], continuousDraws = []
const navigation = new BuddhabrotRealtimeRenderer({
  createRunner: async () => {
    const runner = { continuous: true, starts: [], stopped: false,
      start(params) { this.starts.push(params) },
      stop() {}, terminate() { this.stopped = true },
    }
    continuousViews.push(runner)
    return runner
  },
  draw: (_, params) => continuousDraws.push(params),
  onError: (error) => { throw error },
})
navigation.request(1)
await flush()
continuousViews[0].onChunk()
navigation.request(2)
navigation.request(3)
await flush()
assert.equal(continuousViews[0].stopped, true)
continuousViews[0].onChunk()
assert.deepEqual(continuousViews[1].starts, [3])
continuousViews[1].onChunk()
assert.deepEqual(continuousDraws, [1, 3], 'Replace an infinite view without letting old frames overwrite it')
navigation.cancel()
assert.equal(continuousViews[1].stopped, true)
console.log('PASS: continuous navigation replaces old views and coalesces pending coordinates')

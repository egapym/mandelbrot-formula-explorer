import assert from 'node:assert/strict'
import { test } from 'node:test'
import { AnimationBuddhabrotSession } from '../animationBuddhabrotSession.mjs'
import * as fxp from '../fxp.mjs'

function fixture(params = {}, start) {
  const received = []
  const view = {
    setZoom(zoom) { this.zoom = zoom },
    setCenter(center) { this.center = center },
  }
  const runner = {
    densityR: new Float32Array([1]), densityG: new Float32Array([0]), densityB: new Float32Array([0]),
    start(value) { received.push(value); return start?.(this) },
    stop() { this.stopped = true },
    terminate() { this.terminated = true },
  }
  const session = new AnimationBuddhabrotSession({
    view,
    params: { width: 1, height: 1, maxIter: 50, samples: 64, brightness: 1, gamma: 1.5, ...params },
    samplingRange: () => ({ samplingCenter: undefined, samplingZoom: undefined }),
    createRunner: async () => runner,
  })
  return { session, runner, received }
}
const frame = {
  center: [fxp.fromNumber(-0.75), fxp.fromNumber(0.1)], zoom: fxp.fromNumber(4),
}

test('frame navigation changes projection while retaining Fixed Samples and configured Buddhabrot settings', async () => {
  const settings = { sampleSeed: 123, samplingCenter: { x: 0, y: 0 }, samplingZoom: 1,
    samplingAspect: 0.5, mode: 'anti', buddhaBandMode: 'perPoint', gpu: false }
  const { session, received } = fixture(settings, () => { throw new Error('captured') })
  await assert.rejects(session.render(frame, new AbortController().signal), /captured/)
  assert.deepEqual(received[0], { ...session.params, reusePipelines: true, orbitCacheBytes: 32 * 1024 ** 2,
    center: { x: -0.75, y: frame.center[1].toNumber() }, zoom: 4 })
  session.dispose()
})

test('Current View sampling follows each animation frame', async () => {
  const { session, received } = fixture({}, () => { throw new Error('captured') })
  await assert.rejects(session.render(frame, new AbortController().signal), /captured/)
  assert.equal(received[0].zoom, 4)
  assert.equal(received[0].samplingCenter, undefined)
  assert.equal(received[0].samplingZoom, undefined)
  session.dispose()
})

test('Stop cancels an unfinished frame and releases its runner on disposal', async () => {
  let started
  const active = new Promise((resolve) => { started = resolve })
  const { session, runner } = fixture({}, () => started())
  const controller = new AbortController()
  const rendering = session.render(frame, controller.signal)
  await active
  controller.abort()
  await assert.rejects(rendering, { name: 'AbortError' })
  assert.equal(runner.stopped, true)
  session.dispose()
  assert.equal(runner.terminated, true)
})

test('oversized decoded frames are rejected before creating a runner', () => {
  assert.throws(() => fixture({ width: 8192, height: 8192 }), /image budget/)
})

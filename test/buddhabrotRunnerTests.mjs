import assert from 'node:assert/strict'
import { getRenderPointBatchLimit, normalizeRenderPointBatchSize } from '../buddhabrotRenderConfig.mjs'

for (const [samples, workers, maximum] of [
  [100, 5, 20], [101, 5, 21], [3, 8, 1], [0, 5, 1], [3000000, 8, 1024], [1600, 5, 320],
]) {
  assert.equal(getRenderPointBatchLimit(samples, workers), maximum)
  assert.equal(normalizeRenderPointBatchSize(9999, maximum), maximum)
}

Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { hardwareConcurrency: 1 } })
globalThis.location = { href: 'http://localhost/' }

class FakeWorker {
  constructor() {
    this.messages = []
  }
  postMessage(message) {
    this.messages.push(message)
    if (message.cmd === 'stop' && message.releaseToPool) {
      setTimeout(() => this.onmessage?.({ data: { type: 'released' }, target: this }), 0)
    }
  }
  terminate() {}
}

globalThis.fetch = async () => ({ ok: true, text: async () => '' })
URL.createObjectURL = () => 'blob:fake-worker'
globalThis.Worker = FakeWorker

const { BuddhabrotRunner } = await import('../buddhabrot.mjs')
const makeRunner = (onComplete) => new BuddhabrotRunner({
  workerCount: 1,
  width: 2,
  height: 2,
  onComplete,
})

let completed = 0
const first = makeRunner(() => completed++)
assert.equal(first.renderPointBatchSize, 64, 'Initial default batch size changed')
await first._workersReady
await first.start({ samples: 10 })
const worker = first.workers[0]
assert.equal(worker.messages.at(-1).renderPointBatchSize, 10, 'Start did not clamp batch size to assigned samples')
first.setRenderPointBatchSize(1)
assert.equal(worker.messages.at(-1).cmd, 'setPointBatchSize')
assert.equal(worker.messages.at(-1).renderPointBatchSize, 1)
first.setRenderPointBatchSize(9999)
assert.equal(first.renderPointBatchSize, 10)
first.setRenderPointBatchSize(0)
assert.equal(first.renderPointBatchSize, 1)
first.setRenderPointBatchSize(NaN)
assert.equal(first.renderPointBatchSize, 10)
const oldJobId = first._currentJobId
first.terminate()
await new Promise((resolve) => setTimeout(resolve, 5))

const second = makeRunner(() => completed++)
await second._workersReady
assert.equal(second.workers[0], worker, 'idle worker was not reused')
await second.start({ samples: 10, renderPointBatchSize: 1024 })
assert.equal(worker.messages.at(-1).renderPointBatchSize, 10, 'Start did not propagate the effective batch limit to reused worker')
const newJobId = second._currentJobId
assert.notEqual(newJobId, oldJobId)

second.running = true
second._pendingWorkers = 1
worker.onmessage({ data: { type: 'done', jobId: oldJobId }, target: worker })
assert.equal(second._pendingWorkers, 1, 'stale completion changed the new job')
assert.equal(completed, 0, 'stale completion fired onComplete')

worker.onmessage({ data: { type: 'done', jobId: newJobId }, target: worker })
assert.equal(second.running, false)
assert.equal(completed, 1)
second.terminate()
console.log('PASS: pooled worker reuse ignores stale completion from prior runner')

const frames = new Map()
let frameId = 0
let currentTime = 0
const originalNow = Date.now
Date.now = () => currentTime
globalThis.requestAnimationFrame = (callback) => {
  frames.set(++frameId, callback)
  return frameId
}
globalThis.cancelAnimationFrame = (id) => frames.delete(id)
const nextFrame = (time) => {
  currentTime = time
  const callbacks = [...frames.values()]
  frames.clear()
  for (const callback of callbacks) callback()
}
const displayed = []
const parallel = new BuddhabrotRunner({
  workerCount: 5, width: 2, height: 2,
  onChunk: () => displayed.push(Array.from(parallel.densityR).reduce((sum, value) => sum + value, 0)),
})
await parallel._workersReady
await parallel.start({ samples: 10, renderDelay: 100, renderPointBatchSize: 1 })
for (const [i, worker] of parallel.workers.entries()) {
  assert.equal(worker.messages.at(-1).waitForPresentation, true)
  worker.onmessage({ data: { type: 'chunk', jobId: parallel._currentJobId, chunk: {
    x: 0, y: 0, w: 2, h: 2, indices: [i % 4], r: [1], g: [0], b: [0], presentationId: i + 1,
  } }, target: worker })
}
assert.deepEqual(displayed, [], 'Batches merged before the display frame')
nextFrame(0)
assert.deepEqual(displayed, [1], 'First frame combined multiple workers')
nextFrame(50)
assert.deepEqual(displayed, [1], 'Render Speed Delay was skipped')
for (const [i, worker] of parallel.workers.entries()) {
  worker.onmessage({ data: { type: 'trajectoryBatchDone', jobId: parallel._currentJobId }, target: worker })
  if (i < 4) nextFrame((i + 1) * 100)
}
assert.deepEqual(displayed, [1, 2, 3, 4, 5])
assert.equal(parallel.workers.flatMap((w) => w.messages).filter((m) => m.cmd === 'presented').length, 5)
const pendingWorker = parallel.workers[0]
pendingWorker.onmessage({ data: { type: 'chunk', jobId: parallel._currentJobId, chunk: {
  x: 0, y: 0, w: 2, h: 2, indices: [0], r: [1], presentationId: 6,
} }, target: pendingWorker })
nextFrame(450)
assert.deepEqual(displayed, [1, 2, 3, 4, 5])
parallel.setRenderSpeed(10)
nextFrame(450)
assert.deepEqual(displayed, [1, 2, 3, 4, 5, 6], 'Live Delay change did not update presentation pacing')
pendingWorker.onmessage({ data: { type: 'chunk', jobId: parallel._currentJobId, chunk: {
  x: 0, y: 0, w: 2, h: 2, indices: [0], r: [1], presentationId: 7,
} }, target: pendingWorker })
parallel.stop()
nextFrame(500)
assert.deepEqual(displayed, [1, 2, 3, 4, 5, 6], 'Stop displayed a queued batch')
assert.equal(frames.size, 0)
await parallel.start({ samples: 10, renderDelay: 0 })
pendingWorker.onmessage({ data: { type: 'chunk', jobId: parallel._currentJobId, chunk: {
  x: 0, y: 0, w: 2, h: 2, indices: [3], r: [2],
} }, target: pendingWorker })
assert.equal(displayed.at(-1), 2, 'Zero Delay did not retain immediate chunk merging')
assert.equal(frames.size, 0, 'Zero Delay used the paced presentation queue')
parallel.terminate()
const order = []
const ordered = new BuddhabrotRunner({ workerCount: 2, width: 3, height: 1, onChunk: (chunk) => order.push(chunk.indices[0]) })
await ordered._workersReady
await ordered.start({ samples: 10, renderDelay: 100, renderPointBatchSize: 4 })
const [workerA, workerB] = ordered.workers
const enqueue = (worker, index, presentationId) => worker.onmessage({ data: {
  type: 'chunk', jobId: ordered._currentJobId,
  chunk: { x: 0, y: 0, w: 3, h: 1, indices: [index], r: [1], presentationId },
}, target: worker })
enqueue(workerA, 0, 1)
enqueue(workerB, 1, 1)
nextFrame(500)
enqueue(workerA, 2, 2)
nextFrame(600)
assert.deepEqual(order, [0, 2], 'A trajectory was delayed by switching workers mid-orbit')
workerA.onmessage({ data: { type: 'trajectoryBatchDone', jobId: ordered._currentJobId }, target: workerA })
nextFrame(700)
assert.deepEqual(order, [0, 2, 1])
enqueue(workerA, 0, 3)
ordered.setRenderSpeed(0)
nextFrame(701)
assert.deepEqual(order, [0, 2, 1, 0], 'Zero Delay left another worker waiting for presentation')
ordered.terminate()
Date.now = originalNow
console.log('PASS: five workers show one batch per frame/Delay, acknowledge each display and discard queued batches on Stop')

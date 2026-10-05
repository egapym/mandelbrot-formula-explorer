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
const coalesced = []
const aggregate = new BuddhabrotRunner({
  workerCount: 2,
  width: 2,
  height: 2,
  onChunk: () => coalesced.push(Array.from(aggregate.densityR).reduce((sum, value) => sum + value, 0)),
})
await aggregate._workersReady
await aggregate.start({ samples: 10, renderDelay: 100, renderPointBatchSize: 2 })
for (const [i, aggregateWorker] of aggregate.workers.entries()) {
  aggregateWorker.onmessage({ data: { type: 'chunk', jobId: aggregate._currentJobId, chunk: {
    x: 0, y: 0, w: 2, h: 2, indices: [i], r: [1], presentationId: i + 1,
  } }, target: aggregateWorker })
}
nextFrame(0)
assert.deepEqual(coalesced, [2], 'Ready workers recolored the canvas more than once')
assert.equal(aggregate.workers.flatMap((w) => w.messages).filter((m) => m.cmd === 'presented').length, 2)
aggregate.terminate()

const parallel = new BuddhabrotRunner({
  workerCount: 5, width: 2, height: 2,
  onChunk: () => displayed.push(Array.from(parallel.densityR).reduce((sum, value) => sum + value, 0)),
})
await parallel._workersReady
await parallel.start({ samples: 10, renderDelay: 100, renderPointBatchSize: 1 })
assert.deepEqual(parallel.workers.map((w) => w.messages.at(-1).trajectorySlots), [1, 0, 0, 0, 0])
const emit = (runner, worker, type, chunk) => worker.onmessage({
  data: { type, jobId: runner._currentJobId, chunk }, target: worker,
})
const chunk = (id, index = 0) => ({
  x: 0, y: 0, w: 2, h: 2, indices: [index], r: [1], presentationId: id,
})
emit(parallel, parallel.workers[0], 'chunk', chunk(1))
nextFrame(0)
assert.deepEqual(displayed, [1], 'Batch 1 must initially draw exactly one point')
emit(parallel, parallel.workers[0], 'done')
assert.deepEqual(parallel._trajectorySlots, [0, 1, 0, 0, 0], 'Finished worker did not immediately release its slot')
parallel.setRenderPointBatchSize(2)
assert.deepEqual(parallel._trajectorySlots, [0, 1, 1, 0, 0])
const [, workerA, workerB] = parallel.workers
emit(parallel, workerA, 'chunk', chunk(1))
nextFrame(10)
emit(parallel, workerB, 'chunk', chunk(1))
nextFrame(50)
emit(parallel, workerA, 'chunk', chunk(2))
emit(parallel, workerB, 'chunk', chunk(2))
nextFrame(109)
assert.deepEqual(displayed, [1, 2, 3])
nextFrame(110)
assert.deepEqual(displayed, [1, 2, 3, 4], 'Worker A waited for worker B')
nextFrame(150)
assert.deepEqual(displayed, [1, 2, 3, 4, 5], 'Worker B did not use its independent Delay clock')
emit(parallel, workerA, 'chunk', chunk(3))
nextFrame(210)
assert.equal(displayed.at(-1), 6, 'Worker A stalled when B had no ready chunk')
emit(parallel, workerA, 'chunk', chunk(4))
parallel.setRenderSpeed(10)
nextFrame(220)
assert.equal(displayed.at(-1), 7)
emit(parallel, workerA, 'chunk', chunk(5))
parallel.stop()
nextFrame(300)
assert.equal(displayed.at(-1), 7, 'Stop displayed a queued batch')
assert.equal(frames.size, 0)
await parallel.start({ samples: 10, renderDelay: 0 })
emit(parallel, parallel.workers[0], 'chunk', { ...chunk(undefined), r: [2] })
assert.equal(displayed.at(-1), 2, 'Zero Delay did not merge immediately')
parallel.terminate()
Date.now = originalNow
console.log('PASS: global trajectory slots, independent worker clocks, immediate slot reuse, live Delay and Stop')

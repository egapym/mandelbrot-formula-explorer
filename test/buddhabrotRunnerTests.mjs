import assert from 'node:assert/strict'

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
await first._workersReady
await first.start({ samples: 10 })
const worker = first.workers[0]
const oldJobId = first._currentJobId
first.terminate()
await new Promise((resolve) => setTimeout(resolve, 5))

const second = makeRunner(() => completed++)
await second._workersReady
assert.equal(second.workers[0], worker, 'idle worker was not reused')
await second.start({ samples: 10 })
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

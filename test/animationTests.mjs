import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  AnimationStore,
  animationErrorMessage,
  createAnimationPath,
  IMAGE_BUDGET,
  PreparedAnimation,
  validateAnimationSize,
} from '../animation.mjs'
import * as fxp from '../fxp.mjs'

const point = (x, y) => [fxp.fromNumber(x), fxp.fromNumber(y)]

test('GPU error details go to console, not the user-facing message', (t) => {
  const log = t.mock.method(console, 'error', () => {})
  const error = new Error('Write range (bufferOffset: 0, size: 8016) does not fit in [Buffer "zr buffer"] size (8008)')
  const animation = new PreparedAnimation()
  animation.reportError(error)
  assert.match(animation.error, /could not be prepared/)
  assert.doesNotMatch(animation.error, /8016|bufferOffset|zr buffer/)
  assert.equal(log.mock.calls[0].arguments[1], error)
  animation.reportError(new DOMException('Stopped', 'AbortError'))
  assert.equal(log.mock.calls.length, 1)
})
const options = {
  startCenter: point(-0.5, 0),
  startZoom: fxp.fromNumber(1),
  targetCenter: point(-0.7, 0.2),
  targetZoom: fxp.fromNumber(1000),
}
const wait = () => new Promise((resolve) => setTimeout(resolve, 0))
const deferred = () => {
  let resolve
  const promise = new Promise((done) => {
    resolve = done
  })
  return { resolve, promise }
}
function memoryStore() {
  return {
    files: [],
    disposed: false,
    async put(i, blob) {
      this.files[i] = blob
    },
    async get(i) {
      return this.files[i]
    },
    async dispose() {
      this.disposed = true
    },
  }
}

test('path preserves pan first, slows through the full ending ramp, exact endpoints and fps', () => {
  const path = createAnimationPath(options)
  assert.equal(path.panMs, 700)
  assert.equal(path.at(0).center[0].toNumber(), -0.5)
  assert.equal(path.at(10).zoom.toNumber(), 1)
  const frame = path.at(30) // 1000ms, including 700ms pan and 300ms zoom
  assert.equal(frame.center[0].toNumber(), options.targetCenter[0].toNumber())
  assert.ok(frame.zoom.toNumber() > 1 && frame.zoom.toNumber() < options.targetZoom.toNumber())
  const framesBeforeEnd = [3, 2, 1].map((frames) => path.at(path.count - 1 - frames).zoom.toNumber())
  const finalZoom = path.at(path.count - 1).zoom.toNumber()
  const finalSteps = [
    framesBeforeEnd[1] - framesBeforeEnd[0],
    framesBeforeEnd[2] - framesBeforeEnd[1],
    finalZoom - framesBeforeEnd[2],
  ]
  assert.ok(finalSteps[2] < finalSteps[1] && finalSteps[1] < finalSteps[0], 'Zoom must decelerate until the target')
  assert.deepEqual(path.at(path.count - 1), { center: options.targetCenter, zoom: options.targetZoom })
  const fast = createAnimationPath({ ...options, speed: 0.3 })
  assert.equal(fast.panMs, path.panMs)
  assert.ok(Math.abs(fast.duration - 700 - (path.duration - 700) / 2) < 1e-9)
  for (const fps of [15, 30, 60]) {
    const p = createAnimationPath({ ...options, fps })
    assert.equal(p.count, Math.ceil((p.duration * fps) / 1000) + 1)
  }
})

test('equal zoom and zoom out remain finite and reach exact target', () => {
  for (const zoom of [1, 0.1]) {
    const path = createAnimationPath({ ...options, targetZoom: fxp.fromNumber(zoom) })
    for (let i = 0; i < path.count; i++) assert.ok(Number.isFinite(path.at(i).zoom.toNumber()))
    assert.equal(path.at(path.count - 1).zoom.toNumber(), fxp.fromNumber(zoom).toNumber())
  }
  assert.throws(() => createAnimationPath({ ...options, speed: 0 }))
})

test('preflight rejects unsafe buffers, reference memory, decoded images and dispatch before allocation', () => {
  const limits = {
    maxBufferSize: 2 ** 28,
    maxStorageBufferBindingSize: 2 ** 27,
    maxComputeWorkgroupsPerDimension: 65535,
  }
  const size = { width: 1920, height: 1080, maxIter: 1000 }
  assert.equal(validateAnimationSize(size, limits), 1920 * 1080 * 4)
  assert.throws(() => validateAnimationSize({ ...size, width: NaN }, limits))
  assert.throws(() => validateAnimationSize({ ...size, width: 100000, height: 100000 }, limits))
  assert.throws(() => validateAnimationSize(size, { ...limits, maxStorageBufferBindingSize: 32 }))
  assert.throws(() => validateAnimationSize(size, { ...limits, maxComputeWorkgroupsPerDimension: 1 }))
  assert.throws(() => validateAnimationSize({ ...size, bitmapBytes: 2 ** 29 }, limits))
  assert.throws(() => validateAnimationSize({ ...size, perturbation: true, maxIter: IMAGE_BUDGET }, limits))
})

test('only fully saved frames advance progress; Play is unavailable until all frames are saved', async () => {
  const gate = deferred()
  const store = memoryStore()
  const observed = []
  const put = store.put.bind(store)
  store.put = async (i, blob) => {
    if (i === 1) await gate.promise
    await put(i, blob)
  }
  const animation = new PreparedAnimation({
    createStore: async () => store,
    changed: (a) => observed.push(a.progress),
    wait,
  })
  const path = { count: 2, at: (i) => i }
  const preparing = animation.prepare(path, async (frame) => frame)
  while (animation.progress < 50) await wait()
  assert.equal(animation.state, 'preparing')
  await animation.play(() => assert.fail('cannot play partial preparation'))
  assert.equal(animation.progress, 50)
  gate.resolve()
  await preparing
  assert.equal(animation.state, 'ready')
  assert.equal(animation.progress, 100)
  assert.deepEqual(store.files, [0, 1])
  assert.ok(observed.includes(50))
})

test('Stop waits for in-flight render, discards partial storage and rejects late output', async () => {
  const gate = deferred()
  const entered = deferred()
  const store = memoryStore()
  const animation = new PreparedAnimation({ createStore: async () => store, wait })
  const preparing = animation.prepare({ count: 2, at: (i) => i }, async () => {
    entered.resolve()
    await gate.promise
    return 'late'
  })
  await entered.promise
  const stopping = animation.stop()
  assert.equal(animation.busy, true)
  assert.equal(store.disposed, false)
  gate.resolve()
  await Promise.all([preparing, stopping])
  assert.equal(animation.state, 'idle')
  assert.equal(animation.busy, false)
  assert.equal(store.disposed, true)
  assert.equal(store.files.length, 0)
  const replacement = memoryStore()
  animation.createStore = async () => replacement
  await animation.prepare({ count: 1, at: () => 7 }, async (x) => x)
  assert.deepEqual(replacement.files, [7])
})

test('GPU or storage failures stop preparation instead of producing a ready animation', async (t) => {
  t.mock.method(console, 'error', () => {})
  for (const failure of ['GPU lost', 'QuotaExceededError']) {
    const store = memoryStore()
    if (failure === 'QuotaExceededError')
      store.put = async () => {
        throw new Error(failure)
      }
    const animation = new PreparedAnimation({ createStore: async () => store, wait })
    await animation.prepare({ count: 2, at: (i) => i }, async () => {
      if (failure === 'GPU lost') throw new Error(failure)
      return 'frame'
    })
    assert.equal(animation.state, 'idle')
    assert.equal(animation.error, animationErrorMessage(new Error(failure)))
    assert.equal(store.disposed, true)
  }
})

test('playback never renders, retains at most two decoded images and restarts after Stop', async () => {
  let alive = 0
  let maximum = 0
  const animation = new PreparedAnimation({
    createStore: async () => memoryStore(),
    wait,
    decode: async (value) => {
      alive++
      maximum = Math.max(maximum, alive)
      return {
        value,
        close() {
          alive--
        },
      }
    },
  })
  await animation.prepare({ count: 4, fps: 60, at: (i) => i }, async (x) => x)
  const first = []
  await animation.play((_image, frame) => {
    first.push(frame)
    if (frame === 1) void animation.stop()
  })
  while (animation.busy) await wait()
  assert.deepEqual(first, [0, 1])
  assert.equal(alive, 0)
  const second = []
  await animation.play((_image, frame) => second.push(frame))
  assert.deepEqual(second, [0, 1, 2, 3])
  assert.equal(alive, 0)
  assert.ok(maximum <= 2)
  await animation.discard()
  assert.equal(animation.state, 'idle')
})

test('storage budget is enforced before opening a file', async () => {
  let opened = false
  const store = new AnimationStore({
    budget: 4,
    directory: {
      getFileHandle() {
        opened = true
      },
    },
  })
  await assert.rejects(store.put(0, new Blob(['12345'])), /storage limit/)
  assert.equal(opened, false)
})

test('Stop during image decoding closes the late bitmap without displaying it', async () => {
  const gate = deferred()
  const entered = deferred()
  let closed = 0
  const animation = new PreparedAnimation({
    createStore: async () => memoryStore(),
    wait,
    decode: async () => {
      entered.resolve()
      await gate.promise
      return {
        close() {
          closed++
        },
      }
    },
  })
  await animation.prepare({ count: 1, fps: 30, at: (i) => i }, async (x) => x)
  const playing = animation.play(() => assert.fail('late image displayed'))
  await entered.promise
  const stopping = animation.stop()
  gate.resolve()
  await Promise.all([playing, stopping])
  assert.equal(closed, 1)
  assert.equal(animation.state, 'ready')
})

test('missing stored frames invalidate playback and release storage', async (t) => {
  t.mock.method(console, 'error', () => {})
  const store = memoryStore()
  const animation = new PreparedAnimation({ createStore: async () => store, wait })
  await animation.prepare({ count: 1, fps: 30, at: (i) => i }, async (x) => x)
  store.get = async () => {
    throw new Error('Stored frame missing')
  }
  await animation.play(() => assert.fail('missing frame displayed'))
  assert.equal(animation.state, 'idle')
  assert.equal(animation.store, null)
  assert.equal(animation.error, animationErrorMessage(new Error('Stored frame missing'), 'play'))
  assert.equal(store.disposed, true)
})

test('write failure aborts the writer without counting an unsaved frame', async () => {
  let aborted = false
  const store = new AnimationStore({
    budget: 1024,
    directory: {
      async getFileHandle() {
        return {
          async createWritable() {
            return {
              async write() {
                throw new Error('Disk full')
              },
              async abort() {
                aborted = true
              },
            }
          },
        }
      },
    },
  })
  await assert.rejects(store.put(0, new Blob(['frame'])), /Disk full/)
  assert.equal(aborted, true)
  assert.equal(store.bytes, 0)
})

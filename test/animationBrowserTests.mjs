import { AnimationStore } from '../animation.mjs'
import { AnimationGpuSession } from '../animationGpuSession.mjs'
import * as fxp from '../fxp.mjs'

const assert = (condition, message) => {
  if (!condition) throw new Error(message)
}
const rejects = async (operation, message) => {
  let rejected = false
  try {
    await operation()
  } catch {
    rejected = true
  }
  assert(rejected, message)
}

async function storageChecks() {
  const store = await AnimationStore.create()
  try {
    await store.put(0, new Blob(['frame']))
    await AnimationStore.cleanup()
    assert((await (await store.get(0)).text()) === 'frame', 'Cleanup deleted an active session')
    const root = await AnimationStore.root()
    const orphan = `session-${crypto.randomUUID()}`
    await root.getDirectoryHandle(orphan, { create: true })
    await AnimationStore.cleanup()
    await rejects(() => root.getDirectoryHandle(orphan), 'Abandoned data was not removed')
    const quotaStore = new AnimationStore({ budget: 1, directory: store.directory })
    await rejects(() => quotaStore.put(1, new Blob(['too large'])), 'Storage quota was not enforced')
  } finally {
    await store.dispose()
  }
  await rejects(() => store.root.getDirectoryHandle(store.name), 'Disposed session was not deleted')
}

function fixture(device, kind) {
  const view = {
    width: 63,
    height: 35,
    max_iter: 64,
    precision: 64,
    requiredPrecision: 32,
    smooth: true,
    supersampling: 2,
    escapeRadius: 4,
    fractalType: kind === 'direct' ? 'custom' : 'mandelbrot',
    iterationFunction: kind === 'direct' ? 'z*z + c + 0.01*zDelay(2)' : 'z*z+c',
    palette: [],
    paletteComponent: { palette: { trapSpec: { mode: 'distance_closest', shape: 'ring', size: 0.5 } } },
    setZoom(zoom) {
      this.zoom = zoom
    },
    setCenter(center) {
      this.center = center
    },
    canvas2complex(x, y) {
      return [fxp.fromNumber(-2 + (x / this.width) * 3, 64), fxp.fromNumber(-1 + (y / this.height) * 2, 64)]
    },
  }
  let created = false
  let received = false
  const session = new AnimationGpuSession({
    view,
    sources: { [kind]: { devicePromise: Promise.resolve(device), available: true } },
    select: () => kind,
    z0: [0, 0],
    colorPatternId: 'cosine_rgb',
    createScreen() {
      created = true
      const canvas = document.createElement('canvas')
      canvas.width = view.width
      canvas.height = view.height
      const screen = {
        canvas,
        offscreen: canvas,
        smoothscreen: canvas,
        renderRgba(rgba) {
          received = rgba.some((value) => value !== 0)
          canvas.getContext('2d').putImageData(new ImageData(rgba, view.width, view.height), 0, 0)
        },
        render() {
          received = this.values.some((value) => value > 4)
        },
      }
      for (const key of ['values', 'smooth', 'signs', 'zreal', 'zimag', 'otData'])
        screen[key] = new Float32Array(view.width * view.height)
      return screen
    },
  })
  return { session, view, created: () => created, received: () => received }
}

async function gpuChecks() {
  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()
  assert(device, 'WebGPU is required; CPU fallback must not pass this test')
  const errors = []
  device.addEventListener('uncapturederror', (event) => errors.push(event.error.message))
  const frame = { zoom: fxp.fromNumber(1), center: [fxp.fromNumber(-0.5), fxp.fromNumber(0)] }
  try {
    for (const kind of ['direct', 'perturbation', 'orbit']) {
      const test = fixture(device, kind)
      try {
        const blob = await test.session.render(frame, new AbortController().signal)
        assert(blob.type === 'image/png' && blob.size > 0 && test.received(), `${kind}: no completed GPU frame`)
        const abort = new AbortController()
        abort.abort()
        await rejects(() => test.session.render(frame, abort.signal), `${kind}: ignored cancellation`)
      } finally {
        test.session.dispose()
      }
    }
    const limited = new Proxy(device, {
      get(target, key) {
        if (key === 'limits')
          return { maxBufferSize: 32, maxStorageBufferBindingSize: 32, maxComputeWorkgroupsPerDimension: 1 }
        const value = Reflect.get(target, key, target)
        return typeof value === 'function' ? value.bind(target) : value
      },
    })
    const oversized = fixture(limited, 'direct')
    try {
      await rejects(() => oversized.session.render(frame, new AbortController().signal), 'GPU buffer limit was ignored')
      assert(!oversized.created(), 'Screen allocated before buffer validation')
    } finally {
      oversized.session.dispose()
    }
    const failed = fixture(device, 'direct')
    try {
      const { renderer } = await failed.session.rendererFor('direct')
      renderer.pipeline.getPipeline = async () => {
        throw new Error('Injected GPU pipeline failure')
      }
      await rejects(
        () => failed.session.render(frame, new AbortController().signal),
        'GPU pipeline failure silently succeeded',
      )
    } finally {
      failed.session.dispose()
    }
    assert(errors.length === 0, errors.join('\n'))
    const lost = fixture(device, 'direct')
    try {
      await lost.session.render(frame, new AbortController().signal)
      device.destroy()
      await device.lost
      await rejects(() => lost.session.render(frame, new AbortController().signal), 'Lost device silently succeeded')
    } finally {
      lost.session.dispose()
    }
  } finally {
    device.destroy()
  }
}

try {
  await storageChecks()
  await gpuChecks()
  document.querySelector('#result').textContent =
    'PASS: OPFS roundtrip, active lock protection, orphan cleanup, quota, direct/history, perturbation, Orbit Trap, supersampling, cancellation, allocation preflight, pipeline failure, device loss'
  document.documentElement.dataset.result = 'passed'
} catch (error) {
  document.querySelector('#result').textContent = error.stack
  document.documentElement.dataset.result = 'failed'
  console.error(error)
}

import { AnimationStore, analyzeAnimationIterations, analyzeAnimationPath, createAnimationPath } from '../animation.mjs'
import { AnimationGpuSession } from '../animationGpuSession.mjs'
import * as fxp from '../fxp.mjs'
import { getPalette, initPallet } from '../palette.js'

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

function fixture(device, kind, { maxIter = 64 } = {}) {
  const palette = kind === 'orbit' ? Object.assign(Object.create(getPalette('blue_gold')), {
    trapSpec: { mode: 'distance_closest', shape: 'ring', size: 0.5 },
  }) : getPalette('blue_gold')
  const view = {
    width: 63,
    height: 35,
    max_iter: maxIter,
    precision: 64,
    requiredPrecision: 32,
    smooth: true,
    supersampling: 2,
    escapeRadius: 4,
    fractalType: kind === 'direct' ? 'custom' : 'mandelbrot',
    iterationFunction: kind === 'direct' ? 'z*z + c + 0.01*zDelay(2)' : 'z*z+c',
    palette: initPallet(palette, 20, 0, undefined, maxIter),
    paletteComponent: { palette, density: '20', rotate: '0' },
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
  const targetComponent = view.paletteComponent
  const session = new AnimationGpuSession({
    view,
    sources: Object.fromEntries(['direct', 'perturbation', 'orbit'].map((name) =>
      [name, { devicePromise: Promise.resolve(device), available: true }])),
    select: (_view, { analysis = false } = {}) => analysis && kind === 'orbit' ? 'direct' : kind,
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
        render(lookup, _maxIter, smooth, paletteObj) {
          received = this.values.some((value) => value > 4)
          const base = new ImageData(view.width, view.height)
          const overlay = new ImageData(view.width, view.height)
          paletteObj.renderPixels(base.data, overlay.data, this.values, this.smooth, this.signs, lookup,
            smooth, this.zreal, this.zimag, this.otData)
          const context = canvas.getContext('2d')
          context.putImageData(base, 0, 0)
          if (smooth && paletteObj.supportsSmooth) {
            const temporary = document.createElement('canvas')
            temporary.width = view.width
            temporary.height = view.height
            temporary.getContext('2d').putImageData(overlay, 0, 0)
            context.drawImage(temporary, 0, 0)
          }
        },
      }
      for (const key of ['values', 'smooth', 'signs', 'zreal', 'zimag', 'otData'])
        screen[key] = new Float32Array(view.width * view.height)
      return screen
    },
  })
  return { session, view, targetComponent, created: () => created, received: () => received }
}

const imageHash = async (blob) => {
  const bitmap = await createImageBitmap(blob)
  try {
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const context = canvas.getContext('2d')
    context.drawImage(bitmap, 0, 0)
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data
    return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', pixels)), (byte) => byte.toString(16).padStart(2, '0')).join('')
  } finally { bitmap.close() }
}

async function adjustmentChecks(device) {
  const metrics = []
  for (const name of ['mandelbrot', 'history', 'deep', 'stripe', 'grid', 'orbit']) {
    document.querySelector('#result').textContent = `Checking adjustment: ${name}`
    const kind = name === 'deep' ? 'perturbation' : name === 'orbit' ? 'orbit' : 'direct'
    const targetMaxIter = name === 'deep' ? 1200 : 512
    const test = fixture(device, kind, { maxIter: targetMaxIter })
    const { view, session } = test
    view.width = 320
    view.height = 180
    view.precision = 96
    view.requiredPrecision = name === 'deep' ? 80 : 32
    if (name !== 'history') view.iterationFunction = 'z*z+c'
    if (name === 'stripe' || name === 'grid') view.paletteComponent.palette = getPalette(name)
    view.palette = initPallet(view.paletteComponent.palette, 20, 0, undefined, targetMaxIter)
    view.canvas2complex = function (x, y) {
      const scale = fxp.fromNumber(this.width / 4, this.precision).multiply(this.zoom.withScale(this.precision))
      return [x - this.width / 2, y - this.height / 2].map((offset, i) =>
        this.center[i].withScale(this.precision).add(fxp.fromNumber(offset, this.precision).divide(scale)))
    }
    const center = name === 'deep' ? [0.2500098571777344, 0] : [-0.745, 0.186]
    const targetCenter = center.map((v) => fxp.fromNumber(v, 96))
    const startZoom = fxp.fromNumber(1, 96)
    const targetZoom = fxp.fromNumber(name === 'deep' ? 1e10 : 100, 96)
    const targetFrame = { center: targetCenter, zoom: targetZoom }
    const controller = new AbortController()
    try {
      const baseline = await imageHash(await session.render(targetFrame, controller.signal))
      const path = createAnimationPath({ startCenter: targetCenter, startZoom, targetCenter, targetZoom, speed: 0.6, fps: 15 })
      let probes = 0
      let targetStats
      const adjusted = await analyzeAnimationPath({ path, startZoom, targetZoom, targetCenter,
        minimum: '16', targetMaxIter, targetDensity: 20 }, async (frame, signal) => {
        const answer = await session.probe(frame, signal)
        assert(answer.values.length === 192 * 108, `${name}: incorrect probe resolution`)
        probes++
        if (probes === 17) targetStats = analyzeAnimationIterations(answer.values)
        return answer
      }, controller.signal, () => {})
      assert(probes === 17, `${name}: incorrect analysis count`)
      assert(view.smooth && view.supersampling === 2, `${name}: probes changed final quality`)
      const first = adjusted.at(0)
      assert(first.maxIter === 16 && first.paletteDensity === 20, `${name}: incorrect start settings`)
      await session.render(first, controller.signal)
      const middle = adjusted.at(Math.floor(adjusted.count / 2))
      await session.render(middle, controller.signal)
      const final = adjusted.at(adjusted.count - 1)
      assert(final.maxIter === targetMaxIter && final.paletteDensity === 20, `${name}: incorrect target settings`)
      assert(await imageHash(await session.render(final, controller.signal)) === baseline, `${name}: final image changed`)
      assert(await imageHash(await session.render(targetFrame, controller.signal)) === baseline, `${name}: OFF image changed`)
      assert(test.targetComponent.density === '20' && test.targetComponent !== view.paletteComponent,
        `${name}: target component changed`)
      // Measure actual intermediate truncation against the same-depth target
      // budget, without supersampling averaging in-set and escaped markers.
      view.smooth = false
      view.supersampling = 0
      const reference = await session.compute({ center: middle.center, zoom: middle.zoom }, controller.signal)
      const result = await session.compute(middle, controller.signal)
      let escaped = 0
      let lost = 0
      for (let i = 0; i < reference.values?.length; i++) {
        if (reference.values[i] >= 4) { escaped++; if (result.values[i] < 4) lost++ }
      }
      const referenceStats = reference.values && analyzeAnimationIterations(reference.values)
      const currentStats = result.values && analyzeAnimationIterations(result.values)
      metrics.push({ name, iterations: middle.maxIter, density: middle.paletteDensity,
        escaped, lost, lossFraction: escaped ? lost / escaped : null,
        bandWidth: currentStats ? 2 ** (middle.paletteDensity / 10) * currentStats.spread : null,
        referenceBandWidth: referenceStats ? 2 ** (20 / 10) * referenceStats.spread : null,
        targetBandWidth: targetStats ? 2 ** (20 / 10) * targetStats.spread : null })
    } finally { session.dispose() }
  }
  globalThis.animationAdjustmentMetrics = metrics
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
    await adjustmentChecks(device)
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
    'PASS: OPFS roundtrip, active lock protection, orphan cleanup, quota, direct/history, perturbation, Orbit Trap, supersampling, analysis, adaptive iterations with target density, target/OFF image parity, intermediate metrics, cancellation, allocation preflight, pipeline failure, device loss'
  document.documentElement.dataset.result = 'passed'
} catch (error) {
  document.querySelector('#result').textContent = error.stack
  document.documentElement.dataset.result = 'failed'
  console.error(error)
}

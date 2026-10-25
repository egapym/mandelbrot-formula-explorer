import { PreparedAnimation } from '../animation.mjs'
import { AnimationGpuSession } from '../animationGpuSession.mjs'
import { AnimationBuddhabrotSession } from '../animationBuddhabrotSession.mjs'
import { createBuddhaRunner } from '../buddhabrot.mjs'
import { buildBuddhaStops } from '../buddhaPalettes.mjs'
import { getPalette, initPallet } from '../palette.js'
import * as fxp from '../fxp.mjs'

const output = document.getElementById('result')
const params = new URLSearchParams(location.search)
const metrics = { cases: [], errors: [] }
globalThis.animationPrepareMetrics = metrics
const stats = { maps: 0, readBytes: 0, pipelines: 0, buffers: 0, references: 0 }
const adapter = await navigator.gpu.requestAdapter()
const device = await adapter.requestDevice()
device.addEventListener('uncapturederror', (e) => metrics.errors.push(e.error.message))
const trackedDevice = new Proxy(device, {
  get(target, key) {
    if (key === 'createBuffer') return (options) => {
      stats.buffers++
      const buffer = target.createBuffer(options)
      const map = buffer.mapAsync.bind(buffer)
      buffer.mapAsync = (...args) => { stats.maps++; stats.readBytes += args[2] ?? buffer.size; return map(...args) }
      return buffer
    }
    if (key === 'createComputePipeline') return (...args) => { stats.pipelines++; return target.createComputePipeline(...args) }
    const value = Reflect.get(target, key, target)
    return typeof value === 'function' ? value.bind(target) : value
  },
})
function makeView(width, height, kind, maxIter) {
  const palette = getPalette('blue_gold')
  const precision = 96
  return {
    width, height, precision, requiredPrecision: kind === 'deep' ? 80 : 32,
    max_iter: maxIter, smooth: true, supersampling: 0, escapeRadius: 4,
    fractalType: 'mandelbrot', iterationFunction: 'z*z+c',
    palette: initPallet(palette, 20, 0, undefined, maxIter),
    paletteComponent: { palette, density: '20', rotate: '0' },
    setZoom(zoom) { this.zoom = zoom }, setCenter(center) { this.center = center },
    canvas2complex(x, y) {
      const scale = fxp.fromNumber(this.width / 4, this.precision).multiply(this.zoom.withScale(this.precision))
      return [x - this.width / 2, y - this.height / 2].map((offset, i) =>
        this.center[i].withScale(this.precision).add(fxp.fromNumber(offset, this.precision).divide(scale)))
    },
  }
}
function makeScreen(view) {
  const canvas = document.createElement('canvas')
  canvas.width = view.width; canvas.height = view.height
  const overlayCanvas = document.createElement('canvas')
  overlayCanvas.width = view.width; overlayCanvas.height = view.height
  const context = canvas.getContext('2d')
  const overlayContext = overlayCanvas.getContext('2d')
  const base = context.createImageData(view.width, view.height)
  const overlay = context.createImageData(view.width, view.height)
  const screen = {
    canvas, offscreen: overlayCanvas, smoothscreen: overlayCanvas,
    renderRgba(rgba) { base.data.set(rgba); context.putImageData(base, 0, 0) },
    render(lookup, _maxIter, smooth, palette) {
      palette.renderPixels(base.data, overlay.data, this.values, this.smooth, this.signs, lookup,
        smooth, this.zreal, this.zimag, this.otData)
      context.putImageData(base, 0, 0)
      if (smooth && palette.supportsSmooth) { overlayContext.putImageData(overlay, 0, 0); context.drawImage(overlayCanvas, 0, 0) }
    },
  }
  for (const name of ['values', 'smooth', 'signs', 'zreal', 'zimag', 'otData'])
    screen[name] = new Float32Array(view.width * view.height)
  return screen
}
async function pixelHash(blob) {
  const bitmap = await createImageBitmap(blob)
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width; canvas.height = bitmap.height
  const context = canvas.getContext('2d')
  context.drawImage(bitmap, 0, 0); bitmap.close()
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', pixels)), b => b.toString(16).padStart(2, '0')).join('')
}
async function run(name, repeat) {
  const cpu = name === 'buddha-cpu'
  const deep = name === 'deep'
  const width = deep ? 320 : 640, height = deep ? 180 : 360
  const maxIter = deep ? 4096 : 512
  const count = deep ? 12 : 24
  const view = makeView(width, height, name, maxIter)
  const path = { count, fps: 30, at: (index) => ({
    center: [deep ? -0.743643887037151 : -0.745, deep ? 0.13182590420533 : 0.186].map(v => fxp.fromNumber(v, 96)),
    zoom: fxp.fromNumber(deep ? 1e9 * 1.12 ** index : 1.08 ** index, 96),
  }) }
  let session
  if (name.startsWith('buddha')) {
    session = new AnimationBuddhabrotSession({ view, workerCount: 4,
      params: { width, height, samples: cpu ? 32768 : 65536, maxIter, gpu: !cpu,
        fractalType: 'mandelbrot', iterationFunction: 'z*z+c', escapeRadius: 4,
        palette: { bands: buildBuddhaStops('Preset-01') }, mode: 'buddha', buddhaBandMode: 'perTrajectory',
        brightness: 1, gamma: 1.5, sampleSeed: 123456,
        samplingCenter: { x: -0.5, y: 0 }, samplingZoom: 1, samplingAspect: height / width },
      samplingRange: () => ({}),
      createRunner: options => createBuddhaRunner({ ...options, devicePromise: Promise.resolve(trackedDevice) }),
    })
  } else session = new AnimationGpuSession({ view,
    sources: Object.fromEntries(['direct', 'perturbation', 'orbit'].map(k => [k, { available: true, devicePromise: Promise.resolve(trackedDevice) }])),
    select: () => deep ? 'perturbation' : 'direct', z0: [0, 0], colorPatternId: 'cosine_rgb',
    createScreen: () => makeScreen(view),
  })
  if (deep) {
    const { renderer } = await session.rendererFor('perturbation')
    const calculate = renderer.mandelbrot_high_precision.bind(renderer)
    renderer.mandelbrot_high_precision = (...args) => { stats.references++; return calculate(...args) }
  }
  // The benchmark shares a device across cases; session disposal must not own it.
  const originalDispose = session.dispose.bind(session)
  session.dispose = () => {
    if (session.runner?.devicePromise) session.runner.devicePromise = null
    originalDispose()
  }
  const animation = new PreparedAnimation()
  const before = { ...stats }
  const started = Date.now()
  try {
    const timeout = setTimeout(() => { metrics.errors.push(`${name}: preparation timed out`); animation.stop() }, 60000)
    try {
      await animation.prepare(path, (frame, signal) =>
        session.prepareFrame ? session.prepareFrame(frame, signal) : session.render(frame, signal))
    } finally { clearTimeout(timeout) }
    const elapsed = Date.now() - started
    if (animation.state !== 'ready') throw new Error(animation.error || 'Animation did not finish')
    const hashes = []
    for (let i = 0; i < count; i++) hashes.push(await pixelHash(await animation.store.get(i)))
    const entry = { name, repeat, width, height, maxIter, frames: count, elapsed, bytes: animation.store.bytes,
      ...Object.fromEntries(Object.entries(stats).map(([key, value]) => [key, value - before[key]])), hashes }
    metrics.cases.push(entry)
    output.textContent = JSON.stringify(metrics, null, 2)
  } finally { session.dispose(); await animation.discard() }
}
try {
  const cases = (params.get('case') || 'normal,buddha-gpu,buddha-cpu,deep').split(',')
  for (const name of cases) for (let i = 0; i < Number(params.get('repeats') || 2); i++) await run(name, i)
  if (metrics.errors.length) throw new Error(metrics.errors.join('\n'))
  document.documentElement.dataset.result = 'passed'
  output.textContent = `PASS\n${JSON.stringify(metrics, null, 2)}`
} catch (error) {
  metrics.errors.push(error.stack)
  output.textContent = JSON.stringify(metrics, null, 2)
  document.documentElement.dataset.result = 'failed'
} finally { device.destroy() }

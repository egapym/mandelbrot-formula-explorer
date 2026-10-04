import * as fxp from './fxp.mjs'

export const IMAGE_BUDGET = 128 * 1024 ** 2
export const STORAGE_BUDGET = 2 * 1024 ** 3
// Animation tuning: increase DECELERATION_MS to make the destination approach
// gentler, or MAX_LOG_RATE_PER_MS to raise the maximum default zoom speed.
export const ZOOM_PROFILE = {
  MAX_LOG_RATE_PER_MS: 0.002,
  ACCELERATION_MS: 150,
  DECELERATION_MS: 600,
  DEFAULT_SPEED: 0.15,
}
const aborted = () => new DOMException('Animation stopped', 'AbortError')
export function checkSignal(signal) {
  if (signal?.aborted) throw signal.reason || aborted()
}

export const ANALYSIS_SAMPLES = 17
export const ANALYSIS_LONG_EDGE = 192
const clamp = (value, min, max) => Math.max(min, Math.min(max, value))
const smoothstep = (t) => t * t * (3 - 2 * t)

export function resolveAnimationMinimum(value, targetMaxIter) {
  const minimum = String(value ?? '').trim() === '' ? 1000 : Number(value)
  if (![minimum, targetMaxIter].every((v) => Number.isSafeInteger(v) && v > 0))
    throw new Error('Minimum iterations must be a positive safe integer')
  return Math.min(minimum, targetMaxIter)
}

// Probe shaders run without smoothing or supersampling. Values 0..3 are
// skipped/in-set markers; an escaped value stores the zero-based iteration + 4.
export function analyzeAnimationIterations(values) {
  const escaped = Array.from(values)
    .filter((v) => Number.isFinite(v) && v >= 4)
    .map((v) => v - 3)
  if (escaped.length < 32) return null
  escaped.sort((a, b) => a - b)
  const quantile = (p) => {
    const position = p * (escaped.length - 1)
    const lower = Math.floor(position)
    return escaped[lower] + (escaped[Math.ceil(position)] - escaped[lower]) * (position - lower)
  }
  return { count: escaped.length, high: quantile(0.999), spread: quantile(0.95) - quantile(0.05) }
}

// Equal-weight isotonic regression. All candidates are bounded by the fixed
// endpoints, so pooling adjacent violations cannot move either endpoint.
function monotoneCurve(candidates) {
  const blocks = []
  for (const value of candidates) {
    blocks.push({ sum: value, count: 1 })
    while (blocks.length > 1) {
      const b = blocks[blocks.length - 1]
      const a = blocks[blocks.length - 2]
      if (a.sum / a.count <= b.sum / b.count) break
      blocks.splice(-2, 2, { sum: a.sum + b.sum, count: a.count + b.count })
    }
  }
  const values = blocks.flatMap(({ sum, count }) => Array(count).fill(sum / count))
  const differences = values.slice(1).map((value, i) => value - values[i])
  // Harmonic-mean interior tangents (uniform-grid PCHIP), with stationary
  // endpoints to enter/leave the zoom without a sudden parameter change.
  const slopes = values.map((_, i) => {
    if (!i || i === values.length - 1) return 0
    const a = differences[i - 1]
    const b = differences[i]
    return a > 0 && b > 0 ? (2 * a * b) / (a + b) : 0
  })
  return (progress) => {
    if (progress <= 0) return values[0]
    if (progress >= 1) return values[values.length - 1]
    const position = progress * (values.length - 1)
    const i = Math.floor(position)
    const t = position - i
    const t2 = t * t
    const t3 = t2 * t
    const value =
      (2 * t3 - 3 * t2 + 1) * values[i] +
      (t3 - 2 * t2 + t) * slopes[i] +
      (-2 * t3 + 3 * t2) * values[i + 1] +
      (t3 - t2) * slopes[i + 1]
    return clamp(value, values[i], values[i + 1])
  }
}

export function createAnimationAdjustmentProfile({
  minimum = '', targetMaxIter, targetDensity, startZoom, targetZoom, samples,
}) {
  const minIter = resolveAnimationMinimum(minimum, targetMaxIter)
  if (!Number.isFinite(targetDensity)) throw new Error('Animation requires finite palette density')
  const startLog = Math.log(startZoom.toNumber())
  const targetLog = Math.log(targetZoom.toNumber())
  const delta = targetLog - startLog
  if (![startLog, targetLog].every(Number.isFinite)) throw new Error('Animation requires finite positive zoom')
  const targetSettings = { maxIter: targetMaxIter, paletteDensity: targetDensity }
  if (delta === 0) return { atZoom: () => ({ ...targetSettings }) }
  if (!samples || samples.length < 2) throw new Error('Animation requires analysis samples')
  const iterations = []
  samples.forEach((sample, i) => {
    const fraction = smoothstep(i / (samples.length - 1))
    iterations.push(
      sample && sample.spread > 0
        ? clamp(Math.ceil(sample.high + sample.high / 10), minIter, targetMaxIter)
        : minIter + (targetMaxIter - minIter) * fraction,
    )
  })
  iterations[0] = minIter
  iterations[iterations.length - 1] = targetMaxIter
  const iterationAt = monotoneCurve(iterations)
  return {
    atZoom(zoom) {
      const progress = clamp((Math.log(zoom.toNumber()) - startLog) / delta, 0, 1)
      if (progress >= 1) return { ...targetSettings }
      // The first pan frame must have the same palette density as the target.
      // Keeping it constant also avoids a density discontinuity when zoom begins.
      return { maxIter: Math.ceil(iterationAt(progress)), paletteDensity: targetDensity }
    },
  }
}

export function withAnimationAdjustment(path, profile) {
  return {
    ...path,
    at(index) {
      const frame = path.at(index)
      return { ...frame, ...profile.atZoom(frame.zoom) }
    },
  }
}

export async function analyzeAnimationPath(
  { path, startZoom, targetZoom, targetCenter, minimum, targetMaxIter, targetDensity },
  probe, signal, report,
) {
  const samples = []
  const startLog = Math.log(startZoom.toNumber())
  const delta = Math.log(targetZoom.toNumber()) - startLog
  if (delta !== 0) {
    for (let i = 0; i < ANALYSIS_SAMPLES; i++) {
      checkSignal(signal)
      const zoom = i === 0 ? startZoom : i === ANALYSIS_SAMPLES - 1 ? targetZoom : fxp.fromNumber(
        Math.exp(startLog + delta * i / (ANALYSIS_SAMPLES - 1)),
        Math.max(startZoom.scale, targetZoom.scale),
      )
      const result = await probe({ center: [...targetCenter], zoom }, signal)
      checkSignal(signal)
      samples.push(analyzeAnimationIterations(result.values))
      report((i + 1) / ANALYSIS_SAMPLES)
    }
  }
  const profile = createAnimationAdjustmentProfile({ minimum, targetMaxIter, targetDensity, startZoom, targetZoom, samples })
  return withAnimationAdjustment(path, profile)
}

export function animationErrorMessage(error, phase = 'prepare') {
  const detail = `${error?.name || ''} ${error?.message || error || ''}`
  if (/quota|storage.*(limit|space)|disk full/i.test(detail))
    return 'There is not enough storage space for this animation. Free up browser storage or reduce the frame rate, then prepare again.'
  if (/device.*lost|gpu lost/i.test(detail))
    return 'The connection to the graphics device was interrupted. Reload the page and prepare the animation again.'
  if (/memory|buffer.*limit|resolution.*budget|reference.*capacity|dispatch.*limit/i.test(detail))
    return 'These settings exceed the available graphics resources. Lower the resolution or iteration count, then prepare again.'
  if (/OPFS|Web Locks|WebGPU.*unavailable|cannot be rendered with WebGPU/i.test(detail))
    return 'Animation is unavailable with the current browser or rendering settings. Check GPU support and try again.'
  if (phase === 'play') return 'The animation could not be played. Please prepare it again.'
  if (phase === 'cleanup') return 'Temporary animation files could not be removed. Reload the page and try again.'
  return 'The animation could not be prepared. Please try again. If the problem continues, reload the page.'
}

// Sample the original pan and logarithmic acceleration profile independently
// of rendering speed. Both endpoints are explicit, including a zero-length zoom.
export function createAnimationPath({ startCenter, startZoom, targetCenter, targetZoom, speed = 0.15, fps = 30 }) {
  const a = startZoom.toNumber()
  const b = targetZoom.toNumber()
  if (![a, b, speed, fps].every(Number.isFinite) || a <= 0 || b <= 0 || speed <= 0 || fps <= 0) {
    throw new Error('Animation requires finite positive zoom, speed and frame rate')
  }
  const distance = Math.hypot(...targetCenter.map((v, i) => (v.toNumber() - startCenter[i].toNumber()) * a))
  if (!Number.isFinite(distance)) throw new Error('Animation coordinates are outside the supported range')
  const baseDuration = Math.floor(2000 / 0.15)
  const panMs = Math.min(Math.max(700, (distance / 800) * 1000), baseDuration * 0.9)
  const delta = Math.log(b) - Math.log(a)
  const amount = Math.abs(delta)
  const { MAX_LOG_RATE_PER_MS, ACCELERATION_MS, DECELERATION_MS, DEFAULT_SPEED } = ZOOM_PROFILE
  // The area beneath the rate curve is the log-zoom distance. This duration
  // keeps the current maximum speed and reserves a long deceleration instead
  // of reaching the target during the cruising phase and stopping abruptly.
  const profileMs = Math.max(
    ACCELERATION_MS + DECELERATION_MS,
    amount / MAX_LOG_RATE_PER_MS + (ACCELERATION_MS + DECELERATION_MS) / 2,
  )
  const peakRate = amount === 0 ? 0 : amount / (profileMs - (ACCELERATION_MS + DECELERATION_MS) / 2)
  const areaAt = (time) => {
    const t = Math.max(0, Math.min(profileMs, time))
    if (t < ACCELERATION_MS) return (peakRate * t * t) / (2 * ACCELERATION_MS)
    const cruiseEnd = profileMs - DECELERATION_MS
    if (t < cruiseEnd) return peakRate * (t - ACCELERATION_MS / 2)
    const tail = t - cruiseEnd
    return peakRate * (cruiseEnd - ACCELERATION_MS / 2 + tail - (tail * tail) / (2 * DECELERATION_MS))
  }
  const zoomMs = amount === 0 ? 0 : (profileMs * DEFAULT_SPEED) / speed
  const duration = panMs + zoomMs
  const count = Math.ceil((duration * fps) / 1000) + 1
  if (!Number.isSafeInteger(count) || count > 100000) throw new Error('Animation has too many frames')
  const scale = Math.max(
    startZoom.scale,
    targetZoom.scale,
    ...targetCenter.map((v) => v.scale),
    ...startCenter.map((v) => v.scale),
  )
  const interpolate = (from, to, t) =>
    from.withScale(scale).add(to.withScale(scale).subtract(from.withScale(scale)).multiply(fxp.fromNumber(t, scale)))
  return {
    duration,
    count,
    fps,
    panMs,
    at(index) {
      if (index >= count - 1) return { center: [...targetCenter], zoom: targetZoom }
      const time = Math.min(duration, (index * 1000) / fps)
      if (time <= panMs)
        return { center: startCenter.map((v, i) => interpolate(v, targetCenter[i], time / panMs)), zoom: startZoom }
      const fraction = amount ? Math.min(1, areaAt(((time - panMs) * speed) / DEFAULT_SPEED) / amount) : 1
      return { center: [...targetCenter], zoom: fxp.fromNumber(Math.exp(Math.log(a) + delta * fraction), scale) }
    },
  }
}

export function validateAnimationSize({ width, height, maxIter, bitmapBytes = 4, perturbation = false }, limits) {
  const bytes = width * height * 4
  if (![width, height, maxIter].every((v) => Number.isSafeInteger(v) && v > 0) || !Number.isSafeInteger(bytes)) {
    throw new Error('Invalid animation dimensions or iteration count')
  }
  // Two decoded images may coexist during playback (current and next).
  if (bytes * 2 > IMAGE_BUDGET) throw new Error('Animation resolution exceeds the 128 MiB image budget')
  const sizes = [bytes, bitmapBytes, ...(perturbation ? [(maxIter + 2) * 8] : [])]
  if (
    sizes.some(
      (size) => !Number.isSafeInteger(size) || size > limits.maxBufferSize || size > limits.maxStorageBufferBindingSize,
    )
  ) {
    throw new Error('Animation exceeds this GPU’s buffer limits')
  }
  if (
    Math.ceil(width / 16) > limits.maxComputeWorkgroupsPerDimension ||
    Math.ceil(height / 8) > limits.maxComputeWorkgroupsPerDimension ||
    (perturbation && Math.ceil(Math.min(width * height, 2 ** 18) / 64) > limits.maxComputeWorkgroupsPerDimension)
  ) {
    throw new Error('Animation exceeds this GPU’s dispatch limits')
  }
  if (perturbation && (maxIter + 2) * 64 > IMAGE_BUDGET)
    throw new Error('Animation reference calculation exceeds the memory budget')
  return bytes
}

// Each directory is protected by a Web Lock for its entire lifetime. Cleanup
// only removes directories whose lock can be acquired, including crashed tabs.
export class AnimationStore {
  static supported() {
    return !!(globalThis.navigator?.storage?.getDirectory && globalThis.navigator?.locks)
  }

  static async root() {
    if (!AnimationStore.supported())
      throw new Error('Animation requires browser temporary storage (OPFS and Web Locks)')
    return (await navigator.storage.getDirectory()).getDirectoryHandle('fractal-animation-v1', { create: true })
  }

  static async cleanup() {
    const root = await AnimationStore.root()
    for await (const [name] of root.entries()) {
      if (!name.startsWith('session-')) continue
      await navigator.locks.request(`fractal-animation:${name}`, { ifAvailable: true }, async (lock) => {
        if (lock) await root.removeEntry(name, { recursive: true })
      })
    }
  }

  static async create() {
    const root = await AnimationStore.root()
    const { quota = 0, usage = 0 } = await navigator.storage.estimate()
    const budget = Math.min(STORAGE_BUDGET, Math.floor(Math.max(0, quota - usage) * 0.8))
    if (!budget) throw new Error('No temporary storage space is available')
    const name = `session-${crypto.randomUUID()}`
    let unlock
    let acquired
    const ready = new Promise((resolve) => {
      acquired = resolve
    })
    const hold = new Promise((resolve) => {
      unlock = resolve
    })
    const lockTask = navigator.locks.request(`fractal-animation:${name}`, async () => {
      acquired()
      await hold
    })
    await Promise.race([ready, lockTask])
    try {
      const directory = await root.getDirectoryHandle(name, { create: true })
      return new AnimationStore({ root, name, directory, budget, unlock, lockTask })
    } catch (error) {
      unlock()
      await lockTask
      throw error
    }
  }

  constructor(options) {
    Object.assign(this, options)
    this.bytes = 0
  }
  async put(index, blob) {
    if (this.bytes + blob.size > this.budget) throw new Error('Animation temporary storage limit reached')
    const file = await this.directory.getFileHandle(`${index}.png`, { create: true })
    const writer = await file.createWritable()
    try {
      await writer.write(blob)
      await writer.close()
    } catch (error) {
      await writer.abort().catch(() => {})
      throw error
    }
    this.bytes += blob.size
  }
  async get(index) {
    return (await this.directory.getFileHandle(`${index}.png`)).getFile()
  }
  async dispose() {
    try {
      await this.root.removeEntry(this.name, { recursive: true })
    } finally {
      this.unlock()
      await this.lockTask
    }
  }
}

export class PreparedAnimation {
  constructor({
    changed = () => {},
    createStore = () => AnimationStore.create(),
    decode = (blob) => createImageBitmap(blob),
    wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  } = {}) {
    Object.assign(this, { changed, createStore, decode, wait })
    this.state = 'idle'
    this.progress = 0
    this.phase = ''
    this.generation = 0
    this.operation = Promise.resolve()
  }
  notify() {
    this.changed(this)
  }
  reportError(error, phase = 'prepare') {
    if (error?.name === 'AbortError') return
    console.error(`[Animation: ${phase}]`, error)
    this.error = animationErrorMessage(error, phase)
  }
  get busy() {
    return this.state === 'preparing' || this.state === 'playing' || this.stopping
  }
  async discard() {
    await this.stop()
    const store = this.store
    this.store = null
    this.path = null
    this.state = 'idle'
    this.progress = 0
    this.phase = ''
    if (store) await store.dispose()
    this.notify()
  }
  async stop() {
    this.generation++
    this.controller?.abort(aborted())
    if (this.state !== 'preparing' && this.state !== 'playing' && !this.stopping) return
    this.stopping = true
    this.notify()
    await this.operation
    this.stopping = false
    this.notify()
  }
  async prepare(path, render, { analyze } = {}) {
    await this.discard()
    const generation = ++this.generation
    const controller = new AbortController()
    this.controller = controller
    this.state = 'preparing'
    this.progress = 0
    this.phase = analyze ? 'analysis' : 'frames'
    this.error = ''
    this.notify()
    this.operation = (async () => {
      let store
      try {
        store = await this.createStore()
        checkSignal(controller.signal)
        if (analyze) {
          path = await analyze(controller.signal, (progress) => {
            checkSignal(controller.signal)
            this.progress = clamp(progress, 0, 1) * 10
            this.notify()
          })
          checkSignal(controller.signal)
          this.phase = 'frames'
          this.progress = 10
          this.notify()
        }
        for (let i = 0; i < path.count; i++) {
          const blob = await render(path.at(i), controller.signal)
          checkSignal(controller.signal)
          await store.put(i, blob)
          checkSignal(controller.signal)
          this.progress = (analyze ? 10 : 0) + ((i + 1) / path.count) * (analyze ? 90 : 100)
          this.notify()
          // GPU readback, PNG encoding and OPFS writes already yield to the
          // browser. Avoid an additional timer delay on every prepared frame.
        }
        if (generation !== this.generation) throw aborted()
        this.store = store
        this.path = path
        store = null
        this.state = 'ready'
      } catch (error) {
        this.state = 'idle'
        this.progress = 0
        this.reportError(error)
      } finally {
        this.phase = ''
        if (store)
          await store.dispose().catch((error) => {
            this.reportError(error, 'cleanup')
          })
        this.notify()
      }
    })()
    await this.operation
  }
  async play(display) {
    if (this.busy || !this.store || !this.path) return
    const generation = ++this.generation
    const controller = new AbortController()
    this.controller = controller
    this.state = 'playing'
    this.error = ''
    this.notify()
    this.operation = (async () => {
      let current
      let failed = false
      try {
        for (let i = 0; i < this.path.count; i++) {
          const started = performance.now()
          const next = await this.decode(await this.store.get(i))
          if (controller.signal.aborted || generation !== this.generation) {
            next.close()
            break
          }
          if (i) await this.wait(Math.max(0, 1000 / this.path.fps - (performance.now() - started)))
          if (controller.signal.aborted) {
            next.close()
            break
          }
          current?.close()
          current = next
          display(next, this.path.at(i))
        }
      } catch (error) {
        if (error.name !== 'AbortError') {
          this.reportError(error, 'play')
          failed = true
        }
      } finally {
        current?.close()
        if (failed) {
          const store = this.store
          this.store = null
          this.path = null
          this.progress = 0
          await store?.dispose().catch((error) => {
            this.reportError(error, 'cleanup')
          })
        }
        this.state = this.store ? 'ready' : 'idle'
        this.notify()
      }
    })()
    await this.operation
  }
}

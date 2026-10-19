/** This file is part of Mandelbrot Formula Explorer. Licensed under GPL-3.0. */
import { getRenderPointBatchLimit } from './buddhabrotRenderConfig.mjs'

export const CONTINUOUS_SAMPLES = Infinity

// One native job retains workers/pipelines and cumulative density until Stop.
export class BuddhabrotContinuousRunner {
  constructor({ runner, createCpuRunner, onProgress, onChunk, onComplete }) {
    Object.assign(this, { runner, createCpuRunner })
    this.onProgress = onProgress || (() => {})
    this.onChunk = onChunk || (() => {})
    this.onComplete = onComplete || (() => {})
    this.continuous = true
    this.useGpu = typeof runner.devicePromise !== 'undefined'
    for (const key of ['width', 'height', 'maxIter', 'palette', 'brightness', 'gamma']) this[key] = runner[key]
    this.samples = CONTINUOUS_SAMPLES
    this.running = false
    this.totalSamples = 0
    this.generation = 0
  }
  get devicePromise() { return this.useGpu ? this.runner.devicePromise : undefined }
  get maxRenderPointBatchSize() { return getRenderPointBatchLimit(CONTINUOUS_SAMPLES) }
  setRenderSpeed(delay) {
    this.renderDelay = delay
    ;(this.runner._cpuRunner || this.runner).setRenderSpeed?.(delay)
  }
  setRenderPointBatchSize(size) {
    this.renderPointBatchSize = size
    ;(this.runner._cpuRunner || this.runner).setRenderPointBatchSize?.(size)
  }
  _syncDensity() {
    const native = this.runner._cpuRunner || this.runner
    if (this.runner._cpuRunner) this.useGpu = false
    for (const key of ['densityR', 'densityG', 'densityB']) this[key] = native[key]
  }
  _report(delta = 0, force = false) {
    this.totalSamples += Math.max(0, delta)
    this.onProgress({ delta, total: this.totalSamples, continuous: true, force })
  }
  async start(params) {
    if (this.running) return
    const generation = ++this.generation
    this.running = true
    this.totalSamples = 0
    this.started = false
    this.finished = false
    for (const key of ['width', 'height', 'maxIter', 'palette', 'brightness', 'gamma', 'renderDelay', 'renderPointBatchSize']) {
      if (params[key] !== undefined) this[key] = params[key]
    }
    this._report(0, true)
    try {
      if (this.useGpu && !(await this.runner.devicePromise) && this.createCpuRunner) {
        if (!this.running || generation !== this.generation) return
        this.runner.stop?.()
        const cpu = await this.createCpuRunner()
        if (!this.running || generation !== this.generation) {
          cpu.stop?.(); cpu.terminate?.()
          return
        }
        this.runner = cpu
        this.useGpu = false
      }
      if (!this.running || generation !== this.generation) return
      const current = () => generation === this.generation && !this.finished
      this.finished = false
      this.runner.onProgress = data => {
        if (!current()) return
        this._syncDensity()
        this._report(Number(data?.delta) || 0)
      }
      this.runner.onChunk = (chunk, presented) => {
        if (!current()) return
        this._syncDensity()
        this.onChunk(chunk, presented)
      }
      this.runner.onComplete = result => {
        if (!current()) return
        this.finished = true
        this.running = false
        this._syncDensity()
        this._report(0, true)
        this.onComplete(result)
      }
      this.started = true
      const task = this.runner.start({ ...params, continuous: true, samples: CONTINUOUS_SAMPLES,
        renderDelay: this.renderDelay, renderPointBatchSize: this.renderPointBatchSize })
      this._syncDensity()
      Promise.resolve(task).catch(error => this._fail(generation, error))
    } catch (error) { this._fail(generation, error) }
  }
  _fail(generation, error) {
    if (generation !== this.generation || this.finished) return
    this.terminate()
    this._report(0, true)
    this.onComplete({ width: this.width, height: this.height, error })
  }
  stop() {
    if (!this.running) return
    this.running = false
    this.runner.stop?.()
    this.runner._cpuRunner?.stop?.()
    this._report(0, true)
    // GPU Stop reads submitted work back asynchronously; CPU has already merged
    // all delivered chunks and its stop intentionally sends no completion event.
    if (!this.useGpu || !this.started) {
      const generation = this.generation
      queueMicrotask(() => {
        if (generation !== this.generation || this.finished) return
        this.finished = true
        this._syncDensity()
        this.onComplete({ width: this.width, height: this.height, aborted: true })
      })
    }
  }
  terminate() {
    this.running = false
    this.generation++
    this.runner.stop?.()
    this.runner._cpuRunner?.terminate?.()
    this.runner.terminate?.()
  }
}
